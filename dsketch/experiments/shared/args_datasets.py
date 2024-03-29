import pathlib
import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, Omniglot, KMNIST

from dsketch.datasets.quickdraw import QuickDrawDataGroupDataset, QuickDrawRasterisePIL
from dsketch.experiments.shared.utils import list_class_names
import os
from torchvision import datasets
from skimage import io, transform

random_seed = 1
torch.manual_seed(random_seed)


def compose(tf, args):
    if 'additional_transforms' in args and args.additional_transforms is not None:
        return transforms.Compose([tf, args.additional_transforms])
    else:
        return tf


def skeleton(image):
    image = np.asarray(image)
    thresh = threshold_otsu(image)
    binary = image > thresh
    # out = binary_closing(skeletonize(binary))
    out = skeletonize(binary)
    return Image.fromarray(out)


def _split(args, trainset):
    vallen_per_class = args.valset_size_per_class
    targets = [trainset[i][1] for i in range(len(trainset))]
    numclasses = len(np.unique(targets))

    train_idx, valid_idx = train_test_split(np.arange(len(trainset)), test_size=vallen_per_class * numclasses,
                                            shuffle=True, stratify=targets, random_state=args.dataset_seed)

    train = torch.utils.data.Subset(trainset, train_idx)
    valid = torch.utils.data.Subset(trainset, valid_idx)

    return train, valid


class _Dataset(ABC):
    @classmethod
    def add_args(cls, p):
        cls._add_args(p)
        p.add_argument("--batch-size", help="batch size", type=int, default=48, required=False)
        p.add_argument("--num-workers", help="number of dataloader workers", type=int, default=4, required=False)

    @staticmethod
    @abstractmethod
    def _add_args(p):
        pass

    @classmethod
    @abstractmethod
    def get_size(cls, args):
        pass

    @classmethod
    def get_channels(cls, args):
        return 1

    @classmethod
    def inv_transform(cls, x):
        """
        This needs to un-invert image tensors so they are in 0..1 ready for saving
        Args:
            x: the input tensor

        Returns:

        """
        return x

    @classmethod
    @abstractmethod
    def create(cls, args):
        pass


class MNISTDataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--valset-size-per-class", help="number of examples to use in validation set per class",
                       type=int, default=10, required=False)
        p.add_argument("--dataset-seed", help="random seed for the train/validation split", type=int,
                       default=1234, required=False)
        p.add_argument("--skeleton", help="Convert each image to its morphological skeleton with a 1px wide stroke",
                       default=False, required=False, action='store_true')

    @classmethod
    def get_transforms(cls, args, train=False):
        if args.skeleton:
            return compose(transforms.Compose([skeleton, transforms.ToTensor()]), args)
        return compose(transforms.ToTensor(), args)

    @classmethod
    def get_size(cls, args):
        return 28

    @classmethod
    def create(cls, args):
        trainset = MNIST(args.dataset_root, train=True, transform=cls.get_transforms(args, True), download=True)
        testset = MNIST(args.dataset_root, train=False, transform=cls.get_transforms(args, False), download=True)

        train, valid = _split(args, trainset)

        return train, valid, testset

    @classmethod
    def inv_transform(cls, x):
        return x


class ScaledMNISTDataset(MNISTDataset):
    @classmethod
    def get_transforms(cls, args, train=False):
        # MNIST preprocess following StrokeNet code
        mnist_resize = 120
        brightness = 0.6
        tf = [
            transforms.Resize(mnist_resize),
            transforms.Pad(int((256 - mnist_resize) / 2)),
            transforms.ToTensor(),
            lambda x: x * brightness
        ]

        if args.skeleton:
            tf.insert(2, skeleton)

        return compose(transforms.Compose(transforms=tf), args)

    @classmethod
    def get_size(cls, args):
        return 256


class OmniglotDataset(_Dataset):

    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--valset-size-per-class", help="number of examples to use in validation set per class",
                       type=int, default=2, required=False)
        p.add_argument("--dataset-seed", help="random seed for the train/validation split", type=int,
                       default=1234, required=False)
        p.add_argument("--augment", help="add augmentation", default=False, required=False, action='store_true')
        p.add_argument("--skeleton", help="Convert each image to its morphological skeleton with a 1px wide stroke",
                       default=False, required=False, action='store_true')

    @classmethod
    def get_size(cls, args):
        return 105

    @classmethod
    def get_transforms(cls, args, train=False):
        tf = [transforms.ToTensor(), transforms.Lambda(lambda x: 1 - x)]
        if train is True and args.augment is True:
            tf.insert(0,
                      transforms.RandomAffine(3.0, translate=(0.07, 0.07), scale=(0.99, 1.01), shear=1, fillcolor=255))

        if args.skeleton:
            tf.insert(0, skeleton)

        return compose(transforms.Compose(tf), args)

    @classmethod
    def inv_transform(cls, x):
        return 1 - x

    @classmethod
    def create(cls, args):
        trainset = Omniglot(args.dataset_root, background=True, transform=cls.get_transforms(args, True), download=True)
        testset = Omniglot(args.dataset_root, background=False, transform=cls.get_transforms(args, False),
                           download=True)

        train, valid = _split(args, trainset)

        return train, valid, testset
    


def _centre_pil_image(pil):
    img = np.array(pil)
    cx = np.expand_dims(np.arange(pil.height), axis=0) - (pil.height // 2)
    cy = np.expand_dims(np.arange(pil.width), axis=1) - (pil.width // 2)

    area = img.sum()
    y_mean = (cy * img).sum() // area
    x_mean = (cx * img).sum() // area

    return pil.transform(pil.size, Image.AFFINE, (1, 0, -x_mean, 0, 1, -y_mean), fillcolor=255)


class C28pxOmniglotDataset(OmniglotDataset):
    @classmethod
    def get_size(cls, args):
        return 28

    @classmethod
    def get_transforms(cls, args, train=False):
        tf = [_centre_pil_image, transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Lambda(lambda x: 1 - x)]

        if train is True and args.augment is True:
            tf.insert(2,
                      transforms.RandomAffine(3.0, translate=(0.07, 0.07), scale=(0.99, 1.01), shear=1, fillcolor=255))

        if args.skeleton:
            tf.insert(2, skeleton)

        return compose(transforms.Compose(tf), args)


class Jon_QuickDrawDataset(_Dataset):

    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--valset-size-per-class", help="number of examples to use in validation set per class",
                       type=int, default=2, required=False)
        p.add_argument("--dataset-seed", help="random seed for the train/validation split", type=int,
                       default=1234, required=False)
        p.add_argument("--augment", help="add augmentation", default=False, required=False, action='store_true')
        p.add_argument("--skeleton", help="Convert each image to its morphological skeleton with a 1px wide stroke",
                       default=False, required=False, action='store_true')
        p.add_argument("--size", help="target image size", required=False, type=int, default=256)
        p.add_argument("--stroke-width", help="initial stroke width before resize", default=16, required=False,
                       type=int)
        p.add_argument("--group", help="qd group name", default="yoga", required=False, type=str)

    @classmethod
    def get_size(cls, args):
        return args.size

    @classmethod
    def inv_transform(cls, x):
        return 1 - x

    @classmethod
    def get_transforms(cls, args, train=False):
        ras = QuickDrawRasterisePIL(True, args.stroke_width)
        tf = [ras, transforms.Resize((args.size, args.size)), transforms.ToTensor()]
        # transforms.Lambda(lambda x: 1 - x)]

        if train is True and args.augment is True:
            tf.insert(2,
                      transforms.RandomAffine(3.0, translate=(0.07, 0.07), scale=(0.99, 1.01), shear=1, fillcolor=255))

        if args.skeleton:
            tf.insert(2, skeleton)

        return compose(transforms.Compose(tf), args)

    @classmethod
    def create(cls, args):
        ds = QuickDrawDataGroupDataset(args.group, max_drawings=70000)
        trainset, testset = torch.utils.data.random_split(ds, [50000, 20000])
        testset, valset = torch.utils.data.random_split(testset, [10000, 10000])

        class WD(Dataset):
            def __init__(self, data, train=False):
                self.tf = cls.get_transforms(args, train)
                self.data = data

            def __getitem__(self, index):
                return self.tf(self.data[index]), 0  # 0 as label indicator

            def __len__(self):
                return len(self.data)

        return WD(trainset, True), WD(valset, False), WD(testset, False)


    
#the Kuzushiji-MNIST Dataset
class KMNISTDataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--valset-size-per-class", help="number of examples to use in validation set per class",
                       type=int, default=10, required=False)
        p.add_argument("--dataset-seed", help="random seed for the train/validation split", type=int,
                       default=1234, required=False)
        p.add_argument("--skeleton", help="Convert each image to its morphological skeleton with a 1px wide stroke",
                       default=False, required=False, action='store_true')

    @classmethod
    def get_transforms(cls, args, train=False):
        if args.skeleton:
            return compose(transforms.Compose([skeleton, transforms.ToTensor()]), args)
        return compose(transforms.ToTensor(), args)

    @classmethod
    def get_size(cls, args):
        return 28
    
    @classmethod
    def inv_transform(cls, x):
        return x   

    @classmethod
    def create(cls, args):
        trainset = KMNIST(args.dataset_root, train=True, transform=cls.get_transforms(args, True), download=True)
        testset = KMNIST(args.dataset_root, train=False, transform=cls.get_transforms(args, False), download=True)

        train, valid = _split(args, trainset)

        return train, valid, testset

     
from PIL import Image

class CustomImageNet(Dataset):
    def __init__(self, split_file, root_dir, transform=None):
        """
        Args:
            split_file (string): Path to the split file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split_file = split_file
        self.root_dir = root_dir
        self.transform = transform
        
        f=open(self.split_file, 'r')
        self.lines_in_file = f.readlines()
        f.close()
        
        self.list_of_items=[]
        for line in self.lines_in_file:
            split_line=line.split()
            self.list_of_items.append(split_line)

    def __len__(self):
        return len(self.lines_in_file)
    
    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        img_name = os.path.join(self.root_dir,
                                self.list_of_items[index][0])
        image = Image.open(img_name)
        if(image.mode!='RGB'):
            image=image.convert('RGB')
        target = torch.tensor(int(self.list_of_items[index][1]))

        if self.transform:
            image = self.transform(image)

        return image, target
    
class StratifiedImageNetDataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("/data/ILSVRC2012/"), required=False)
        p.add_argument("--split", help="split of the dataset: 1 or 2", type=int,
                       default=1, required=True)
        p.add_argument("--image-size", help="size of resampled images", type=int, default=64, required=False)
        p.add_argument("--imagenet-norm", help="normalise data with imagenet statistics", action='store_true',
                       required=False)

    @classmethod
    def get_transforms(cls, args, train=False):

        if train:
            base = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            base = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]

        if args.imagenet_norm:
            base.append(IMAGENET_NORM)

        return compose(transforms.Compose(base), args)

    @classmethod
    def get_size(cls, args):
        return args.image_size

    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def create(cls, args):
        
        traindir = os.path.join(str(args.dataset_root) + '/train/')
        valdir = os.path.join(str(args.dataset_root) + '/val/')
        
        train_splits = os.path.join('/home/adm1g15/DifferentiableSketching/dsketch/datasets/', 'ImageNetSplit_train'+ str(args.split))
        val_splits = os.path.join('/home/adm1g15/DifferentiableSketching/dsketch/datasets/', 'ImageNetSplit_val'+ str(args.split))
        
        trainset = CustomImageNet(train_splits, traindir, cls.get_transforms(args, True))
        valset = CustomImageNet(val_splits, valdir, cls.get_transforms(args, False))
        
#         testdir = os.path.join('/data/ILSVRC2012/' + '/test_dir/')
#         testset = datasets.ImageFolder(testdir, cls.get_transforms(args, False))
        
#         trainset = datasets.ImageFolder(traindir,cls.get_transforms(args, True))
#         valset = datasets.ImageFolder(valdir,cls.get_transforms(args, False))

        print(len(trainset))
        print(len(valset))
#         print(len(testset))


#         trainset = Balancing(trainset, transform=cls.get_transforms(args, True))
#         valset = Transforming(valset, transform=cls.get_transforms(args, False))
#         testset = Transforming(testset, transform=cls.get_transforms(args, False))

        return  trainset, valset, valset #testset #trainset1, trainset2,

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 1000    
    
def get_dataset(name):
    ds = getattr(sys.modules[__name__], name + 'Dataset')
    if not issubclass(ds, _Dataset):
        raise TypeError()
    return ds


def build_dataloaders(args):
    ds = get_dataset(args.dataset)
    args.size = ds.get_size(args)

    train, valid, test = ds.create(args)

    trainloader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valloader = DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    testloader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return trainloader, valloader, testloader


def dataset_choices():
    return [i.replace('Dataset', '') for i in list_class_names(_Dataset, __name__)]
