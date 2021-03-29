"""
Modified Hausdorff distance computation based on code from the omniglot repository:
https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification
"""

import torch


def mod_hausdorff_distance(item1: torch.Tensor, item2: torch.Tensor) -> torch.Tensor:
	"""
	Modified Hausdorff Distance

	M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
	International Conference on Pattern Recognition, pp. 566-568.

	Args:
		item1: [n, 2] coordinates of "inked" pixels
		item2: [m, 2] coordinates of "inked" pixels

	Returns:
		computed distance

	"""
	d = torch.cdist(item1, item2)
	mindist_1, _ = d.min(axis=1)
	mindist_2, _ = d.min(axis=0)
	mean_1 = torch.mean(mindist_1, dim=0)
	mean_2 = torch.mean(mindist_2, dim=0)

	return torch.maximum(mean_1, mean_2)


def binary_image_to_points(img: torch.Tensor, invert=False) -> torch.Tensor:
	"""
	Convert (~binary) image tensor to a list of mean-centred coordinates of the non-zero pixels.

	Args:
		img: [1, H, W] the image tensor
		invert: if true the input image will be inverted (e.g. when strokes are assumed to be zero and background one)

	Returns:
		the mean-centred coordinates

	"""

	if invert:
		img = torch.logical_not(img.squeeze(0))
	else:
		img = img.squeeze(0)

	coords = torch.nonzero(img).float()
	coords = coords - coords.mean(dim=0)

	return coords
