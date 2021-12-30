from setuptools import setup

setup(
    name='DifferentiableSketching',
    version='0.0.1',
    packages=['dsketch', 'dsketch.losses', 'dsketch.utils', 'dsketch.raster', 'dsketch.models', 'dsketch.datasets', 'dsketch.experiments.shared', 'dsketch.experiments', 'dsketch.experiments.imageopt', 'dsketch.experiments.characters', 'dsketch.experiments.characters.models'],
    url='github.com/jonhare/DifferentiableSketching',
    license='BSD 3-Clause',
    author='Jonathon Hare <jsh2@soton.ac.uk>, Daniela Mihai <adm1g15@soton.ac.uk>',
    author_email='jsh2@soton.ac.uk',
    description='PyTorch differentiable rasterisation for lines, points and curves',
    entry_points={
        'console_scripts': ['imageopt=dsketch.experiments.imageopt.imageopt:main',
                            'autoencoder-experiment=dsketch.experiments.characters.autoencoder:main',
                            'train_classifiers=dsketch.experiments.classifiers.train:main',
                            'train_classifiers_barlow=dsketch.experiments.classifiers.train_barlow:main',
                            'evaluate_one_shot=dsketch.experiments.classifiers.evaluate_one_shot:main']
    }
)
