from setuptools import setup

setup(
    name='DifferentiableSketching',
    version='0.0.1',
    packages=['dsketch'],
    url='github.com/jonhare/DifferentiableSketching',
    license='Apache 2.0',
    author='Jonathon Hare <jsh2@soton.ac.uk>, Daniela Mihai <adm1g15@soton.ac.uk>',
    author_email='jsh2@soton.ac.uk',
    description='PyTorch differentiable rasterisation for lines, points and curves',
    entry_points={
        'console_scripts': ['imageopt=dsketch.experiments.imageopt.imageopt:main',
                            'autoencoder-experiment=dsketch.experiments.characters.autoencoder:main',
                            'train_classifiers=dsketch.experiments.classifiers.train:main',
                            'evaluate_one_shot=dsketch.experiments.classifiers.evaluate_one_shot:main']
    }
)
