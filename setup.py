from setuptools import setup, find_packages

from torchpack import __version__

setup(
    name='torchpack',
    version=__version__,
    packages=find_packages(exclude=['examples', 'tests']),
    url='https://github.com/mit-han-lab/torchpack',
    install_requires=[
        'numpy>=1.14',
        'six',
        'tensorboardX>=1.8',
        'termcolor>=1.1',
        'torch>=1.2',
        'torchvision>=0.4',
        'tqdm>=4.31.0'
    ]
)
