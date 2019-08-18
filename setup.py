from setuptools import setup, find_packages

from torchpack import __version__

setup(
    name='torchpack',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/mit-han-lab/torchpack',
    install_requires=[
        'numpy>=1.14',
        'six',
        'tqdm',
        'termcolor>=1.1',
        'torch>=1.2',
        'torchvision>=0.4'
    ]
)
