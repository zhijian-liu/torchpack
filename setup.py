from setuptools import find_packages, setup

from torchpack import __version__

setup(
    name='torchpack',
    version=__version__,
    packages=find_packages(exclude=['examples']),
    author='Zhijian Liu',
    author_email='torchpack@zhijianliu.com',
    url='https://github.com/zhijian-liu/torchpack',
    install_requires=[
        'loguru',
        'multimethod',
        'numpy',
        'pyyaml',
        'torch>=1.5.0',
        'torchvision',
        'tqdm',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['torchpack = torchpack.launch:main'],
    },
)
