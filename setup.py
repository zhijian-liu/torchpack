from setuptools import find_packages, setup

from torchpack import __version__

setup(
    name='torchpack',
    version=__version__,
    packages=find_packages(exclude=['examples']),
    author='Zhijian Liu',
    author_email='zhijianliu.cs@gmail.com',
    url='https://github.com/zhijian-liu/torchpack',
    install_requires=[
        'loguru',
        'numpy',
        'pyyaml',
        'torch>=1.4',
        'torchvision>=0.5',
        'tqdm>=4.31.0',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'torchpack = torchpack.launch:main',
        ],
    },
)
