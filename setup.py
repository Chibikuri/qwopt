from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f]

setup(
    name='qwopt',
    version='0.0.1',
    description='Quantum Walk Circuit Optimizer',
    long_description=long_description,
    url='https://github.com/takezyou/pydata',
    author='takezyou',
    author_email='kaitokun07@icloud.com',
    license='Apache-2.0',
    install_requires=['beautifulsoup4', 'lxml'],
    keywords='qwopt',
    packages=find_packages(exclude=('tests')),
    entry_points={
        "console_scripts": [
            "pydata=pydata.__init__:main",
        ],
    },
    classifiers=[
        'License :: OSI Approved :: Apache-2.0 License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)