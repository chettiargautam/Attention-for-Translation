from setuptools import setup, find_packages

"""
This setup.py is used to install the package in the current directory
To install the package, run the following command in the terminal:
>>> python setup.py build develop --user
"""

setup(
    name='attention-for-translation',
    version='0.1',
    packages=find_packages(),
    install_requires = open("requirements.txt").read().splitlines(),
    packages = find_packages(),
)