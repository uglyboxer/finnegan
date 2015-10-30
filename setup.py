# try:
#     from setuptools import setuptools
# except ImportError:
# from distutils.core import setup


import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


config = {
    'description': 'An exetensible neural net',
    'author': 'Cole Howard',
    'url': 'https://uglyboxer.github.io/finnegan',
    'download_url': 'https://github.com/uglyboxer/linear_neuron',
    'author_email': 'uglyboxer@gmail.com',
    'version': '0.1',
    'install_requires': required,
    'packages': ['finnegan'],
    'scripts': [],
    'name': 'finnegan'
}

setup(**config)