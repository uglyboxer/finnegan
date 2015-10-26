try:
    from setuptools import setuptools
except ImportError:
    from distutils.core import setup

config = {
    'description': 'An exetensible neural net',
    'author': 'Cole Howard',
    'url': 'https://uglyboxer.github.io/finnegan',
    'download_url': 'https://github.com/uglyboxer/linear_neuron',
    'author_email': 'uglyboxer@gmail.com'
    'version': '0.1',
    'install_requires': ['pytest'],
    'packages': ['finnegan'],
    'scripts': [],
    'name': 'finnegan'
}

setup(**config)