import setuptools
from setuptools import setup

setup(name='sigma_graph',
      version='0.1.1',
      description='A graph-based multi-agent environment scenario for skirmish simulations',
      url='https://github.com/o101010o/Figure8Squad',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy', 'gym>=0.17.0', 'networkx', 'matplotlib'],
      python_requires='>=3.6',
      )
