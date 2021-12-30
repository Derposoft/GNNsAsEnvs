import setuptools
from setuptools import setup
import sys

if len(sys.argv) != 4:
      print(sys.argv)
      print('usage: python3 setup_package.py package_name package_github_url package_install_reqs_comma_sep.')
      print('example: python3 setup_package.py sigma_graph https://github.com/o101010o/Figure8Squad "numpy,scipy,gym>=0.17.0,networkx,matplotlib"')
      sys.exit()
print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3].split(','))

setup(name=sys.argv[1],
      version='0.0.1',
      description='A package required for this combinatorial opt RL study. More info on this specific package at the github repo; more info on this study in the README.',
      url=sys.argv[2],
      packages=setuptools.find_packages(),
      install_requires=[sys.argv[3].split(',')],
      python_requires='>=3.8',
      )
