import setuptools
from setuptools import setup

setup(name='attention_routing',
      version='0.1.1',
      description='A package required for this combinatorial opt RL study. More info on this specific package at the github repo; more info on this study in the README.',
      url='https://github.com/wouterkool/attention-learn-to-route.git',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy', 'torch>=1.7', 'tqdm', 'tensorboard_logger', 'matplotlib'],
      python_requires='>=3.6',
      )
