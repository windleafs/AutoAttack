import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = ['numpy']

setup(name='AutoAttack',
      version='1.1',
      description='AutoAttack',
      author='windleaf',
      url='https://gitee.com/windleafs/auto-attack.git',
      keywords='attack',
      packages=find_packages(),
      license='LICENSE',
      install_requires=requires)
