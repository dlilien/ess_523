#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@dlilienMBP>
#
# Distributed under terms of the MIT license.

"""
Some information, and compile the cython code
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup( name='funniest',
      version='0.1a1',
      description='The funniest joke in the world',
      url='http://github.com/dlilien/fem_2d',
      author='David Lilien',
      author_email='dal22@uw.edu',
      license='MIT',
      packages=['fem_2d']
      )
