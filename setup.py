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

from setuptools import setup

setup( name='fem2d',
      version='0.1a1',
      description='A simple finite elment solver',
      url='http://github.com/dlilien/fem_2d',
      author='David Lilien',
      author_email='dal22@uw.edu',
      license='MIT',
      install_requires = ['numpy', 'scipy', 'matplotlib'],
      packages=['fem2d','fem2d.core','fem2d.lib'],
      test_suite = 'nose.collector'
      )
