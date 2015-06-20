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

setup( ext_modules = cythonize("lib/*.pyx"), include_dirs = [numpy.get_include()] )
