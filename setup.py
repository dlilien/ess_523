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

setup( ext_modules = cythonize("lib/*.pyx") )
