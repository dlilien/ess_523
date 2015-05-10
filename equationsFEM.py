#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
These are a bunch (hopefully) of equations that the FEM code can solve
"""

import numpy as np

def area(**kwargs):
    """This is really just for testing. Calculate area"""
    return np.sum(kwargs['areas'])

def diffusion(**kwargs):
    """Let's solve the diffusion equation"""
    if 'k' in kwargs.keys():
        k=kwargs['k']
    else:
        k=lambda x: 1
    return np.sum(k)

        



