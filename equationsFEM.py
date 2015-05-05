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


