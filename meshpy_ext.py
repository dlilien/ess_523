#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien90@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Subclass the meshpy class to add conveniences
"""

from meshpy import gmsh_reader
import numpy as np

class gmshNumpy(gmsh_reader.GmshMeshReceiverNumPy):
    def __init__(self,fn):
        super(gmshNumpy,self).__init__()
        gmsh_reader.read_gmsh(self,fn)
    def CreateBases(self):
        for element in self.elements:
            
