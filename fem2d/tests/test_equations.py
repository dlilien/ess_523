#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
unittests on each equation
"""

import numpy as np
import unittest
from fem2d.core.classes import Model,LinearModel
from fem2d.core.equations import *
from os import path


class TestEqautions(unittest.TestCase):
    def test_diffusion(self):
        mo=Model(path.join(path.split(__file__)[0],'test_lib/testmesh.msh'))
        mo.add_equation(diffusion())
        mo.add_BC('dirichlet',1,lambda x: 10.0)
        mo.add_BC('neumann',2,lambda x:-1.0) # 'dirichlet',2,lambda x: 10.0)
        mo.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
        mo.add_BC('neumann',4,lambda x:0.0)
        m=mo.makeIterate()
        m.iterate()
        self.assertTrue(True)

    def test_advection_diffusion(self):
        admo=Model(path.join(path.split(__file__)[0],'test_lib/testmesh.msh'))
        admo.add_equation(advectionDiffusion())
        admo.add_BC('dirichlet',1,lambda x: 15.0)
        admo.add_BC('neumann',2,lambda x:0.0) # 'dirichlet',2,lambda x: 10.0)
        admo.add_BC( 'dirichlet',3,lambda x: 5.0)
        admo.add_BC('neumann',4,lambda x:0.0)
        am=LinearModel(admo)
        am.iterate(v=lambda x:np.array([1.0,0.0]))
        self.assertTrue(True)

if __name__=='__main__':
    unittest.main(buffer=True)
