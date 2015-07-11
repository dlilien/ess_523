#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
Test the time dependent model with diffusion.
"""

import unittest
from ..core.classes import Model,TimeDependentModel
from ..core.equations import diffusion
from os import path

class TestTimeDependent(unittest.TestCase):
    def test_td_diffusion(self):
        mod=Model(path.join(path.split(__file__)[0],'testmesh.msh'),td=True)
        mod.add_equation(diffusion())
        mod.add_BC('dirichlet',1,lambda x,t: 26.0)
        mod.add_BC('neumann',2,lambda x,t:0.0)
        mod.add_BC( 'dirichlet',3,lambda x,t: 26.0)
        mod.add_BC('neumann',4,lambda x,t:0.0)
        mi=TimeDependentModel(mod,10.0,2,lambda x:1+(x[0]-5)**2)
        mi.animate(show=False,save='decay.mp4')
        self.assertTrue(True)

if __name__=='__main__':
    unittest.main()
