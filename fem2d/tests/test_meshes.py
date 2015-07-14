#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
General tests for core/classes.py
"""

import numpy as np
import unittest
from fem2d.core import Model,TimeDependentModel,LinearModel,diffusion
from os import path


class TestLinear(unittest.TestCase):
    def test_linear(self):
        mo=Model(path.join(path.split(__file__)[0],'test_lib/testmesh.msh'))
        mo.add_equation(diffusion())
        mo.add_BC('dirichlet',1,lambda x: 10.0)
        mo.add_BC('neumann',2,lambda x:-1.0) # 'dirichlet',2,lambda x: 10.0)
        mo.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
        mo.add_BC('neumann',4,lambda x:0.0)
        m=mo.makeIterate()
        m.iterate()
        self.assertTrue(True)


class TestTimeDependent(unittest.TestCase):
    def test_td_diffusion(self):
        mod=Model(path.join(path.split(__file__)[0],'test_lib/testmesh.msh'),td=True)
        mod.add_equation(diffusion())
        mod.add_BC('dirichlet',1,lambda x,t: 26.0)
        mod.add_BC('neumann',2,lambda x,t:0.0)
        mod.add_BC( 'dirichlet',3,lambda x,t: 26.0)
        mod.add_BC('neumann',4,lambda x,t:0.0)
        mod.add_IC(lambda x:1+(x[0]-5)**2,eqn_name='Diffusion')
        TimeDependentModel(mod,10.0,2)
        self.assertTrue(True)

if __name__=='__main__':
    unittest.main(buffer=True)




def profile():
    import cProfile

    def do_cprofile(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        return profiled_func

    @do_cprofile
    def upwinding():
        import equations
        """test upwinding"""
        class k:
            def __init__(self,vel,k_old,alpha):
                self.vel=vel
                self.k=k_old
                self.alpha=alpha
            def __call__(self,pt):
                v=self.vel(pt)
                return self.k(pt)+self.alpha*0.5/2.0*np.outer(v,v)/max(1.0e-8,np.linalg.norm(v))

        alpha=3.0
        k_old=lambda x:np.array([[1.0, 0.0],[0.0, 1.0]])
        vel=lambda x: np.array([1000.0,0.0])
        k_up=k(vel,k_old,alpha)

        admo=Model('524_project/test_lib/testmesh.msh')
        admo.add_equation(equations.advectionDiffusion())
        admo.add_BC('dirichlet',1,lambda x: 15.0)
        admo.add_BC('neumann',2,lambda x:0.0) # 'dirichlet',2,lambda x: 10.0)
        admo.add_BC( 'dirichlet',3,lambda x: 5.0)
        admo.add_BC('neumann',4,lambda x:0.0)
        am=LinearModel(admo)
        am.iterate(v=vel,k=k_up)
        #am.plotSolution(savefig='figs/upwinded.eps',show=True)
        am.plotSolution()

    upwinding()


