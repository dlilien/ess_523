#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright David Lilien dlilien90@gmail.com
#
# Distributed under terms of the MIT license.

"""
Test the SSA modeling using an ice sheet like MacAyeal uses in the 1989 book

Requirements
------------
Needs the mesh icesheet.msh in test_lib
"""

from os import path
from fem2d.core import Model, shallowShelf
from fem2d.lib import nu, surfaceSlope

yearInSeconds = 365.24 * 24.0 * 60.0 * 60.0

def test_macayeal_ssa():
    model = Model(path.join(path.split(__file__)[0], 'test_lib/icesheet.msh'))

    # need to attach surface slopes. do this lazily
    surfaceSlope(model.mesh, lambda x: 1000.0)

    # Arrange the equation
    model.add_equation(shallowShelf(g=-9.8 * yearInSeconds**2,
                                    rho=917.0 / (1.0e6 * yearInSeconds**2),
                                    b=lambda x: 0.0,
                                    thickness=lambda x: 1000.0,
                                    method='CG'))
    model.add_BC('dirichlet', 1, inflow)
    model.add_BC('neumann', 2, lambda x: (0.0, 0.0))
    model.add_BC('neumann', 3, lambda x: (0.0, 0.0))
    model.add_BC('neumann', 4, lambda x: (0.0, 0.0))

    # Solve the system
    nlmodel = model.makeIterate()
    nlmodel.iterate(gradient=nu())
    nlmodel.plotSolution(threeD=False,x_steps=200,y_steps=200,savefig=['shelf_x_vel.eps','shelf_y_vel.eps'],cutoff=25000.0)



def inflow(x):
    """
    Function for the inflow boundary on the ice shelf
    """
    if x[0]<0.0:
        return (400.0,0)
    else:
        return (0.0,0.0)


if __name__=='__main__':
    test_macayeal_ssa()
