#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien90@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Try doing the shallow shelf approximation on Smith Glacier
"""

import fem2d
from fem2d.lib import Raster,nu,surfaceSlope,rasterizeMesh

# global constant for unit conversion
yearInSeconds=365.25*24.0*60.0*60.0 # This will be convenient for units

def main():
    """Do an SSA approximation on Smith Glacier, steady state

    Returns
    -------
    nlmodel: fem2d.NonLinearModel
        The model object
    """

    # Make some splines of a few different properties

    #velocity for comparison
    #vdm=velocityDEMs()
    vdm=Raster('tiffs/mosaicOffsets_x_vel.tif','tiffs/mosaicOffsets_y_vel.tif')


    #thickness raster, zero out bad values
    thick=Raster('tiffs/smoothed_combination.tif','tiffs/ZBgeo.tif',subtract=True,ndv={0:'<0.0',1:'<-4.0e4'})

    # inverted beta
    #beta=Raster('tiffs/beta.tif')
    beta = lambda x: 0.0


    # surface temperature
    temp=Raster('tiffs/temperature.tif')
    # basic viscosity class
    nus=nu(temp=temp)

    # Create our model
    model=fem2d.Model('floatingsmith.msh')

    # The model now has a mesh associate (mesh.model) to which we attach properties
    # which are not going to change during the simulation (i.e. everything but
    # viscosity

    # surface slope
    surfaceSlope(model.mesh,thick.spline)

    # Add some equation properties to the model
    model.add_equation(fem2d.shallowShelf(g=-9.8*yearInSeconds**2,rho=917.0/(1.0e6*yearInSeconds**2),b=beta,thickness=thick,relaxation=1.0,nl_maxiter=50,nl_tolerance=1.0e-5,method='CG'))

    # Grounded boundaries, done lazily since 2 are not inflows so what do we do?
    model.add_BC('dirichlet',2,vdm,eqn_name='Shallow Shelf')
    model.add_BC('dirichlet',6,vdm,eqn_name='Shallow Shelf')
    model.add_BC('dirichlet',38,vdm,eqn_name='Shallow Shelf')

    # Boundary conditions for the cutouts
    for cut in [10,60,81]:
        model.add_BC('dirichlet',cut,vdm,eqn_name='Shallow Shelf')

    # Make a dirichlet condition at the calving front as well for inversion
    for shelf in [4,8]: # Crosson and Dotson respectively
        model.add_BC('dirichlet',shelf,vdm,eqn_name='Shallow Shelf')

    model.add_equation(fem2d.ssaAdjointBeta())
    for edge in [2,4,6,8,10,38,60,81]:
        model.add_BC('dirichlet',edge,lambda x:(0.0,0.0),eqn_name='Shallow Shelf Adjoint')

    # Associate the measured velocities with the mesh
    rasterizeMesh(model.mesh,vdm,['u_d','v_d'],elementwise=True)
    
    multimodel=model.makeIterate()
    multimodel.iterate(gradient={'Shallow Shelf':nus}) # Adjoint equation is linear


    multimodel.models['Shallow Shelf'].plotSolution(threeD=False,vel=True,x_steps=200,y_steps=200,cutoff=7000.0)
    multimodel.models['Shallow Shelf'].plotSolution(target='h',nodewise=False,show=True,threeD=False,vel=True,x_steps=200,y_steps=200,cutoff=7000.0)
    return multimodel


if __name__=='__main__':
    main()
