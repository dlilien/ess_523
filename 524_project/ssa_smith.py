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

import sys
sys.path.append('..')
sys.path.append('../lib')
import equationsFEM
import classesFEM
from glib3 import gtif2mat_fn
import numpy as np
from scipy.interpolate import RectBivariateSpline

yearInSeconds=365.25*24.0*60*60 # This will be convenient for units


class velocityDEMs:
    """Make a class which returns the thickness at a point"""
    def __init__(self):
        u_fn='/users/dlilien/smith/Velocities/1990s/tiffs/mosaicOffsets_x_vel.tif'
        v_fn='/users/dlilien/smith/Velocities/1990s/tiffs/mosaicOffsets_y_vel.tif'

        # Make spline objects
        x,y,u=gtif2mat_fn(u_fn)
        u[np.isnan(u)]=0.0
        self.uspline=RectBivariateSpline(np.flipud(y),x,np.flipud(u))
        x,y,v=gtif2mat_fn(v_fn)
        v[np.isnan(v)]=0.0
        self.vspline=RectBivariateSpline(np.flipud(y),x,np.flipud(v))
        
        print('Velocity Spline Objects Prepared')
        

    def __call__(self,pt):
        return np.array([self.uspline(pt[1],pt[0])[0],self.vspline(pt[1],pt[0])[0]])


class thickDEM:
    """Make a function which will return the thickness at a point"""
    def __init__(self):
        bb_fn='/users/dlilien/smith/bed_data/ZBgeo.tif'
        x,y,a=gtif2mat_fn(bb_fn)
        a[np.isnan(a)]=0
        self.aspline=RectBivariateSpline(y,x,a)
        surf_fn='/users/dlilien/smith/bed_data/dshean/smoothed_combination.tif'
        x,y,b=gtif2mat_fn(surf_fn)
        self.bspline=RectBivariateSpline(np.flipud(y),x,np.flipud(b))


    def __call__(self,pt):
        return self.bspline(pt[1],pt[0])[0]-self.aspline(pt[1],pt[0])[0]


class surfDEM:
    """Return the surface, cheat by getting it from the pre-formed thickness DEM"""


    def __init__(self,thickDEM):
        self.bspline=thickDEM.bspline


    def __call__(self,pt):
        return self.bspline(pt[1],pt[0])[0]


def dzs(mesh,surface):
    """Calculate the surface slope on a mesh using nodal values and basis functions"""
    # note that this function could be repeatedly re-called for a time-dependent simulation

    # associate a thickness with every node
    for node in mesh.nodes.values():
        node.surf=surface([node.x,node.y])

    # associate a 2d slope with every mesh point
    for element in mesh.elements.values():
        element.dzs=np.sum([mesh.nodes[node].surf*np.array(element.dbases[i]) for i,node in enumerate(element.nodes)],0)


def nu(nlmodel,velocity,B_0=None,temp=lambda x: -10.0,n=3.0,max_val=1.0e32):
    """Calculate the viscosity of ice given a velocity field and a temperature

    Remember viscosity is a function of strain rate not velocity, so we need to
    do some calculating of gradients (definitely do this with finite elements since
    we get the previous velocity on the grid points with FEM)

    Sets the values on the elements of the nlmodel.

    Parameters
    ----------
    nlmodel : classesFEM.NonLinearModel
        The model we are finding the viscosity for
    velocity : array
        The previous solution
    B_0: int, optional
        Fixed viscosity parameter. Defaults to None, just uses standard functions.
    temp: function, optional
        A function which returns temperature as a function of position. Defaults to -10.0
    n: float, optional
        Exponent in Glen's flow law. Defaults to 3.0.
    max_val: float, optional
        Return this if the strain is zero (causes division error). Defaults to 1.0e32
    """

    # save some typing for things we will need to use a lot
    elements=nlmodel.model.mesh.elements

    # We are going to calculate the viscosity element-wise since this is how we have
    # the velocity gradients. The gradients are piecewise constant, so we don't need
    # to do anything fancy with Gauss Points.

    for element in elements.values():
        du=np.sum([velocity[2*(number-1)]*np.array(element.dbases[index]) for index,number in enumerate(element.nodes)],0)
        dv=np.sum([velocity[2*number-1]*np.array(element.dbases[index]) for index,number in enumerate(element.nodes)],0)
        if not hasattr(element,'b'):
            if B_0 is None:
                element.b=getArrheniusFactor(np.sum([gpt[0]*temp(element.F(gpt[1:])) for gpt in element.gpoints]))
            else:
                element.b=B_0
        element.nu=visc(du,dv,element.b,n=n,max_val=max_val)


def getArrheniusFactor(temp):
    """ Get the temperature-dependent factor for Glen's Flow Law.

    Will just give viscosity at 0 degrees if temperature is above zero.

    Parameters
    ----------
    Temp: float
       The temperature in celcius

    Returns
    -------
       The prefactor B: float
    """
    if temp<-10:
        return 3.985e-13*np.exp(-60.0e3/(8.314*(273.15+temp)))
    elif temp<=0:
        return 1.916e3*np.exp(-139.0e3/(8.314*(273.15+temp)))
    else:
        return 1.916e3*np.exp(-139.0e3/(8.314*(273.15)))




def visc(du,dv,B,n=3.0,max_val=1.0e32):
    """The actual viscosity formula, called by nu
    
    Returns
    -------
    Viscosity: float
    """
    return min((2*B*(2.0*(du[0]**2.0+dv[1]**2.0+0.25*(du[1]+dv[0])**2.0+du[0]*dv[1]))**((n-1.0)/(2.0*n))),max_val)


def main():
    """Do an SSA approximation on Smith Glacier, steady state

    Returns
    -------
    nlmodel: classesFEM.NonLinearModel
        The model object
    """

    # Make some splines of a few different properties

    #velocity for comparison
    vdm=velocityDEMs()
    #thickness for computation
    thick=thickDEM()
    # surface to calculate slope later (could finite-difference now, but let's be 
    # finite element-y
    zs=surfDEM(thick)

    # Create our model
    model=classesFEM.Model('floatingsmith.msh')

    # The model now has a mesh associate (mesh.model) to which we attach properties
    # which are not going to change during the simulation (i.e. everything but
    # viscosity

    # surface slope
    dzs(model.mesh,zs)

    # Add some equation properties to the model
    model.add_equation(equationsFEM.shallowShelf(g=-9.8*yearInSeconds**2,rho=917.0/(1.0e6*yearInSeconds**2)))

    # Grounded boundaries, done lazily since 2,4 are not inflows so what do we do?
    model.add_BC('dirichlet',10,vdm)
    model.add_BC('dirichlet',2,vdm)
    model.add_BC('dirichlet',4,vdm)

    # Boundary conditions for the cutouts
    for cut in [8,60,81]:
        model.add_BC('dirichlet',cut,vdm)

    # Getting dicey. Hopefully this is stress-free
    for shelf in [6,38]: # smith and kohler respectively
        # let's be lazy right now
        model.add_BC('dirichlet',shelf,vdm)
        #model.add_BC('neumann',shelf,lambda x: [0.0,0.0])




    # Now set the non-linear model up to be solved
    nlmodel=model.makeIterate()

    nlmodel.iterate(nu,h=thick,b=0.1,nl_maxiter=100,nl_tolerance=1.0e-8,method='GMRES')


    nlmodel.plotSolution(show=True)
    nlmodel.plotSolution(show=True,threeD=False,vel=True,cutoff=5000.0)
    return nlmodel


def test():
    model=classesFEM.Model('smalltestmesh.msh')
    model.add_equation(equationsFEM.shallowShelf(g=-9.8))
    model.add_BC('dirichlet',1,lambda x:[10.0,10.0])
    #model.add_BC('dirichlet',3,lambda x:[10.0,10.0])
    #model.add_BC('dirichlet',2,lambda x:[10.0,10.0])
    #model.add_BC('dirichlet',4,lambda x:[10.0,10.0])
    #model.add_BC('neumann',1,lambda x:[0.0,0.0])
    model.add_BC('neumann',3,lambda x:[0.0,0.0])
    model.add_BC('neumann',2,lambda x:[0.0,0.0])
    model.add_BC('neumann',4,lambda x:[0.0,0.0])
    surf = lambda x : 2.0
    dzs(model.mesh,lambda x:0.0)

    nlmodel=model.makeIterate()
    nlmodel.iterate(nu,b=1.0e10,h=lambda x: 1.0,relaxation=1.0,nl_tolerance=1.0e-8)
    nlmodel.plotSolution(show=True)
    nlmodel.plotSolution(show=True,threeD=False,vel=True)
    print(nlmodel.linMod.matrix.toarray())
    return nlmodel


if __name__=='__main__':
    test()
    #main()
