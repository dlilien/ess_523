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
from lib import equations
from lib import classes
from lib.glib3 import gtif2mat_fn
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
        """Calculate velocity

        Parameters
        ----------
        pt : array
           The x,y coordinate of the location at which to calculate the value

        Returns
        -------
        velocity : array
           The vector velocity
        """

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


class betaDEM:
    """Make a function to return the inverted friction parameter from elmer
    
    Parameters
    ----------
    beta_n : string,optional
       The file containing the geotiff of beta values
       """
    def __init__(self,
            beta_fn='/Users/dlilien/smith/elmer/auto/1990s_adjoint_ffnewbenbl_05tau/stress1/beta.tif'):

        x,y,beta=gtif2mat_fn(beta_fn)
        beta[np.isnan(beta)]=0
        self.spline=RectBivariateSpline(np.flipud(y),x,np.flipud(beta))


    def __call__(self,pt):
        return self.spline(pt[1],pt[0])[0]


def dzs(mesh,surface):
    """Calculate the surface slope on a mesh using nodal values and basis functions"""
    # note that this function could be repeatedly re-called for a time-dependent simulation

    # associate a thickness with every node
    for node in mesh.nodes.values():
        node.surf=surface([node.x,node.y])

    # associate a 2d slope with every mesh point
    for element in mesh.elements.values():
        element.dzs=np.sum([mesh.nodes[node].surf*np.array(element.dbases[i]) for i,node in enumerate(element.nodes)],0)


class nuDEM:
    """Class for doing the viscosity calculation

    critical_shear_rate : float,optional
        Minimum shear rate for shear calculations
    B_0: int, optional
        Fixed viscosity parameter. Defaults to None, just uses standard functions.
    temp: function, optional
        A function which returns temperature as a function of position. Defaults to -10.0
    n: float, optional
        Exponent in Glen's flow law. Defaults to 3.0.
    max_val: float, optional
        Return this if the strain is zero (causes division error). Defaults to 1.0e32
    """
    def __init__(self,critical_shear_rate=1.0e-09,B_0=None,temp=lambda x: -10.0,n=3.0):
        self.critical_shear_rate=critical_shear_rate
        self.B_0=B_0
        self.temp=temp
        self.n=n
    
    
    def __call__(self,nlmodel,velocity,*args,**kwargs):
        """Calculate the viscosity of ice given a velocity field and a temperature

        Remember viscosity is a function of strain rate not velocity, so we need to
        do some calculating of gradients (definitely do this with finite elements since
        we get the previous velocity on the grid points with FEM)

        Sets the values on the elements of the nlmodel.

        Parameters
        ----------
        nlmodel : classes.NonLinearModel
            The model for which we are finding the viscosity
        velocity : array
            The previous solution
        """

        # save some typing for things we will need to use a lot
        elements=nlmodel.model.mesh.elements

        # We are going to calculate the viscosity element-wise since this is how we have
        # the velocity gradients. The gradients are piecewise constant, so we don't need
        # to do anything fancy with Gauss Points.

        for element in elements.values():
            du=np.sum([velocity[2*(number-1)]*np.array(element.dbases[index]) for index,number in enumerate(element.nodes)],0)
            dv=np.sum([velocity[2*number-1]*np.array(element.dbases[index]) for index,number in enumerate(element.nodes)],0)
            if not hasattr(element,'_b'):
                if self.B_0 is None:
                    element._af=getArrheniusFactor(np.sum([gpt[0]*self.temp(element.F(gpt[1:])) for gpt in element.gpoints]))
                else:
                    element._af=self.B_0
            element.nu=visc(du,dv,element._af,n=self.n,critical_shear_rate=self.critical_shear_rate)
        print('Average viscosity is ',np.average([elm.nu for elm in elements.values()]),end='  ')


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


def testnu(nlmodel,*args,**kwargs):
    elements=nlmodel.model.mesh.elements

    # We are going to calculate the viscosity element-wise since this is how we have
    # the velocity gradients. The gradients are piecewise constant, so we don't need
    # to do anything fancy with Gauss Points.

    for element in elements.values():
        element.nu=1.0e12


def visc(du,dv,af,n=3.0,critical_shear_rate=1.0e9*yearInSeconds):
    """The actual viscosity formula, called by nu
    
    Returns
    -------
    Viscosity: float
    """


    # Get the coefficient
    pref=(yearInSeconds*af)**(-1.0/3.0)*1.0e-6
    strainRate=du[0]**2.0+dv[1]**2.0+0.25*(du[1]+dv[0])**2.0+du[0]*dv[1]
    if strainRate<critical_shear_rate:
        strainRate=critical_shear_rate
    return pref*strainRate**(-(n-1.0)/(2*n))


class tempDEM:
    """Use some lapse rates and a surface DEM to calculate temperature
    
    Coordinates must be in Antarctic Polar Stereographic, or you need to write a new function to calculate latitude
    Parameters
    ----------
    surf : function
        Surface height as a function of height, temperature
    lat_lapse : float,optional
        Lapse rate per degree of latitude
    alt_lapse : float,optional
        Lapse rater per meter of elevation
    base : float,optional
        Temperature at the equator at 0 degrees
    """
    def __init__(self,surf,lat_lapse=0.68775,alt_lapse=9.14e-3,base=34.36):
        self.ll = lat_lapse
        self.al = alt_lapse
        self.surf=surf
        self.base = base
    def __call__(self, pt):
        """ Return the temperature in Celcius

        Parameters
        ----------
        pt : array
           The coordinates of the point (x,y)

        Returns
        -------
        temp : float
           Temperature in degrees C
        """

        lat=(-np.pi/2.0 + 2.0 * np.arctan(np.sqrt(pt[0]**2.0 + pt[1]**2.0)/(2.0*6371225.0*0.97276)))*360.0/(2.0*np.pi)
        return self.base  - self.ll * abs(lat) - self.al * self.surf(pt)


def main():
    """Do an SSA approximation on Smith Glacier, steady state

    Returns
    -------
    nlmodel: classes.NonLinearModel
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
    # inverted beta
    beta=betaDEM()
    # surface temperature
    temp=tempDEM(zs)
    # basic viscosity class
    nu=nuDEM(temp=temp)

    # Create our model
    model=classes.Model('floatingsmith.msh')

    # The model now has a mesh associate (mesh.model) to which we attach properties
    # which are not going to change during the simulation (i.e. everything but
    # viscosity

    # surface slope
    dzs(model.mesh,zs)

    # Add some equation properties to the model
    model.add_equation(equations.shallowShelf(g=-9.8*yearInSeconds**2,rho=917.0/(1.0e6*yearInSeconds**2),b=beta,thickness=thick))

    # Grounded boundaries, done lazily since 2 are not inflows so what do we do?
    model.add_BC('dirichlet',2,vdm)
    model.add_BC('dirichlet',6,vdm)
    model.add_BC('dirichlet',38,vdm)

    # Boundary conditions for the cutouts
    for cut in [10,60,81]:
        model.add_BC('dirichlet',cut,vdm)

    # Getting dicey. Hopefully this is stress-free
    for shelf in [4,8]: # Crosson and Dotson respectively
        model.add_BC('neumann',shelf,lambda x: [0.0,0.0])

    # Now set the non-linear model up to be solved
    nlmodel=model.makeIterate()

    nlmodel.iterate(nu,relaxation=0.8,nl_maxiter=10,nl_tolerance=1.0e-8,method='CG')


    #nlmodel.plotSolution(show=True)
    #nlmodel.plotSolution(show=True,threeD=False,vel=True,x_steps=200,y_steps=200,cutoff=7000.0)
    return nlmodel


def test():
    model=classes.Model('testmesh.msh')
    model.add_equation(equations.shallowShelf(g=10.0,rho=1000.0))
    model.add_BC('dirichlet',1,lambda x:[0.1,0.0])
    model.add_BC('dirichlet',3,lambda x:[0.2,0.0])
    #model.add_BC('dirichlet',2,lambda x:[10.0,10.0])
    #model.add_BC('dirichlet',4,lambda x:[10.0,10.0])
    #model.add_BC('neumann',1,lambda x:[0.0,0.0])
    #model.add_BC('neumann',3,lambda x:[0.0,0.0])
    model.add_BC('neumann',2,lambda x:[0.0,0.0])
    model.add_BC('neumann',4,lambda x:[0.0,0.0])
    #surf = lambda x : 2.0
    dzs(model.mesh,lambda x: 0.0)

    nlmodel=model.makeIterate()
    nlmodel.iterate(testnu,b=0.0,h=lambda x: 1.0,relaxation=1.0,nl_tolerance=1.0e-8)
    nlmodel.plotSolution(show=True)
    #nlmodel.plotSolution(show=True,threeD=False,vel=True)
    return nlmodel


if __name__=='__main__':
    #test()
    main()
