#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dlilien90@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Some functions for use with map view shallow shelf approximations
"""

import numpy as np
yearInSeconds=365.25*24.0*60.0*60.0

class nu:
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
    units : string,optional
        Must be MPaA or PaS. I.e. do you want to scale nicely for numerics? Default is MPaA.   
        """
    def __init__(self,critical_shear_rate=1.0e-09,B_0=None,temp=lambda x: -10.0,n=3.0,units='MPaA'):
        self.critical_shear_rate=critical_shear_rate
        self.B_0=B_0
        self.temp=temp
        self.n=n
        self.units=units
    
    
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
            element.phys_vars['nu']=visc(du,dv,element._af,n=self.n,critical_shear_rate=self.critical_shear_rate,units=self.units)
        print('Average viscosity is {:e}'.format(float(np.average([elm.phys_vars['nu'] for elm in elements.values()]))),end=' ')


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
        return np.exp(-60.0e3/8.314*(1.0/273.15+1.0/(273.15+temp)))
    elif temp<0:
        return np.exp(-115.0e3/8.314*(1.0/273.15+1.0/(273.15+temp)))
    else:
        return np.exp(-115.0e3/8.314*(1.0/273.15))


class lapse_tempDEM:
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


def visc(du,dv,af,n=3.0,critical_shear_rate=1.0e-9,units='MPaA'):
    """The actual viscosity formula, called by nu

    Parameters
    ----------
    du : array
        vector derivative of u
    dv : array
        vector derivative of v
    critical_shear_rate : float,optional
        if not None, return the viscosity at this rate if the shear is lower. Default is 1.0e-9
    units : string,optional
        Must be MPaA or PaS. I.e. do you want to scale nicely for numerics? Default is MPaA.
    
    Returns
    -------
    Viscosity: float
    """


    # Get the coefficient
    if units == 'MPaA':
        pref=(3.5e-25*af)**(-1.0/n)*yearInSeconds**(-(1.0)/n)*1.0e-6
    elif units == 'PaS':
        pref=(3.5e-25*af)**(-1.0/n)
    else:
        raise ValueError('Units must be MPaA or PaS')


    strainRate=float(du[0]**2.0+dv[1]**2.0+0.25*(du[1]+dv[0])**2.0+du[0]*dv[1])
    if critical_shear_rate is not None:
        if strainRate<critical_shear_rate:
            strainRate=critical_shear_rate
    return pref*strainRate**(-(n-1.0)/(2*n))/2.0


def surfaceSlope(mesh,surface):
    """Calculate the surface slope on a mesh using nodal values and basis functions"""
    # note that this function could be repeatedly re-called for a time-dependent simulation

    # associate a thickness with every node
    for node in mesh.nodes.values():
        node.surf=surface([node.x,node.y])

    # associate a 2d slope with every mesh point
    for element in mesh.elements.values():
        element.phys_vars['dzs']=np.sum([mesh.nodes[node].surf*np.array(element.dbases[i]) for i,node in enumerate(element.nodes)],0)
