#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
These are a bunch (hopefully) of equations that the FEM code can solve, and the base class for these equations
"""

import numpy as np
cimport numpy as np

class Equation:
    """Class for equations. Really just make a callable function API.

    Besides the attributes which follow, every equation must be callable. The :py:meth:`__call__` method must take four positional arguments (which are probably needed for any finite element solution) and should probably also accept one keyword arguments which are generally needed. Additional arguments can be passed as kwargs. The parent mesh can be accessed via :py:attr:`Element.parent`. The required/suggested arguments, and required returns are:

    Parameters
    ----------
    node1 : int
       The number of the node corresponding to the basis function
    node2 : int
       The number of the node corresponding to the weight/test function
    elements : list
       A list of elements, as 2-tuples of (element_number,:py:class:`classesFEM.Element`), which are shared in common between the two nodes. This is only triangular elements. Linear elements are dealt with by the boundary condition methods of the solver
    rhs : bool
       If True, return a value for the right-hand side of the matrix equation as well. This is necessary to get the returns correct. In general, the right hand side portion will likely be a straightforward integration of the basis function for node1 against the source term.
    max_nei : int,optional 
       The amount of space to allocate for the element-wise integrals, should be the largest number of neighbors any node has times the number of degrees of freedom. Recommended default between 12 and 24.

    Returns
    -------
    integrals : float if 1D, 4-tuple if 2D
       In 1D return the coefficient for the node1, node2 coefficient of the FEM matrix. In 2D, return the ((z1,z1),(z2,z2),(z1,z2),(z2,z1)) coefficients for node1,node2.
    rhs : float if 1D, 2-tuple if 2D
       Only should be called on the diagonal, so return the node1-th rhs value in 1D.  Should be a 2-tuple of the two components in 2D.
    
    Attributes
    ----------
    lin : bool
       True if the equation is linear and false otherwise. Should be overridden by subclass.
    dofs : int
       The number of degrees of freedom of the variable for which we are solving. Is 1 unless overridden.
    """
    # equations must override lin to be boolean and have a call method
    def __init__(self):
        self.lin=None
        self.dofs=1


class area(Equation):
    
    def __init__(self):
        self.lin=True
        self.dofs=1
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,kwargs={}):
        """This is really just for testing. Calculate area"""
        return np.sum([elm[1].area for elm in elements])


class diffusion(Equation):
    def __init__(self):
        self.lin=True
        self.dofs=1
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,kwargs={}):
        """Let's solve the diffusion equation"""
        if 'k' in kwargs:
            k=kwargs['k']
        else:
            k=lambda x:1.0
        ints=np.zeros((max_nei,))
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1)
            n2b=elm[1].nodes.index(node2)
            ints[i]=elm[1].area*np.sum([gp[0]*(elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1])*k(elm[1].F(gp[1:])) for gp in elm[1].gpoints]) 
        if rhs:
            if 'f' in kwargs:
                f=kwargs['f']
            else:
                f=lambda x:0.0
            ints_rhs=np.zeros((max_nei,))
            for i,elm in enumerate(elements):
                ints_rhs[i]=elm[1].area*np.sum([gp[0]*elm[1].bases[n1b](elm[1].F(gp[1:]))*f(elm[1].F(gp[1:])) for gp in elm[1].gpoints])
            return np.sum(ints),np.sum(ints_rhs)
        else:
            return np.sum(ints)


class advectionDiffusion(Equation):
    """ Defines the advection-diffusion equation"""
    def __init__(self):
        self.lin=True
        self.dofs=1
    def __call__(self,int node1,int node2,list elements,int max_nei=8,rhs=False,**kwargs):
        """Solve the advection-diffusion equation
        
        Keyword Arguments
        -----------------
        k : array or float
            The value of the diffusion coefficient. It can be an array if you want anisotropy (it will be dotted/multiplied with the gradient of the temperature variable). If scalar, it is just multiplied. Defaults to 1.0.
        v : function
            Required. The velocity field as a function of space. Should accept a length 2 array (x,y) as an argument and return a length 2 array (vx,vy).

        """
        if 'k' in kwargs:
            if type(kwargs['k'](elements[0][1].F(elements[0][1].gpoints[0][1:])))==float:
                k=lambda x: kwargs['k']*np.array([1.0, 1.0])
            else:
                k=kwargs['k']
        else:
            k=lambda x:np.array([1.0, 1.0])

        if 'v' in kwargs:
            v=kwargs['v']
        elif 'u' in kwargs:
            v=kwargs['u']
        elif 'V' in kwargs:
            v=kwargs['V']
        else:
            raise RuntimeError('You cannot have advection/diffusion without giving me a velocity. Use diffusion')

        ints=np.zeros((max_nei,))
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1)
            n2b=elm[1].nodes.index(node2)
            ints[i]=elm[1].area*np.sum([gp[0]*(np.dot(np.r_[elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0],elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]].flatten(),k(elm[1].F(gp[1:])))+
                np.dot(np.r_[elm[1].dbases[n1b]].flatten(),elm[1].bases[n2b](elm[1].F(gp[1:]))*v(elm[1].F(gp[1:])))) for gp in elm[1].gpoints]) 
        if rhs:
            if 'f' in kwargs:
                f=kwargs['f']
            else:
                f=lambda x:0.0
            ints_rhs=np.zeros((max_nei,))
            for i,elm in enumerate(elements):
                ints_rhs[i]=elm[1].area*np.sum([gp[0]*elm[1].bases[n1b](elm[1].F(gp[1:]))*f(elm[1].F(gp[1:])) for gp in elm[1].gpoints])
            return np.sum(ints),np.sum(ints_rhs)
        else:
            return np.sum(ints)


class shallowShelf(Equation):
    """Shallow shelf equation over a map-view domain. Buggy.
    
    Parameters
    ----------
    g : float,optional
       The value of gravity. Allows for unit flexibility. Defaults to 9.8.
    rho : float,optional
       The density of ice. Defaults to 917

    Keyword Arguments
    -----------------
    b : float
       The value of the friction coefficient. Can be a float or a function. I use elemental average values. 
    thickness : function
       A function to give the thickness of the ice. Needs to accept a length two vector as an argument and return a scalar. Only set it here if you don't need it to change (i.e. steady state, or fixed domain time-dependent)
    """


    def __init__(self,float g=9.8,float rho=917.0,b=lambda x: 0.0,**kwargs):
        """Need to set the dofs"""
        # nonlinear, 2 dofs, needs gravity and ice density (which I insist are constant scalars)
        self.lin=False
        self.dofs=2
        if not type(g)==float:
            raise TypeError('Gravity must be a float')
        self.g=g
        if not type(rho)==float:
            raise TypeError('Density of ice must be a float')
        self.rho=rho

        # Some optional parameters:
        self.b = b
     
        if 'h' in kwargs:
            self.thickness=None
            self.h=kwargs['h']
        elif 'thickness' in kwargs:
            self.thickness = kwargs['thickness']
            self.h=None
        else:
            self.h=None
            self.thickness=None 


    def __call__(self,int node1,int node2,list elements,int max_nei=12,rhs=False,**kwargs):
        """Attempt to solve the shallow-shelf approximation.


        The elements passed to this method must each have a property :py:attr:`dzs` which is a 2-vector which is the surface slope on that element. They also must have a function for viscosity, :py:attr:`nu` associated with them which is a scalar. Since both of these are not spatially variable within an element using piecewise linear basis functions, they should just be values not functions.

        Parameters
        ----------
        node1 : int
           The number of the node corresponding to the basis function
        node2 : int
           The number of the node corresponding to the weight/test function
        elements : list
           A list of elements, as 2-tuples of (element_number,:py:class:`classesFEM.Element`), which are shared in common between the two nodes.
        rhs : bool
           If True, return a value for the right-hand side of the matrix equation as well. This is necessary to get the returns correct. In general, the right hand side portion will likely be a straightforward integration of the basis function for node1 against the source term.
        max_nei : int,optional 
           The amount of space to allocate for the element-wise integrals, should be the largest number of neighbors any node has times the number of degrees of freedom. Recommended default between 12 and 24.

        Keyword Arguments
        -----------------
        h : function
           Required if thickness is not set. The ice thickness as a function of space. Should accept a 2-vector of x,y coordinates and return a float.
        b : float
           Required if not set when the equation instance is created. Square root of basal friction coefficient.
        """
        # We need basal friction in kwargs, call this b or beta
        # need the thickness, called h or thickness
        # need gravity (not hard coded for unit flexibility) called g
        # need ice density (again not hard codes for unit flexibility) called rho
        # need viscosity, call it nu or visc

        # Check for required inputs
        if not np.all([hasattr(elm[1],'b') for elm in elements]):
            for elm in elements:
                elm[1].b=np.average([self.b(elm[1].parent.nodes[node].coords()) for node in elm[1].nodes])

        if not hasattr(elements[0][1],'dzs'):
            raise AttributeError('No surface slope associated with mesh, need tuple/array')


        if not hasattr(elements[0][1],'nu'):
            raise AttributeError('No element-wise viscosity associated with mesh')


        # We are going to have 4 returns for the lhs, so set up a sport to receive this info
        ints=np.zeros((max_nei,4))

        # Now loop through the neighboring elements
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1) 
            # this should be the index of the weight/equation (j direction in the supplement)
            n2b=elm[1].nodes.index(node2)
            # this is the index of the basis function (i direction in the supplement)
            
            # calculate the gauss points once only
            gps=[(gp[0],elm[1].F(gp[1])) for gp in elm[1].gpts]
            
            # thickness being passed gets precedence
            if 'thickness' in kwargs:
                elm[1].h=np.sum([gp[0]*(kwargs['thickness'](gp[1])) for gp in gps])

            # For first time through if constant thickness
            if not hasattr(elm[1],'h'):
                if self.h is not None:
                    elm[1].h=self.h
                elif self.thickness is not None:
                    for elm in elements:
                        elm[1].h=np.sum([gp[0]*(self.thickness(gp[1])) for gp in gps])
                else:
                    raise AttributeError('No thickness found')


            # indices based on a 2x2 submatrix of A for i,j
            # 1,1
            ints[i,0]=2*elm[1].area*(elm[1].b**2+elm[1].h*elm[1].nu*(4*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]))
            # 2,2
            ints[i,1]=2*elm[1].area*(elm[1].b**2+elm[1].h*elm[1].nu*(4*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]))
            # 1,2
            ints[i,2]=2*elm[1].area*(elm[1].b**2*(1.0+(n1b==n2b))/24.0+elm[1].nu*elm[1].h*(2*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]))
            # 2,1
            ints[i,3]=2*elm[1].area*(elm[1].b**2*(1.0+(n1b==n2b))/24.0+elm[1].nu*elm[1].h*(2*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]))

        if rhs:
            # TODO the integrals, check for more parameters?
            ints_rhs=np.zeros((max_nei,2))
            for i,elm in enumerate(elements):
                # 1st SSA eqn (d/dx) rhs
                ints_rhs[i,0]=self.rho*self.g*elm[1].h*elm[1].dzs[0]*elm[1].area
                # 2nd SSA eqn (d/dy) rhs
                ints_rhs[i,1]=self.rho*self.g*elm[1].h*elm[1].dzs[1]*elm[1].area
            
            # return with rhs
            return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3]),np.sum(ints_rhs[:,0]),np.sum(ints_rhs[:,1])

        # return if rhs is false
        return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3])
