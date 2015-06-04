#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
These are a bunch (hopefully) of equations that the FEM code can solve
"""

import numpy as np

class Equation:
    """Class for equations. Really just make  a function API, eqns only need __call__"""
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
    def __init__(self):
        self.lin=True
        self.dofs=1
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,kwargs={}):
        """Let's solve the diffusion equation"""
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
    """Shallow shelf equation, I hope"""

    def __init__(self,g=9.8,rho=917.0,**kwargs):
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

        # Some optional parameters

        # I need beta to be a scalar, so only use this with constant beta
        if 'beta' in kwargs:
            self.b = kwargs['beta']
        elif 'b' in kwargs:
            self.b = kwargs['b']
     
        if 'thickness' in kwargs:
            self.thickness = kwargs['thickness']
        elif 'h' in kwargs:
            self.thickness = kwargs['h']


    def __call__(self,node1,node2,elements,max_nei=12,rhs=False,kwargs={}):
        """Be careful on what the return is for 2d"""
        # We need basal friction in kwargs, call this b or beta
        # need the thickness, called h or thickness
        # need gravity (not hard coded for unit flexibility) called g
        # need ice density (again not hard codes for unit flexibility) called rho
        # need viscosity, call it nu or visc

        # Check for required inputs
        if hasattr(self,'b'):
            b=self.b
        else:
            # We would like b to be assigned element-wise and then to force it piecewise-constant
            # Deal with complexity on the model end for now?
            if 'b' in kwargs:
                b = kwargs['b']
            else:
                raise RuntimeError('SSA needs basal friction input (b or beta kwarg)')

        if hasattr(self,'thickness'):
            h=self.thickness
        else:
            if 'thickness' in kwargs:
                h = kwargs['thickness']
            elif 'h' in kwargs:
                h = kwargs['h']
            else:
                raise RuntimeError('Need ice thickness (thickness or h kwarg) for SSA')

        if hasattr(self,'dzs'):
            dzs=self.dzs
        else:
            if 'dzs' in kwargs:
                dzs = kwargs['dzs']
            else:
                raise AttributeError('Need surface (dzs kwarg) for SSA')

        if 'gradient' in kwargs:
            nu = kwargs['gradient']
        elif 'nu' in kwargs:
            nu = kwargs['nu']
        elif 'visc' in kwargs:
            nu = kwargs['visc']
        else:
            raise AttributeError('Need viscosity (gradient, nu, or visc kwarg) for SSA')

        # We are going to have 4 returns for the lhs, so set up a sport to receive this info
        ints=np.zeros((max_nei,4))

        # Now loop through the neighboring elements
        for i,elm in enumerate(elements):


            # The indices of each node within the element, to tell us which weights and bases to use

            n1b=elm[1].nodes.index(node1) 
            # this should be the index of the weight/equation (j direction in the supplement)

            n2b=elm[1].nodes.index(node2)
            # this is the index of the basis function (i direction in the supplement)

            #gps=[(gp[0],elm[1].F(gp[1:])) for gp in elm[1].gpts]
            # indices based on a 2x2 submatrix of A for i,j
            # 1,1
            ints[i,0]=elm[1].area*(-b**2+h*nu*(4*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]))
            # 2,2
            ints[i,1]=elm[1].area*(-b**2+h*nu*(4*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]))
            # 1,2
            ints[i,2]=elm[1].area*nu*h*(2*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1])
            # 2,1
            ints[i,3]=elm[1].area*nu*h*(2*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0])

        if rhs:
            # TODO the integrals, check for more parameters?
            ints_rhs=np.zeros((max_nei,2))
            for i,elm in enumerate(elements):
                # 1st SSA eqn (d/dx) rhs
                ints_rhs[i,0]=self.rho*self.g*h*dzs*elm[1].area
                # 2nd SSA eqn (d/dy) rhs
                ints_rhs[i,1]=self.rho*self.g*h*dzs*elm[1].area
            
            # return with rhs
            return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3]),np.sum(ints_rhs[:,0]),np.sum(ints_rhs[:,1])

        # return if rhs is false
        return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3])
