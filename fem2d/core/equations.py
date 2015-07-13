#! /usr/bin/env python3
#cython: embedsignature=True
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
These are a few equations that the FEM code can solve, and the base class for these equations
"""

import numpy as np

class Equation:
    """Class for equations. Really just make a callable function API.

    Parameters
    ----------
    relaxation : float,optional
       The amount to relax. Use less than 1 if you have convergence problems. Defaults to 1.0.
    ss_maxiter : int,optional
       The maximum number of steady state iterations allowed. Default 50.
    ss_tolerance : float, optional
       The tolerance in steady state for each equation. Default 1.0e-3.
    nl_tolerance : float,optional
       When to declare things converged
    guess : array,optional
       An initial guess at the solution. If None, use all zeros. Defaults to None.
    nl_maxiter : int,optional
       Maximum number of nonlinear iterations. Defaults to 50.
    method : list of strings,optional
       Solution method to use for the linear system. Defaults to BiCGStab. Done using :py:meth:`modeling.ModelIterate.solveIt`.
    precond : string,optional
       Preconditioning method for the linear system if solved iteratively. Defaults to ILU. Can also be a LinearOperator which does the solving using a preconditioning matrix or matrices.
    lin_tolerance : float,optional
       Linear system convergence tolerance for iterative methods. Defaults to 1.0e-5.
    max_nei : int,optional
       Maximum number of neighboring elements times dofs. Err large. Defaults to 16.


        Attributes
        ----------
        lin : bool
           True if the equation is linear and false otherwise. Should be overridden by subclass.
        dofs : int
           The number of degrees of freedom of the variable for which we are solving. Is 1 unless overridden.
        BCs : dictionary
            The associated boundary conditions, should be attached using :py:meth:`add_BC`
        ICs : function
            Needed for time dependent simluations
        """
    # equations must override lin to be boolean and have a call method
    def __init__(self,dofs=1,lin=True,ss_tolerance=1.0e-3,relaxation=1.0,nl_tolerance=1.0e-5,guess=None,nl_maxiter=50,method='BiCGStab',precond='LU',lin_tolerance=1.0e-5,max_nei=16):
        self.dofs=dofs
        self.lin=lin
        self.ss_tolerance=ss_tolerance
        self.relaxation=relaxation
        self.nl_tolerance=nl_tolerance
        self.guess=guess
        self.nl_maxiter=nl_maxiter
        self.method=method
        self.precond=precond
        self.lin_tolerance=lin_tolerance
        self.max_nei=16
        self.BCs={}
        self.IC=None


    def __call__(self,node1,node2,rhs=False):
        pass
        """
        Find the entry for the matrix to solve

        The :py:meth:`__call__` method must take four positional arguments (which are probably needed for any finite element solution) and should probably also accept one keyword arguments which are generally needed.
        Additional arguments can be passed as kwargs. The parent mesh can be accessed via :py:attr:`Element.parent`.
        The required/suggested arguments, and required returns are:
        
        Parameters
        ----------
        node1 : int
           The number of the node corresponding to the basis function
        node2 : int
           The number of the node corresponding to the weight/test function
        elements : list
           A of elements, as 2-tuples of (element_number,:py:class:`meshes.Element`), which are shared in common between the two nodes. This is only triangular elements. Linear elements are dealt with by the boundary condition methods of the solver
        rhs : bool
           If True, return a value for the right-hand side of the matrix equation as well. This is necessary to get the returns correct. In general, the right hand side portion will likely be a straightforward integration of the basis function for node1 against the source term.

        Returns
        -------
        integrals : if 1D, 4-tuple if 2D
           In 1D return the coefficient for the node1, node2 coefficient of the FEM matrix. In 2D, return the ((z1,z1),(z2,z2),(z1,z2),(z2,z1)) coefficients for node1,node2.
        rhs : if 1D, 2-tuple if 2D
           Only should be called on the diagonal, so return the node1-th rhs value in 1D.  Should be a 2-tuple of the two components in 2D.
        """


class dummy(Equation):
    """Dummy equation for simple debugging."""
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name='Dummy'

    def __call__(self,node1,node2,elements,rhs=False,**kwargs):
        """ Dummy return. If not rhs, returns 1.0. Else returns 1.0,1.0."""
        if rhs:
            return 1.0,1.0
        else:
            return 1.0


class area(Equation):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,kwargs={}):
        """This is really just for testing. Calculate area"""
        return np.sum([elm[1].area for elm in elements])


class diffusion(Equation):
    def __init__(self,**kwargs):
        super().__init__(lin=True,dofs=1,**kwargs)
        self.name='Diffusion'
        
    def __call__(self,node1,node2,elements,rhs=False,**kwargs):
        """Let's solve the diffusion equation"""
        if 'k' in kwargs:
            k=kwargs['k']
        else:
            k=lambda x:1.0
        ints=np.zeros((self.max_nei,))
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1)
            n2b=elm[1].nodes.index(node2)
            ints[i]=elm[1].area*np.sum([gp[0]*(elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1])*k(elm[1].F(gp[1:])) for gp in elm[1].gpoints]) 
        if rhs:
            if 'f' in kwargs:
                f=kwargs['f']
            else:
                f=lambda x:0.0
            ints_rhs=np.zeros((self.max_nei,))
            for i,elm in enumerate(elements):
                ints_rhs[i]=elm[1].area*np.sum([gp[0]*elm[1].bases[n1b](elm[1].F(gp[1:]))*f(elm[1].F(gp[1:])) for gp in elm[1].gpoints])
            return np.sum(ints),np.sum(ints_rhs)
        else:
            return np.sum(ints)


class advectionDiffusion(Equation):
    """ Defines the advection-diffusion equation"""


    def __init__(self,**kwargs):
        super().__init__(lin=True,dofs=1,**kwargs)
        self.name='Advection-Diffusion'

    def __call__(self,node1,node2,elements,rhs=False,**kwargs):
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
        else:
            raise RuntimeError('You cannot have advection/diffusion without giving me a velocity. Use diffusion')

        ints=np.zeros((self.max_nei,))
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
            ints_rhs=np.zeros((self.max_nei,))
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
       The value of the friction coefficient. Can be a or a function. I use elemental average values. 
    thickness : function
       A function to give the thickness of the ice. Needs to accept a length two vector as an argument and return a scalar. Only set it here if you don't need it to change (i.e. steady state, or fixed domain time-dependent)
    """


    def __init__( self, b, g=9.8,rho=917.0,**kwargs):
        """Need to set the dofs"""
        # nonlinear, 2 dofs, needs gravity and ice density (which I insist are constant scalars)     
        if 'h' in kwargs:
            self.thickness=None
            self.h=kwargs['h']
            del kwargs['h']
        elif 'thickness' in kwargs:
            self.thickness = kwargs['thickness']
            del kwargs['thickness']
            self.h=None
        else:
            self.h=None
            self.thickness=None 
        super().__init__(dofs=2,lin=False,**kwargs)
        self.name='Shallow Shelf'
        if not type(g)==float:
            raise TypeError('Gravity must be a float')
        self.g=g
        if not type(rho)==float:
            raise TypeError('Density of ice must be a float')
        self.rho=rho
        self.b = b



    def __call__(self,node1,node2,elements,rhs=False,**kwargs):
        """Attempt to solve the shallow-shelf approximation.


        The elements passed to this method must each have a property :py:attr:`dzs` which is a 2-vector which is the surface slope on that element. They also must have a function for viscosity, :py:attr:`nu` associated with them which is a scalar. Since both of these are not spatially variable within an element using piecewise linear basis functions, they should just be values not functions.

        Parameters
        ----------
        node1 : int
           The number of the node corresponding to the basis function
        node2 : int
           The number of the node corresponding to the weight/test function
        elements : list
           A of elements, as 2-tuples of (element_number,:py:class:`meshes.Element`), which are shared in common between the two nodes.
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
        # need viscosity, call it nu

        # Check for required inputs
        if not np.all(['b' in elm[1].phys_vars for elm in elements]):
            for elm in elements:
                elm[1].phys_vars['b']=np.average([self.b(elm[1].parent.nodes[node].coords()) for node in elm[1].nodes])

        if not 'dzs' in elements[0][1].phys_vars:
            raise AttributeError('No surface slope associated with mesh, need tuple/array')


        if not 'nu' in elements[0][1].phys_vars:
            raise AttributeError('No element-wise viscosity associated with mesh')


        # We are going to have 4 returns for the lhs, so set up a sport to receive this info
        ints=np.zeros((self.max_nei,4))

        # Now loop through the neighboring elements
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1) 
            # this should be the index of the weight/equation (j direction in the supplement)
            n2b=elm[1].nodes.index(node2)
            # this is the index of the basis function (i direction in the supplement)
            
            # thickness being passed gets precedence
            if 'thickness' in kwargs:
                elm[1].h=2*np.sum([gp[0]*(kwargs['thickness'](gp[1])) for gp in elm[1].gpts])

            # For first time through if constant thickness
            if not 'h' in elm[1].phys_vars:
                if self.h is not None:
                    elm[1].phys_vars['h']=self.h
                elif self.thickness is not None:
                    for elm in elements:
                        elm[1].phys_vars['h']=2*np.sum([gp[0]*(self.thickness(gp[1])) for gp in elm[1].gpts])
                else:
                    raise AttributeError('No thickness found')


            # indices based on a 2x2 submatrix of A for i,j
            # 1,1
            ints[i,0]=elm[1].area*(elm[1].phys_vars['b']**2+elm[1].phys_vars['h']*elm[1].phys_vars['nu']*(4*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]))
            # 2,2
            ints[i,1]=elm[1].area*(elm[1].phys_vars['b']**2+elm[1].phys_vars['h']*elm[1].phys_vars['nu']*(4*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]))
            # 1,2
            ints[i,2]=elm[1].area*(elm[1].phys_vars['nu']*elm[1].phys_vars['h']*(2*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]))
            # 2,1
            ints[i,3]=elm[1].area*(elm[1].phys_vars['nu']*elm[1].phys_vars['h']*(2*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]))

        if rhs:
            # TODO the integrals, check for more parameters?
            ints_rhs=np.zeros((self.max_nei,2))
            for i,elm in enumerate(elements):
                # 1st SSA eqn (d/dx) rhs
                ints_rhs[i,0]=self.rho*self.g*elm[1].phys_vars['h']*elm[1].phys_vars['dzs'][0]*elm[1].area
                # 2nd SSA eqn (d/dy) rhs
                ints_rhs[i,1]=self.rho*self.g*elm[1].phys_vars['h']*elm[1].phys_vars['dzs'][1]*elm[1].area
            
            # return with rhs
            return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3]),np.sum(ints_rhs[:,0]),np.sum(ints_rhs[:,1])

        # return if rhs is false
        return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3])


class ssaAdjointBeta(Equation):
    """This is to solve the adjoint equation w.r.t. beta for the SSA

    The shallowShelf class is set up with a beta^2 formulation, so I follow that here

    Parameters
    ----------
    guess : function,optional
       A guess at beta for which to start. 0 everywhere if none.
    """

    def __init__(self,beta=lambda x: 0.0,**kwargs):
        super().__init__(dofs=2,lin=True,**kwargs)
        self.name='Shallow Shelf Adjoint'
        
        # Even though this is the same equations as the SSA in some sense, it is linear because viscosity is independent of the Lagrange multipliers
        
        if 'h' in kwargs:
            self.thickness=None
            self.h=kwargs['h']
            del kwargs['h']
        elif 'thickness' in kwargs:
            self.thickness = kwargs['thickness']
            del kwargs['thickness']
            self.h=None
        else:
            self.h=None
            self.thickness=None 


    def __call__(self,node1,node2,elements,rhs=False,**kwargs):
        """Attempt to solve adjoint of the shallow-shelf approximation.


        The elements passed to this method must each have a property :py:attr:`dzs` which is a 2-vector which is the surface slope on that element. They also must have a function for viscosity, :py:attr:`nu` associated with them which is a scalar. Since both of these are not spatially variable within an element using piecewise linear basis functions, they should just be values not functions.

        Parameters
        ----------
        node1 : int
           The number of the node corresponding to the basis function
        node2 : int
           The number of the node corresponding to the weight/test function
        elements : list
           A of elements, as 2-tuples of (element_number,:py:class:`meshes.Element`), which are shared in common between the two nodes.
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
        # need viscosity, call it nu

        # Check for required inputs
        if not np.all(['b' in elm[1].phys_vars for elm in elements]):
            for elm in elements:
                elm[1].phys_vars['b']=np.average([self.b(elm[1].parent.nodes[node].coords()) for node in elm[1].nodes])

        if not 'dzs' in elements[0][1].phys_vars:
            raise AttributeError('No surface slope associated with mesh, need tuple/array')

        if not 'nu' in elements[0][1].phys_vars:
            raise AttributeError('No element-wise viscosity associated with mesh')

        # We are going to have 4 returns for the lhs, so set up a sport to receive this info
        ints=np.zeros((self.max_nei,4))

        # Now loop through the neighboring elements
        for i,elm in enumerate(elements):
            n1b=elm[1].nodes.index(node1) 
            # this should be the index of the weight/equation (j direction in the supplement)
            n2b=elm[1].nodes.index(node2)
            # this is the index of the basis function (i direction in the supplement)
            
            # thickness being passed gets precedence
            if 'thickness' in kwargs:
                elm[1].h=2*np.sum([gp[0]*(kwargs['thickness'](gp[1])) for gp in elm[1].gpts])

            # For first time through if constant thickness
            if not 'h' in elm[1].phys_vars:
                if self.h is not None:
                    elm[1].phys_vars['h']=self.h
                elif self.thickness is not None:
                    for elm in elements:
                        elm[1].phys_vars['h']=2*np.sum([gp[0]*(self.thickness(gp[1])) for gp in elm[1].gpts])
                else:
                    raise AttributeError('No thickness found')


            # indices based on a 2x2 submatrix of A for i,j
            # 1,1
            ints[i,0]=elm[1].area*(elm[1].phys_vars['b']**2+elm[1].phys_vars['h']*elm[1].phys_vars['nu']*(4*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]))
            # 2,2
            ints[i,1]=elm[1].area*(elm[1].phys_vars['b']**2+elm[1].phys_vars['h']*elm[1].phys_vars['nu']*(4*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][0]))
            # 1,2
            ints[i,2]=elm[1].area*(elm[1].phys_vars['nu']*elm[1].phys_vars['h']*(2*elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]+elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]))
            # 2,1
            ints[i,3]=elm[1].area*(elm[1].phys_vars['nu']*elm[1].phys_vars['h']*(2*elm[1].dbases[n1b][1]*elm[1].dbases[n2b][0]+elm[1].dbases[n1b][0]*elm[1].dbases[n2b][1]))

        if rhs:
            # TODO the integrals, check for more parameters?
            ints_rhs=np.zeros((self.max_nei,2))
            for i,elm in enumerate(elements):
                # 1st SSA eqn (d/dx) rhs
                ints_rhs[i,0]=elm[1].phys_vars['u_d']-elm[1].phys_vars['u']
                # 2nd SSA eqn (d/dy) rhs
                ints_rhs[i,1]=elm[1].phys_vars['v_d']-elm[1].phys_vars['v']

            
            # return with rhs
            return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3]),np.sum(ints_rhs[:,0]),np.sum(ints_rhs[:,1])

        # return if rhs is false
        return np.sum(ints[:,0]),np.sum(ints[:,1]),np.sum(ints[:,2]),np.sum(ints[:,3])

