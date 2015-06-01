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
    lin=None


class area(Equation):
    lin=True
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,kwargs={}):
        """This is really just for testing. Calculate area"""
        return np.sum([elm[1].area for elm in elements])


class diffusion(Equation):
    lin=True
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
    lin=True
    def __call__(self,node1,node2,elements,max_nei=8,rhs=False,dim=2,kwargs={}):
        """Let's solve the diffusion equation"""
        if dim==2:    
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



            

            



