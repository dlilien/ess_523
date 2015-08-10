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
try:
    from ..core.equations import Function
except ValueError:
    from core.equations import Function
from scipy.optimize import minimize_scalar


yearInSeconds = 365.25 * 24.0 * 60.0 * 60.0


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
    units : string, optional
        Must be MPaA or PaS. I.e. do you want to scale nicely for numerics? Default is MPaA.
    dummy : bool, optional
        Don't put the velocities on the elements, call the viscosity nu_dummy
        """

    def __init__(
            self,
            critical_shear_rate=1.0e-09,
            B_0=None,
            temp=lambda x: -
            10.0,
            n=3.0,
            units='MPaA',
            dummy=False):
        self.critical_shear_rate = critical_shear_rate
        self.B_0 = B_0
        self.temp = temp
        self.n = n
        self.units = units
        self.dummy = dummy

    def __call__(self, nlmodel, velocity, *args, **kwargs):
        """Calculate the viscosity of ice given a velocity field and a temperature

        Remember viscosity is a function of strain rate not velocity, so we need to
        do some calculating of gradients (definitely do this with finite elements since
        we get the previous velocity on the grid points with FEM)

        Sets the values on the elements of the nlmodel.

        Parameters
        ----------
        nlmodel : modeling.NonLinearModel
            The model for which we are finding the viscosity
        velocity : array
            The previous solution
        """

        # save some typing for things we will need to use a lot
        elements = nlmodel.model.mesh.elements

        # We are going to calculate the viscosity element-wise since this is how we have
        # the velocity gradients. The gradients are piecewise constant, so we don't need
        # to do anything fancy with Gauss Points.

        for element in elements.values():
            du = np.sum([velocity[2 * (number - 1)] * np.array(element.dbases[index])
                         for index, number in enumerate(element.nodes)], 0)
            dv = np.sum([velocity[2 * number - 1] * np.array(element.dbases[index])
                         for index, number in enumerate(element.nodes)], 0)
            if not hasattr(element, '_b'):
                if self.B_0 is None:
                    element._af = getArrheniusFactor(
                        np.sum([gpt[0] * self.temp(element.F(gpt[1:])) for gpt in element.gpoints]))
                else:
                    element._af = self.B_0
            if not self.dummy:
                element.phys_vars['u'] = np.average(
                    [velocity[2 * (number - 1)] for index, number in enumerate(element.nodes)])
                element.phys_vars['v'] = np.average(
                    [velocity[2 * number - 1] for index, number in enumerate(element.nodes)])
                nu_name='nu'

            else:
                nu_name='nu_dummy'
            
            element.phys_vars[nu_name] = visc(
                du,
                dv,
                element._af,
                n=self.n,
                critical_shear_rate=self.critical_shear_rate,
                units=self.units)

        print('Average viscosity is {:e}'.format(
            float(np.average([elm.phys_vars[nu_name] for elm in elements.values()]))), end=' ')


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
    if temp < -10:
        return np.exp(-60.0e3 / 8.314 * (1.0 / 273.15 + 1.0 / (273.15 + temp)))
    elif temp < 0:
        return np.exp(-115.0e3 / 8.314 *
                      (1.0 / 273.15 + 1.0 / (273.15 + temp)))
    else:
        return np.exp(-115.0e3 / 8.314 * (1.0 / 273.15))


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

    def __init__(self, surf, lat_lapse=0.68775, alt_lapse=9.14e-3, base=34.36):
        self.ll = lat_lapse
        self.al = alt_lapse
        self.surf = surf
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

        lat = (-np.pi / 2.0 + 2.0 * np.arctan(np.sqrt(pt[0]**2.0 + pt[1]**2.0) / (
            2.0 * 6371225.0 * 0.97276))) * 360.0 / (2.0 * np.pi)
        return self.base - self.ll * abs(lat) - self.al * self.surf(pt)


def visc(du, dv, af, n=3.0, critical_shear_rate=1.0e-9, units='MPaA'):
    """The actual viscosity formula, called by nu

    The actual formula used for PaS is :math:`(3.5\\times10^{-25}\\times a)^{\\frac1n}`
    and for MPaA is :math:`(3.5\\times10^{-25}\\times a\\times \\text{yearinsec})^{\\frac1n}\\times10^{6}`

    where :math:`3.5\\times10^{-25}` is taken from Cuffey and Patterson, :math:`a` is the Arrhenius factor calculated by :py:meth:`getArrheniusFactor` and :math:`n` is the Glen's flow law exponent.


    Parameters
    ----------
    du : array
        vector derivative of u
    dv : array
        vector derivative of v
    af : float
        The Arrhenius factor
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
        pref = (3.5e-25 * af)**(-1.0 / n) * \
            yearInSeconds**(-(1.0) / n) * 1.0e-6
    elif units == 'PaS':
        pref = (3.5e-25 * af)**(-1.0 / n)
    else:
        raise ValueError('Units must be MPaA or PaS')

    strainRate = float(du[0]**2.0 +
                       dv[1]**2.0 +
                       0.25 *(du[1] + dv[0])**2.0 + du[0] * dv[1])
    if critical_shear_rate is not None:
        if strainRate < critical_shear_rate:
            strainRate = critical_shear_rate
    return pref * strainRate**(-(n - 1.0) / (2.0 * n)) / 2.0


def surfaceSlope(mesh, surface):
    """Calculate the surface slope on a mesh using nodal values and basis functions"""
    # note that this function could be repeatedly re-called for a
    # time-dependent simulation

    # associate a thickness with every node
    for node in mesh.nodes.values():
        node.surf = surface([node.x, node.y])

    # associate a 2d slope with every mesh point
    for element in mesh.elements.values():
        element.phys_vars['dzs'] = np.sum(
            [
                mesh.nodes[node].surf *
                np.array(
                    element.dbases[i]) for i,
                node in enumerate(
                    element.nodes)],
            0)


def j_of_c(c,model,solution,gradJ,nu,eqn_name='SSA Dummy',beta_name='b',dummy_beta_name='beta_dummy'):
    """Return the cost as a function of the coefficient to gradJ. For optimization

    Set up so that you pass this function to minimize_scalar. The `args` parameter of `minimize_scalar` should have the `model`, `solution`, and `gradJ` arguments in a 3-tuple.

    Parameters
    ----------
    c : float
       Multiply gradJ by this to optimize. Dealt with behind the scenes.
    model : :py:ref:`modeling.model`
       Need to access varuious nodal and elemental variables
    gradJ : array
       The gradient to the cost function as calculated by OptimizeBeta.
    eqn_name : str, optional 
       The name of the equation for solving the forward problem. Could just use the full-on forward problem but let's not.
    """
    for node in model.mesh.nodes.values():
        node.phys_vars[dummy_beta_name]=c*node.phys_vars[beta_name]
    dummy_model=model.makeIterate(eqn_name)
    dummy_model.eqn.guess=solution['Shallow Shelf']
    sol=dummy_model.iterate(gradient=nu)[eqn_name]
    if not 'U_d' in model.mesh.phys_vars:
        U_d=np.zeros(sol.shape)
        for i,node in model.mesh.nodes.items():
            U_d[2*(i-1)]=node.phys_vars['u_d']
            U_d[2*i-1]=node.phys_vars['v_d']
        model.mesh.phys_vars['U_d']=U_d
    return np.sum((sol-model.mesh.phys_vars['U_d'])**2)


class OptimizeBeta(Function):

    """ This function does the optimization to seek for the solution for beta

    Parameters
    ----------
    optimization : str, optional
       The optimization method to use. Default is trying the gradient, half, 2 times, 1/4, etc. Called tryHalf. Other option is brent of scipy's minimize_scalar.
    ssa_sol_name : str, optional
       Name of the shallow shelf solver. Default 'Shallow Shelf'
    ssa_adjoint_sol_name : str, optional
       Name of the shallow shelf adjoint solver. Default 'Shallow Shelf Adjoint'
    beta : str,optional
       Name of the slip variable
    min_steps : int,optional
       Minimum number of step sizes to try in search for descent direction. Default 0.
    max_steps : int,optional
       Maximum number of steps to use. Default 10.
    """

    def __init__(
            self,
            dummy_viscosity,
            optimization='tryHalf',
            ssa_sol_name='Shallow Shelf',
            ssa_adjoint_sol_name='Shallow Shelf Adjoint',
            beta_name='b',
            min_steps=0,
            max_steps=10):
        self.optimization=optimization
        self.ssan = ssa_sol_name
        self.ssaan = ssa_adjoint_sol_name
        self.beta = beta_name
        self.min_steps = min_steps
        self.max_steps = max_steps
        if not optimization=='exponential':
            self.nu=dummy_viscosity
        self.call = 0

    def __call__(self, mesh, model, solution):
        self.call += 1
        gradJ = np.zeros(mesh.numnodes)
        for node_num in mesh.nodes:
            for el_num in mesh.nodes[node_num].ass_elms:
                elm = mesh.elements[el_num[0]]
                if elm.eltypes == 2:
                    k = elm.nodes.index(node_num)
                    for i in range(3):
                        for j in range(3):
                            if i == j:
                                if i == k:
                                    gradJ[node_num - 1] += 2 * elm.phys_vars[self.beta] * (solution[self.ssan][2 * (elm.nodes[i] - 1)] * solution[self.ssaan][2 * (
                                        elm.nodes[j] - 1)] + solution[self.ssan][2 * elm.nodes[i] - 1] * solution[self.ssaan][2 * elm.nodes[j] - 1]) * elm.area / 10.0
                                else:
                                    gradJ[node_num - 1] += 2 * elm.phys_vars[self.beta] * (solution[self.ssan][2 * (elm.nodes[i] - 1)] * solution[self.ssaan][2 * (
                                        elm.nodes[j] - 1)] + solution[self.ssan][2 * elm.nodes[i] - 1] * solution[self.ssaan][2 * elm.nodes[j] - 1]) * elm.area / 30.0
                            elif j == k or i == k:
                                gradJ[node_num - 1] += 2 * elm.phys_vars[self.beta] * (solution[self.ssan][2 * (elm.nodes[i] - 1)] * solution[self.ssaan][2 * (
                                    elm.nodes[j] - 1)] + solution[self.ssan][2 * elm.nodes[i] - 1] * solution[self.ssaan][2 * elm.nodes[j] - 1]) * elm.area / 30.0
                            else:
                                gradJ[node_num - 1] += 2 * elm.phys_vars[self.beta] * (solution[self.ssan][2 * (elm.nodes[i] - 1)] * solution[self.ssaan][2 * (
                                    elm.nodes[j] - 1)] + solution[self.ssan][2 * elm.nodes[i] - 1] * solution[self.ssaan][2 * elm.nodes[j] - 1]) * elm.area / 60.0

        # Normalize gradJ
        # gradJ = gradJ / np.linalg.norm(gradJ)


        if self.optimization=='exponential':
            # This is probably useless, but leave it in for debugging
            scale = 10.0**float(-self.call)
        elif self.optimization == 'tryHalf':
            if not 'U_d' in model.mesh.phys_vars:
                U_d=np.zeros(solution[self.ssaan].shape)
                for i,node in model.mesh.nodes.items():
                    U_d[2*(i-1)]=node.phys_vars['u_d']
                    U_d[2*i-1]=node.phys_vars['v_d']
                model.mesh.phys_vars['U_d']=U_d
            initial_norm = np.sum((solution[self.ssaan]-model.mesh.phys_vars['U_d'])**2)

            if not self.min_steps==0:
                scales = np.zeros(self.min_steps*2+1)
                scale[0]=1.0
                for i in range(1,self.min_steps):
                    scale[2*i-1] = 1.0 / 2.0**i
                    scale[2*i] = 1.0 * 2.0**i
                norms = np.zeros(self.min_steps) # container for norms
                found = False # Do we have a solution?


            for i in scale in enumerate(scales):
                norms[i] = j_of_c(scale,model,solution,gradJ,self.nu,eqn_name='SSA Dummy',beta_name='b',dummy_beta_name='beta_dummy')
            if not self.min_steps==0:
                best = np.argmin(norms)
                scale = scales[best]
                if best < initial_norm:
                    found = True


            if not found:
                for i in range(self.min_steps,self.max_steps):
                    new_norm = j_of_c(1.0 / 2.0**(float(i)),model,solution,gradJ,self.nu,eqn_name='SSA Dummy',beta_name='b',dummy_beta_name='beta_dummy')
                    if new_norm < initial_norm:
                        scale = 1.0 / 2.0**(float(i))
                        break

                    if not i==0:
                        new_norm = j_of_c(1.0 * 2.0**(float(i)),model,solution,gradJ,self.nu,eqn_name='SSA Dummy',beta_name='b',dummy_beta_name='beta_dummy')
                        if new_norm < initial_norm:
                            scale = 1.0 * 2.0**(float(i))
                            break
            
        elif self.optimization=='brent' or self.optimization=='golden':
            res=minimize_scalar(j_of_c, 
                                args=(model,solution,gradJ,self.nu),
                                method=self.optimization)
            scale=res.x
        print('Optimal scaling found to be ',scale)

        for node_num, node in mesh.nodes.items():
            node.phys_vars[self.beta] = node.phys_vars[self.beta] + \
                    scale * gradJ[node_num - 1]
