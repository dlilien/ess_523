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

from .glib3 import gtif2mat_fn
import numpy as np
from scipy.interpolate import RectBivariateSpline


class Raster:
    """A class for importing geotifs for interpolation
    
    Parameters
    ----------
    *DEMs : list of strings
        The DEMs we should be acting upon

    Keyword Arguments
    -----------------
    subtract : bool
       If True, __call__ returns the first minus the second DEM
    ndv : dict
       Dictionary of no data values. Listed as dem number (0 or 1), then a string (e.g. <0.0)
    """
    def __init__(self,*DEMs,**kwargs):
        print('Initializing DEM for ',*DEMs)
        if len(DEMs)==0:
            raise RuntimeError('You need to give at least one raster name')
        elif len(DEMs)>2:
            raise RuntimeError('Cannot handle more than 2 raster names')
        else:
            dems=[gtif2mat_fn(DEM) for DEM in DEMs]

        for i,dem in enumerate(dems):
            dem[2][np.isnan(dem[2])]=0.0
            if 'ndv' in kwargs:
                if i in kwargs['ndv']:
                    dem[2][eval('dem[2]'+kwargs['ndv'][i])]=0.0



        def splinify(dem):
            try:
                spline=RectBivariateSpline(dem[1],dem[0],dem[2])
            except:
                spline=RectBivariateSpline(np.flipud(dem[1]),dem[0],np.flipud(dem[2]))
            return spline

        self.spline1=splinify(dems[0])
        self.splines=1
        if len(dems)==2:
           self.spline2=splinify(dems[1])
           self.splines=2


        if 'subtract' in kwargs:
            self.subtract = kwargs['subtract']
        else:
            self.subtract = False


    def __call__(self,pt):
        if self.splines==1:
            return self.spline1(pt[1],pt[0])[0]
        else:
            if self.subtract:
                return self.spline1(pt[1],pt[0])[0]-self.spline2(pt[1],pt[0])[0]
            else:
                return np.array([self.spline1(pt[1],pt[0])[0],self.spline2(pt[1],pt[0])[0]])


    def spline(self,pt,number=1):
        if number==1:
            return self.spline1(pt[1],pt[0])[0]
        elif number==2:
            try:
                return self.spline2(pt[1],pt[0])[0]
            except AttributeError:
                raise ValueError('Object has only 1 DEM associated')
        else:
            raise ValueError('Object does not have this many DEMs associated')


def rasterizeMesh(mesh,raster,var_names,elementwise=True):
    """Interpolate the raster onto the mesh

    Parameters
    ----------
    mesh: py:`classes.Mesh`
       The mesh to interpolate onto
    raster: Raster
       The raster which we are applying
    varnames: list of strings
       The names of the variables to put into phys_vars dictionaries
    elementwise: bool,optional
       Apply this to the elements. If false, apply nodewise. Default True.
    """
    if not len(var_names)==raster.splines:
        raise ValueError('Number of variable names does not match number of splines')

    if elementwise:
        for elm in mesh.elements.values():
            if elm.eltypes==2:
                if raster.splines==1:
                    elm.phys_vars[var_names[0]]=elm.area*2*np.sum([gp[0]*raster(gp[1]) for gp in elm.gpts])
                else:
                    elm.phys_vars[var_names[0]],elm.phys_vars[var_names[1]]=elm.area*2*np.sum(np.array([gp[0]*raster(gp[1]) for gp in elm.gpts]),0)

    else:
        for node in mesh.nodes.values():
            if raster.splines==1:
                node.phys_vars[var_names[0]]=raster(node.coords)
            else:
                node.phys_vars[var_names[0]],node.phys_vars[var_names[1]]=raster(node.coords)
            
               
