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
    *DEMs : string
        The DEMs we should be acting upon

    Keyword Arguments
    -----------------
    subtract : bool
       If True, __call__ returns the first minus the second DEM
    """
    def __init__(self,*DEMs,**kwargs):
        print('Initializing DEMs')
        if len(DEMs)==0:
            raise RuntimeError('You need to give at least one raster name')
        elif len(DEMs)>2:
            raise RuntimeError('Cannot handle more than 2 raster names')
        else:
            dems=[gtif2mat_fn(DEM) for DEM in DEMs]

        for dem in dems:
            dem[2][np.isnan(dem[2])]=0.0


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
            print('1')
            return self.spline1(pt[1],pt[0])[0]
        else:
            if self.subtract:
                return self.spline1(pt[1],pt[0])[0]-self.spline2(pt[1],pt[0])[0]
            else:
                return np.array([self.spline1(pt[1],pt[0])[0],self.spline2(pt[1],pt[0])[0]])
