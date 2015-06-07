#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dlilien90@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Let's try to do the forward problem for Ken's class!
"""

import sys
sys.path.append('..')
sys.path.append('../lib')
import classesFEM as cfm
import  equationsFEM
import numpy as np
from scipy.interpolate import RectBivariateSpline
from glib3 import gtif2mat_fn
#import matplotlib.pyplot as plt


class velocityDEMs:
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


class accumulationDEM:
    def __init__(self):
        a_fn='/users/dlilien/Documents/GenData/SMB_RACMO2.3_monthly_ANT27_1979_2013/SMB_climate_average_amcoast.tif'
        x,y,a=gtif2mat_fn(a_fn)
        a[np.isnan(a)]=0
        self.aspline=RectBivariateSpline(np.flipud(y),x,np.flipud(a))


    def __call__(self,pt):
        return 1.0e-3*self.aspline(pt[1],pt[0])[0]


class thickDEM:
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


class k:
    def __init__(self,vel):
        self.vel=vel
    def __call__(self,pt):
        v=self.vel(pt)
        return 7.0e3/2.0*np.outer(v,v)/min(1.0,np.linalg.norm(v))


def main(): 
    admo=cfm.Model('fullsmithmesh.msh')
    vel=velocityDEMs()
    diffu=k(vel)
    acc=accumulationDEM()
    admo.add_equation(equationsFEM.advectionDiffusion())
    smbbm=thickDEM()
    admo.add_BC('dirichlet',2,smbbm)
    admo.add_BC('dirichlet',4,smbbm) # 'dirichlet',2,lambda x: 10.0)
    admo.add_BC( 'dirichlet',32,smbbm)
    admo.add_BC('dirichlet',54,smbbm)
    am=cfm.LinearModel(admo)
    am.iterate(v=vel,f=acc,k=diffu)
    am.plotSolution(savefig='figs/wild2D.eps',threeD=False,cutoff=5000.0,x_steps=200,y_steps=200,savesol=True)
    am.plotSolution(savefig='figs/wild3D.eps')


    return am
        





if __name__=='__main__':
    main()
