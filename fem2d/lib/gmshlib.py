#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@dlilienMBP>
#
# Distributed under terms of the MIT license.

"""
gmsh support functions
"""
import numpy as np


def shp_to_xy(in_file):
    """Unintelligent extraction of 2d shapefile data. Points need x,y fields of their coords"""
    import shapefile
    sf = shapefile.Reader(in_file)
    rec = sf.records()
    n = len(rec)
    exterior = np.empty([n, 2])
    for i in range(0, n):
        exterior[i, 0] = rec[i][1]
        exterior[i, 1] = rec[i][2]
    return exterior


def gmsh_outline(fname, outline, outlc):
    """Write points to a file which then can be meshed"""
    # This is going to make a somewhat messy mesh in terms of numbering. Oh well
    fid = open(fname, 'w')
    fid.write('lc={:4.2f};\n'.format(outlc.min()))
    formatspec = 'p{:d}=newp; Point(p{:d})={{{:4.3f}, {:4.3f}, 0.0, {:4.3f}}};\n'
    for i in range(0, len(outline)):
        fid.write(
            formatspec.format(i + 1, i + 1, outline[i, 0], outline[i, 1], outlc[i, 0]))
    lx = len(outline)
    fid.write('s{:d}=newreg; Spline(s{:d})={{p{:d}'.format(1, 1, 1))
    for j in range(0, lx - 1):
        fid.write(',p{:d}'.format(j + 2))
    fid.write(',p{:d}'.format(1))
    fid.write('};\n')
    fid.write(
        'pl{:d}=newreg; Physical Line(pl{:d})={{s{:d}}};\n'.format(1, 1, 1))

    # lineloop
    fid.write('ll1=newreg; Line Loop(ll1)={s1')
    fid.write('};\n')

    fid.write('ps1=newreg; Plane Surface(ps1)={ll1')
    fid.write('};\n')
    fid.write('ps2=newreg; Physical Surface(ps2)={ps1};\n')
    fid.close()
    return


def gmsh_outline_shp(out_fn, in_fn, lc):
    """convenience wrapper for the other two functions"""
    outline = shp_to_xy(in_fn)
    lcs = np.zeros([len(outline), 1])
    lcs[lcs == 0] = lc
    gmsh_outline(out_fn, outline, lcs)

