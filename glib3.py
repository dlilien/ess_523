#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 dlilien <dlilien90@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Some routines for use with matrices and geotiffs, oriented for use with elmer and qgis
"""
import gdal
import osr
import numpy as np
import os


def LinearInterp(dem, xx, yy, nx, ny, x, y):
    Dx = (xx[nx - 1] - xx[0]) / (nx - 1)
    Dy = (yy[ny - 1] - yy[0]) / (ny - 1)
    DxDy = Dx * Dy

    # lower left point in DEM
    nx_1 = np.floor((x - xx[0]) / Dx) + 1
    ny_1 = np.floor((y - yy[0]) / Dy) + 1
    nx_1 = min(nx_1, nx - 1)
    ny_1 = min(ny_1, ny - 1)

    x_1 = xx[nx_1]
    y_1 = yy[ny_1]

    B = [0, 0, 0, 0]
    # DEM Value in surroundings points
    #       4 ----- 3
    #       |       |
    #       1 ----- 2
    B[0] = dem[nx_1, ny_1]
    B[1] = dem[nx_1 + 1, ny_1]
    B[2] = dem[nx_1 + 1, ny_1 + 1]
    B[3] = dem[nx_1, ny_1 + 1]

    InterP1 = -500
    if (min(B) != -9999):
        # Linear Interpolation at Point x,y
        InterP1 = (x - x_1) * (y - y_1) * (B[2] + B[0] - B[1] - B[3]) / DxDy
        InterP1 = InterP1 + \
            (x - x_1) * (B[1] - B[0]) / Dx + \
            (y - y_1) * (B[3] - B[0]) / Dy + B[0]
    else:
        for i in [0, 1]:
            for j in [0, 1]:
                dist = max(abs(x - xx(nx_1 + i)), abs(y - yy(ny_1 + j)))
                if (dist <= 0.5 * Dx and dem(nx_1 + i, ny_1 + j) != -9999):
                    InterP1 = dem(nx_1 + i, ny_1 + j)
    return InterP1


def readgeotif(filename, xmin=-np.Inf, xmax=np.Inf, ymin=-np.Inf, ymax=np.Inf):
    import math
    """Fancy geotiff reader, from Daniel"""
    geotiffile = gdal.Open(filename, gdal.GA_ReadOnly)

    # Get the size of the dataset
    nx = geotiffile.RasterXSize
    ny = geotiffile.RasterYSize

    # Load the data from the file
    z = np.zeros((ny, nx))
    z = geotiffile.GetRasterBand(1).ReadAsArray()
    z = z[::-1, :]

    # Get the coordinates of the image
    gt = geotiffile.GetGeoTransform()
    x = np.zeros(nx)
    y = np.zeros(ny)
    for i in range(nx):
        x[i] = gt[0] + i * gt[1]
    for i in range(ny):
        y[i] = gt[3] + i * gt[5]
    y = y[::-1]

    dx = math.fabs(gt[1])
    dy = math.fabs(gt[5])

    j0 = int(max(0, (xmin - x[0]) / dx))
    j1 = int(min(nx, (xmax - x[0]) / dx) + 1)
    i0 = int(max(0, (ymin - y[0]) / dy))
    i1 = int(min(ny, (ymax - y[0]) / dy) + 1)

    return (x[j0:j1], y[i0:i1], z[i0:i1, j0:j1])


def mat2gtif(fn, x, y, z, t_srs='sps'):
    """Simple matrix to geotiff conversion"""
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(fn, len(x), len(y), 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    if t_srs == 'sps':
        srs.ImportFromProj4(
            '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
        # srs.ImportFromEPSG(3031)
        # srs.SetProjection('Polar_Stereographic')
    elif t_srs == 'll':
        srs.SetWellKnownGeogCS('WGS84')
    else:
        print('Unrecognized coordinate reference system, defaulting to EPSG 3031')
        srs.ImportFromProj4(
            '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(z)
    dst_ds.SetGeoTransform(
        [min(x), (max(x) - min(x)) / len(x), 0, y[0], 0, (y[-1] - y[0]) / len(y)])


def mat2gtif_limits(fn, lowerleft, upperright, z, t_srs='sps'):
    """Simple matrix to geotiff conversion"""
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(fn, z.shape[1], z.shape[0], 1, gdal.GDT_Float32)
    dst_ds.GetRasterBand(1).WriteArray(z)
    srs = osr.SpatialReference()
    if t_srs == 'sps':
        srs.ImportFromProj4(
            '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    elif t_srs == 'll':
        srs.SetWellKnownGeogCS('WGS84')
    else:
        print('Unrecognized coordinate reference system, defaulting to EPSG 3031')
        srs.ImportFromProj4(
            '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.SetGeoTransform([lowerleft[0], (upperright[0] - lowerleft[0]) / z.shape[
                           0], 0, upperright[1], 0, (upperright[1] - lowerleft[1]) / z.shape[1]])


def gtif2mat(dst):
    """Turn a geotiff object into matrices, ignore projection hoopla"""
    z = np.array(dst.GetRasterBand(1).ReadAsArray())
    len_y, len_x = z.shape
    dst_gt = dst.GetGeoTransform()
    x0 = dst_gt[0]
    y0 = dst_gt[3]
    xM = dst_gt[1] * len_x + x0
    yM = dst_gt[5] * len_y + y0
    X = np.linspace(x0, xM, len_x)
    Y = np.linspace(y0, yM, len_y)
    return X, Y, z


def gtif2mat_fn(fn):
    """Turn a geotiff file into matrices, ignore projection hoopla"""
    dst = gdal.Open(fn)
    X, Y, z = gtif2mat(dst)
    return X, Y, z


def mat2xyz(fn, X, Y, z):
    """Write a matrix in a form that is easy to use with user functions with Elmer"""
    with open(fn, 'w') as f:
        f.write('%d\n%d\n' % (len(X), len(Y)))
        for i in range(0, len(X)):
            for j in range(0, len(Y)):
                if np.isnan(z[-j - 1, i]):
                    f.write('%9.1f %9.1f %9.5f\n' % (X[i], Y[-j - 1], -2.0e9))
                else:
                    f.write('%9.1f %9.1f %9.5f\n' %
                            (X[i], Y[-j - 1], z[-j - 1, i]))


def mat2xyz_new(fn, X, Y, z, intit=True):
    """Write a matrix in a form that is easy to use with user functions with Elmer"""
    if intit:
        X = list(map(int, X))
        Y = list(map(int, Y))
    with open(fn, 'w') as f:
        f.write('%d\n%d\n' % (len(X), len(Y)))
        for j in range(0, len(Y)):
            for i in range(0, len(X)):
                if np.isnan(z[-j - 1, i]):
                    f.write('%9.1f %9.1f %9.5f\n' % (X[i], Y[-j - 1], -2.0e9))
                else:
                    f.write('%9.1f %9.1f %9.5f\n' %
                            (X[i], Y[-j - 1], z[-j - 1, i]))


def xyz2mat(fn, ndv=-2.0e9):
    """Turn an xyz type file back into a matrix, with no data value = ndv"""
    with open(fn, 'r') as f:
        nx = int(f.readline())
        ny = int(f.readline())
        z = np.zeros([ny, nx])
        y = np.zeros(ny)
        x = np.zeros(nx)
        for i in range(0, nx):
            for j in range(0, ny):
                line = f.readline().split(' ')
                x[i] = float(line[0])
                y[-j - 1] = float(line[1])
                z[-j - 1, i] = float(line[2])
    z[z == ndv] = np.nan
    return x, y, z


def gtif2xyz_new(dst, fn):
    """Turn a gtiff object into a form that is easy to use with user functions with Elmer"""
    X, Y, z = gtif2mat(dst)
    mat2xyz_new(fn, X, Y, z)


def gtif2xyz(dst, fn):
    """Turn a gtiff object into a form that is easy to use with user functions with Elmer"""
    X, Y, z = gtif2mat(dst)
    mat2xyz(fn, X, Y, z)


def gtif2xyz_fn_new(in_fn, out_fn):
    """Turn a gtiff file into a form that is easy to use with user functions with Elmer"""
    dst = gdal.Open(in_fn)
    gtif2xyz_new(dst, out_fn)


def gtif2xyz_fn(in_fn, out_fn):
    """Turn a gtiff file into a form that is easy to use with user functions with Elmer"""
    dst = gdal.Open(in_fn)
    gtif2xyz(dst, out_fn)


def mat2gridIn(fn, X, Y, z, zeroval=np.nan):
    """Put a matrix into a form usable by ElmerGrid"""
    with open(fn, 'w') as f:
        if np.isnan(zeroval):
            print("Writing File for ElmerGrid\nIgnoring NaN Points")
            for i in range(0, len(X)):
                for j in range(0, len(Y)):
                    if not np.isnan(z[-j - 1, i]):
                        f.write('%9.1f %9.1f %9.5f\n' %
                                (X[i], Y[-j - 1], z[-j - 1, i]))
        else:
            print("Writing File for ElmerGrid\n Ignoring points with value %f" % zeroval)
            for i in range(0, len(X)):
                for j in range(0, len(Y)):
                    if not (z[-j - 1, i] == zeroval):
                        f.write('%9.1f %9.1f %9.5f\n' %
                                (X[i], Y[-j - 1], z[-j - 1, i]))


def gtif2xyz_and_gridIn(dst, out_fn, out_grid_fn, zeroval=np.nan):
    X, Y, z = gtif2mat(dst)
    mat2xyz(out_fn, X, Y, z)
    mat2gridIn(out_grid_fn, X, Y, z, zeroval)


def gtif2xyz_and_gridIn_fn(in_fn, out_fn, out_grid_fn, zeroval=np.nan):
    dst = gdal.Open(in_fn)
    gtif2xyz_and_gridIn(dst, out_fn, out_grid_fn, zeroval)


def shp_to_xy(in_file):
    """Unintelligent extraction of 2d shapefile data"""
    import shapefile
    sf = shapefile.Reader(in_file)
    rec = sf.records()
    n = len(rec)
    exterior = np.empty([n, 2])
    for i in range(0, n):
        exterior[i, 0] = rec[i][1]
        exterior[i, 1] = rec[i][2]
    return exterior


def shp_to_xyz(in_file):
    """Unintelligent extraction of 3d shapefile data"""
    import shapefile
    sf = shapefile.Reader(in_file)
    rec = sf.records()
    n = len(rec)
    exterior = np.empty([n, 3])
    for i in range(0, n):
        exterior[i, 0] = rec[i][0]
        exterior[i, 1] = rec[i][1]
        exterior[i, 2] = rec[i][2]
    return exterior


def xy_to_shp(rec, out_file):
    """Convert points to a line shapefile"""
    import shapefile
    sf = shapefile.Writer()
    sf.line(parts=rec)
    sf.field('FIRST_FLD')
    sf.record('test', 'Line')
    sf.save(out_file)


def xy_to_shp_pts(rec, out_file, field_names=None, fields=None):
    """Convert points to a points shapefile"""
    import shapefile
    sf = shapefile.Writer(shapefile.POINT)
    if fields is None:
        sf.field('fld')
    else:
        for field in field_names:
            sf.field(field)
    for i in range(0, len(rec)):
        sf.point(rec[i][0], rec[i][1])
        if fields is None:
            sf.record('test ' + str(i), 'Point')
        else:
            if len(field_names) > 1:
                sf.record(*fields[i])
            else:
                sf.record(fields[i])
    sf.save(out_file)


def xyz_to_shp_pts(rec, out_file, field_names=None, fields=None):
    """Convert points to a points shapefile"""
    import shapefile
    sf = shapefile.Writer(shapefile.POINT)
    if fields is None:
        sf.field('fld')
    else:
        for field in field_names:
            sf.field(field)
    for i in range(0, len(rec)):
        sf.point(rec[i][0], rec[i][1], rec[i][2])
        if fields is None:
            sf.record('test ' + str(i), 'Point')
        else:
            if len(field_names) > 1:
                sf.record(*fields[i])
            else:
                sf.record(fields[i])
    sf.save(out_file)


def write_to_qgis(filename, data, xllcorner, yllcorner, dx, no_data):
    """Not sure if this is still used by anything...delete?"""
    fid = open(filename, 'w')
    (ny, nx) = np.shape(data)

    fid.write('ncols         {0}\n'.format(nx))
    fid.write('nrows         {0}\n'.format(ny))
    fid.write('xllcorner     {0}\n'.format(xllcorner))
    fid.write('yllcorner     {0}\n'.format(yllcorner))
    fid.write('cellsize      {0}\n'.format(dx))
    fid.write('NODATA_value  {0}\n'.format(no_data))

    for i in range(ny - 1, -1, -1):
        for j in range(nx):
            fid.write('{0} '.format(data[i, j]))
        fid.write('\n')

    fid.close()


def sparse2mat(data, x_steps=500, y_steps=500, cutoff_dist=2000.0):
    """Grid up some sparse, concave data"""
    from scipy.spatial import cKDTree as KDTree
    from scipy.interpolate import griddata
    tx = np.arange(min(data[0]), max(data[0]), x_steps)
    ty = np.arange(min(data[1]), max(data[1]), y_steps)
    XI, YI = np.meshgrid(tx, ty)
    ZI = griddata(np.c_[data[0], data[1]], data[2], (XI, YI), method='linear')
    tree = KDTree(np.c_[data[0], data[1]])
    dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
    dist = dist.reshape(XI.shape)
    ZI[dist > cutoff_dist] = np.nan
    return [tx, ty, ZI]


def dat2mat(fn, lines=True, threed=True, reg='davg', cd=3700):
    if not os.path.exists(fn):
        print('Could not find file ' + fn)
        return None
    try:
        f = open(fn)
        if lines:
            f.readline()
        mat = f.readlines()
        data = np.empty(len(mat), dtype=[
                        ('Node Number', int), ('x', float), ('y', float), ('z', float), ('dat', float)])
        f.close()
        for i, line in enumerate(mat):
            data[i] = tuple(map(float, line.split()))
    except:
        print('Could not successfully read dat file ' + fn)
        return None
    mat = None
    data = np.sort(data, order=['x', 'y', 'z'])
    if threed:
        if reg == 'davg':
            data = get_da_vars(data)
        elif reg == 'top':
            data = get_bt_vars(data, bottom=False)
        elif reg == 'bottom':
            data = get_bt_vars(data)
        else:
            print('Region to return not understood. Try again with davg, top, or bottom')
            return None
    mat = sparse2mat([data['x'], data['y'], data['dat']], cutoff_dist=cd)
    return mat


def dat2gtif(dat_fn, gtif_fn=None, lines=True, threed=True, reg='davg', cd=3700):
    if gtif_fn is None:
        direc, fn = os.path.split(dat_fn)
        gtif_fn = direc + '/' + os.path.splitext(fn)[0] + '.tif'
    mat2gtif(
        gtif_fn, *dat2mat(dat_fn, lines=lines, threed=threed, reg=reg, cd=cd))


def sparse2xyz(fn, data, x_steps=500, y_steps=500, cutoff_dist=2000.0):
    """Grid data, write to xyz fn"""
    mats = sparse2mat(
        data, x_steps=x_steps, y_steps=y_steps, cutoff_dist=cutoff_dist)
    print('Writing file ', fn)
    mat2xyz(fn, mats[0], mats[1][::-1], mats[2][::-1, :])


def sparse2xyz_and_tif(fn_xyz, data, fn_tif=None, x_steps=500, y_steps=500, cutoff_dist=2000.0):
    """Grid data, write to a tif and to an xyz file"""
    if fn_tif is None:
        fn, ext = os.path.splitext(fn_xyz)
        fn_tif = fn + '.tif'
    mats = sparse2mat(
        data, x_steps=x_steps, y_steps=y_steps, cutoff_dist=cutoff_dist)
    print('Writing file ', fn_xyz)
    mat2xyz(fn_xyz, mats[0], mats[1][::-1], mats[2][::-1, :])
    print('Writing file', fn_tif)
    mat2gtif(fn_tif, mats[0], mats[1][::-1], mats[2][::-1, :])


def bufcount(filename):
    """Gets the length of a file"""
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read  # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def get_variables(variables, result_fn, apps=[2], maps=[], **kwargs):
    """Pull variables from a .result file more quickly"""
    varlen = len(variables)
    if not len(apps) == varlen:
        print('Error, variables must match appearances in length')
        return None
    mesh_dir, res_fn = os.path.split(result_fn)
    parts = None
    if 'partitions' in kwargs:
        parts = kwargs['partitions']
        if os.path.isdir(mesh_dir + '/partitioning.' + str(parts)):
            parts_dir = mesh_dir + '/partitioning.' + str(parts)
            print(str(parts) + ' partition mesh found in ' + parts_dir)
        elif os.path.isdir(mesh_dir + '/../partitioning.' + str(parts)):
            parts_dir = mesh_dir + '/../partitioning.' + str(parts)
            print(str(parts) + ' partition mesh found in ' + parts_dir)
        else:
            print('There is no mesh with the given partitions, trying to find with other partitioning')
            parts = None
    # try to figure out the number of partitions. Capped at 256.
    if parts is None:
        k = 2
        while k < 257:
            # Try result directory
            if os.path.isdir(mesh_dir + '/partitioning.' + str(k)):
                parts = k
                parts_dir = mesh_dir + '/partitioning.' + str(k)
                print(str(parts) + ' partition mesh found in ' + parts_dir)
                break
            # Try parent
            elif os.path.isdir(mesh_dir + '/../partitioning.' + str(k)):
                parts = k
                parts_dir = mesh_dir + '/../partitioning.' + str(k)
                print(str(parts) + ' partition mesh found in ' + parts_dir)
                break
            else:
                k = k + 1
        if not k < 257:
            print('No mesh partitions found')
            return None
    varnames = ['Node Number', 'x', 'y', 'z']
    types = [np.int64, np.float64, np.float64, np.float64]
    for i in range(len(variables)):  # Prepare to receive data about variables
        types.append(np.float64)  # data must be a float
        # node and coordinates fixed, name the desired variables
        varnames.append(variables[i])
    if maps is not None:
        for mp in maps:
            types.append(np.float64)
            varnames.append(variables[mp[1]] + mp[0].split('.')[-1])
    file_lens = [
        bufcount(parts_dir + '/part.' + str(i + 1) + '.nodes') for i in range(0, parts)]
    data = np.empty(sum(file_lens), dtype=list(zip(varnames, types)))
    for i in range(0, parts):  # Get data from each partition file
        # increment number of nodes taken thus far
        so_far = sum(file_lens[0:i])
        fin = open(parts_dir + '/part.' + str(i + 1) + '.nodes')
        points = fin.readlines()
        fin.close
        for k, pt in enumerate(points):
            points[k] = tuple([map(float, pt.split())[j] for j in [
                              0, 2, 3, 4]] + [0 for j in range(len(variables) + len(maps))])
        data[so_far:so_far + file_lens[i]] = points

        if not os.path.exists(result_fn + '.' + str(i)):
            # allows for use of dummy file name to quickly just get coordinates
            pass
        else:
            with open(result_fn + '.' + str(i)) as f:
                dat = f.readlines()

            # get all the indices for each of the variables
            var_indices = [[] for L in variables]
            for L, var in enumerate(variables):
                var_indices[L] = [k for k, x in enumerate(dat) if var in x]
            # prune the indices to get rid of repeats
            tot_apps = [
                len(var_indices[L]) / 2 for L in range(len(var_indices))]
            if any([apps[k] > tot_apps[k] for k in range(len(apps))]):
                print('Your apperances appear to be off')
                return 1
            for L, var in enumerate(variables):
                var_indices[L] = var_indices[L][-apps[L]]

            for L, var in enumerate(variables):
                try:
                    data[var][so_far:so_far + file_lens[i]] = list(map(
                        float, dat[var_indices[L] + 2:var_indices[L] + 2 + file_lens[i]]))
                    if maps is not None:
                        for mp in maps:
                            if mp[1] == L:
                                data[var + mp[0].split('.')[-1]][so_far:so_far + file_lens[i]] = list(map(
                                    eval(mp[0]), list(map(float, dat[var_indices[L] + 2:var_indices[L] + 2 + file_lens[i]]))))
                except ValueError:
                    if i == 0:
                        print('Variable \"' + var + '\" appears to be wonky, attempting to fetch regardless')
                    var_indices[L] = var_indices[L] + file_lens[i]
                    try:
                        data[var][so_far:so_far + file_lens[i]] = list(map(
                            float, dat[var_indices[L] + 2:var_indices[L] + 2 + file_lens[i]]))
                        if maps is not None:
                            for mp in maps:
                                if mp[1] == L:
                                    data[var + mp[0].split('.')[-1]][so_far:so_far + file_lens[i]] = list(map(
                                        eval(mp[0]), list(map(float, dat[var_indices[L] + 2:var_indices[L] + 2 + file_lens[i]]))))
                    except:
                        print('Failed to fetch')
                        return None
    data = np.sort(data, order=['x', 'y', 'z'])
    return data


def get_td_variables(variables, result_fn, apps=[2], t=None, **kwargs):
    """Pull variables from a time dependent .result file"""
    varlen = len(variables)
    if not len(apps) == varlen:
        print('Error, variables must match appearances in length')
        return None
    mesh_dir, res_fn = os.path.split(result_fn)
    parts = None
    if 'partitions' in kwargs:
        parts = kwargs['partitions']
        if os.path.isdir(mesh_dir + '/partitioning.' + str(parts)):
            parts_dir = mesh_dir + '/partitioning.' + str(parts)
            print(str(parts) + ' partition mesh found in ' + parts_dir)
        elif os.path.isdir(mesh_dir + '/../partitioning.' + str(parts)):
            parts_dir = mesh_dir + '/../partitioning.' + str(parts)
            print(str(parts) + ' partition mesh found in ' + parts_dir)
        else:
            print('There is no mesh with the given partitions, trying to find with other partitioning')
            parts = None
    # try to figure out the number of partitions. Capped at 256.
    if parts is None:
        k = 2
        while k < 257:
            # Try result directory
            if os.path.isdir(mesh_dir + '/partitioning.' + str(k)):
                parts = k
                parts_dir = mesh_dir + '/partitioning.' + str(k)
                print(str(parts) + ' partition mesh found in ' + parts_dir)
                break
            # Try parent
            elif os.path.isdir(mesh_dir + '/../partitioning.' + str(k)):
                parts = k
                parts_dir = mesh_dir + '/../partitioning.' + str(k)
                print(str(parts) + ' partition mesh found in ' + parts_dir)
                break
            else:
                k = k + 1
        if not k < 257:
            print('No mesh partitions found')
            return None
    varnames = ['Node Number', 'x', 'y', 'z', 'Time Step', 'Real Time']
    types = [np.int64, np.float64, np.float64,
             np.float64, np.int64, np.float64]
    for i in range(len(variables)):  # Prepare to receive data about variables
        types.append(np.float64)  # data must be a float
        # node and coordinates fixed, name the desired variables
        varnames.append(variables[i])
    file_lens = [
        bufcount(parts_dir + '/part.' + str(i + 1) + '.nodes') for i in range(0, parts)]
    single_data = np.empty(sum(file_lens), dtype=list(zip(varnames, types)))
    for i in range(0, parts):  # Get data from each partition file
        print('Reading in partition ' + str(i + 1))
        # increment number of nodes taken thus far
        so_far = sum(file_lens[0:i])
        fin = open(parts_dir + '/part.' + str(i + 1) + '.nodes')
        points = fin.readlines()
        fin.close
        for k, pt in enumerate(points):
            points[k] = tuple([map(float, pt.split())[j] for j in [
                              0, 2, 3, 4]] + [0 for j in range(len(variables) + 2)])
        single_data[so_far:so_far + file_lens[i]] = points

        if not os.path.exists(result_fn + '.' + str(i)):
            # allows for use of dummy file name to quickly just get coordinates
            pass
        else:
            with open(result_fn + '.' + str(i)) as f:
                dat = f.readlines()
            times = [L for L, x in enumerate(dat) if 'Time' in x]
            if len(times) == 0:
                print('This does not appear to be a time dependent file, going with steady state')
                return get_variables(variables, result_fn, apps=[2], **kwargs)
        # get the actual time values if this is the first result file
            if i == 0:
                real_times = np.zeros(len(times))
                for k, time in enumerate(times):
                    real_times[k] = float(dat[time].split()[-1])
                data = np.empty(
                    sum(file_lens) * len(times), dtype=list(zip(varnames, types)))
                for k in range(len(times)):
                    data[
                        k * sum(file_lens):(k + 1) * sum(file_lens)] = single_data
                    data['Time Step'][
                        k * sum(file_lens):(k + 1) * sum(file_lens)] = k
                    data['Real Time'][
                        k * sum(file_lens):(k + 1) * sum(file_lens)] = real_times[k]

            # get all the indices for each of the variables
            var_indices = [[] for L in variables]
            for L, var in enumerate(variables):
                var_indices[L] = [k for k, x in enumerate(dat) if var in x]
            # prune the indices to get rid of repeats
            tot_apps = [len(var_indices[L]) / (len(times) + 1)
                        for L in range(len(var_indices))]
            if any([apps[k] > tot_apps[k] for k in range(len(apps))]):
                print('Your apperances appear to be off')
                return 1
            app_pos = [(x1 - x2) for (x1, x2) in zip(tot_apps, apps)]
            for L, var in enumerate(variables):
                var_indices[L] = var_indices[L][
                    tot_apps[L] + app_pos[L]::tot_apps[L]]
            for k, time in enumerate(times):
                for L, var in enumerate(variables):
                    try:
                        data[var][k * sum(file_lens) + so_far:k * sum(file_lens) + so_far + file_lens[
                            i]] = [float(dat[var_indices[L][k] + 2 + j]) for j in range(file_lens[i])]
                    except ValueError:
                        if i == 0:
                            print('Variable \"' + var + '\" appears to be wonky, attempting to fetch regardless')
                        var_indices[L][k] = var_indices[L][k] + file_lens[i]
                        try:
                            data[var][k * sum(file_lens) + so_far:k * sum(file_lens) + so_far + file_lens[
                                i]] = [float(dat[var_indices[L][k] + 2 + j]) for j in range(file_lens[i])]
                        except:
                            print('Failed to fetch')
                            return None
    data = np.sort(data, order=['Time Step', 'x', 'y', 'z'])
    if t is not None:
        data = data[data['Time Step'] == t]
    return data


def get_bt_vars(data, bottom=True, nodes=False, varnames=None):
    """Pull the bottom or top out of a data matrix, can return node numbers, data can be z coord"""

    # Do a quick check that we have the necessary variables
    if not set(('Node Number', 'x', 'y')).issubset(set(data.dtype.names)):
        print('Not a valid dataset, does not contain Node Number, x, and y variables, returning None')
        return None

    # Try to get things right with what variables we are returning
    if varnames is None:
        varnames = list(data.dtype.names)
        if not nodes:
            try:
                varnames.remove('Node Number')
            except ValueError:
                print('Could not find node variable to remove, returning whole matrix')
    elif nodes:
        if not 'Node Number' in varnames:
            varnames.insert(0, 'Node Number')

    # Do the actual sorting
    points = []
    for x_val in np.unique(data['x']):
        x_points = data[data['x'] == x_val]
        for y_val in np.unique(x_points['y']):
            y_points = x_points[x_points['y'] == y_val]
            if bottom:
                ind = np.argmin(y_points['z'])
            else:
                ind = np.argmax(y_points['z'])
            points.append(y_points[ind])
    points = np.asarray(points)
    return points[varnames]


def get_da_vars(data):
    """Pull depth averaged values from pointwise data list"""
    # Do a quick check that we have the necessary variables
    if not set(('Node Number', 'x', 'y')).issubset(set(data.dtype.names)):
        print('Not a valid dataset, does not contain Node Number, x, and y variables, returning None')
        return None
    # Try to get things right with what variables we are returning
    varnames = list(data.dtype.names)
    varnames.remove('Node Number')
    # Do the actual sorting
    points = []
    for x_val in np.unique(data['x']):
        x_points = data[data['x'] == x_val]
        for y_val in np.unique(x_points['y']):
            y_points = x_points[x_points['y'] == y_val]
            pt = y_points[np.argmin(y_points['z'])]
            for var in varnames:
                pt[var] = np.sum(y_points[var]) / len(y_points[var])
            points.append(pt)
    points = np.asarray(points)
    return points[varnames]


def fix_bot_node_numbers(data_mat, botfile_in, botfile_out=None):
    """Avoid repartitioning problems by swiping nodenumbers from mesh being used"""
    # data_mat should come from get_bottom(get_variables). Fourth column is
    # irrelevant.
    if botfile_out is None:
        botfile_out = os.path.split(botfile_in)[0] + '/cor_bot.dat'
    f_in = open(botfile_in, 'r')
    lines = f_in.readlines()
    for i in range(len(lines)):
        line = lines[i].split()
        lines[i] = [0, float(line[1]), float(line[2]), float(line[3])]
    f_in.close()
    if not len(lines) == len(data_mat):
        print('Partitioned mesh and bottom data do not match in size')
        print(len(lines), len(data_mat))
        return None
    f_out = open(botfile_out, 'w')
    print('Writing output file ' + botfile_out)
    f_out.write('%d\n' % len(lines))
    for i in range(len(lines)):
        x_points = data_mat[data_mat['x'] == lines[i][1]]
        y_points = x_points[x_points['y'] == lines[i][2]]
        if y_points is None:
            print('Nodes do not match. Not writing output file')
            return None
        lines[i][0] = int(y_points['node'])
        f_out.write('%d %f %f %f\n' %
                    (lines[i][0], lines[i][1], lines[i][2], lines[i][3]))
    f_out.close()
    return 1


def bot_for_forward(mesh_dir, botfile_in, partitions=None):
    """Given a partition directory and file with bed elevation, make a file of bed elevation"""
    a = fix_bot_node_numbers(get_bt_vars(get_variables(
        [], mesh_dir + '/Test', partitions=partitions, apps=[]), nodes=True), botfile_in)
    if a:
        print('Making bottom elevation file successful')


def write_bottom_result(variables, result_fn, fn_out, nodes=False, apps=[2], **kwargs):
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert filetypes')
    parser.add_argument('fn', type=str, help='File to convert')
    parser.add_argument(
        'dst_type', type=str, choices=['tif', 'xyz', 'xy'], help='Dest. filetype')
    parser.add_argument('-o', type=str, default=None, help='Output filename')
    args = parser.parse_args()
    prefix, suffix = os.path.splitext(args.fn)
    if suffix == '.' + args.dst_type:
        print('Destination type is same as source. Quitting.')
        return 1
    if args.o is None:
        args.o = prefix + '.' + args.dst_type
    if suffix == '.tif':
        if args.dst_type == 'xy':
            gtif2xyz_fn_new(args.fn, args.o)
            return 0
        elif args.dst_type == 'xyz':
            gtif2xyz_fn_new(args.fn, args.o)
            return 0
    else:
        print('Conversion type not yet implemented')
        return 1


if __name__ == '__main__':
    main()
