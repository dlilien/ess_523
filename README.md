# fem_2d

Some 2D finite element mesh tools. The majority of the work is done in classes.py. You need to solve equations derived from a parent class contained in equations.py. Both are in the core module of fem2d. Most of the other files are there for testing or convenience. Documentation is at http://students.washington.edu/dal22/FEMdocs

Additional files for import of geophysical data and plotting, as well as for use with the shallow shelf approximation for ice flow are in fem2d.lib.

The executables in ssa and masscon will not run if you don't have fem2d on your python path. The easy solution is to run 'python3 setup.py build' and then 'python3 setup.py install', assuming you have the python 3 framework on your python path.

Dependencies:
1. Python
2. Numpy
3. Scipy
4. Matplotlib

Recommended:
Gmsh
