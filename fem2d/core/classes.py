#! /usr/bin/env python
#cython: embedsignature=True
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.


"""Define a number of different classes which collectively constitute a finite element solver

All these definitions are contained in the file classes.py

Begin by importing a mesh from GMSH to create an instance of the :py:class:`Mesh` class.
This is best done by creating a :py:class:`Model` instance with the GMSH file as the argument; this will give you a model with associated mesh that you can then specify equation and boundary conditions for.
Next, the equation you want to solve (derived from :py:class:`Equation`) should be associated with the model using the :py:meth:`Model.add_equation` method.
The boundary conditions should then be attached using :py:meth:`Model.add_BC`.
At this poyou can either create the object to solve the equation in steady state using the :py:meth:`Model.makeIterate` method, or you can create a :py:class:`TimeDependentModel` instance around this model. In either case, you should call the :py:meth:`iterate` method of the resultant object.

Examples
--------
Basic solution to a simple case with mesh in testmesh.msh

>>> mo=Model('524_project/testmesh.msh')
>>> mo.add_equation(equations.diffusion())
>>> mo.add_BC('dirichlet',1,lambda x: 10.0)
>>> mo.add_BC('neumann',2,lambda x:-1.0)
>>> mo.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
>>> mo.add_BC('neumann',4,lambda x:0.0)
>>> m=LinearModel(mo)
>>> m.iterate()

Nonlinear models are similarly straightforward.
The boundary conditions must accept time as an argument and get an initial condition, for example

>>> mod=Model('524_project/testmesh.msh',td=True)
>>> mod.add_equation(equations.diffusion())
>>> mod.add_BC('dirichlet',1,lambda x,t: 26.0)
>>> mod.add_BC('neumann',2,lambda x,t:0.0)
>>> mod.add_BC( 'dirichlet',3,lambda x,t: 26.0)
>>> mod.add_BC('neumann',4,lambda x,t:0.0)
>>> initial_condition=lambda x:1+(x[0]-5)**2
>>> mi=TimeDependentModel(mod,10.0,2,initial_condition) # This does the solving too
>>> mi.animate(show=True) # Visualize the results
"""

from __future__ import print_function
from scipy.linalg import solve
from scipy.sparse.linalg import bicgstab,cg,spsolve,gmres,spilu,LinearOperator
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_an
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from warnings import warn
from .equations import Equation
from os.path import splitext
from scipy.sparse import csc_matrix,diags
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata
import numpy as np
Axes3D # Avoid the warning




class Node:
    curr_id = 0

    def __init__(self, x, y, z=0.0, ident=None, parent=None):
        self.ass_elms = []
        self.neighbors = {} # Dictionary of nodes and the connecting elements
        self.parent = parent
        self.x = x
        self.y = y
        if ident is not None:
            self.id = ident
            Node.curr_id = max(Node.curr_id, self.id)
        else:
            self.id = Node.curr_id
            Node.curr_id += 1

    def __str__(self):
        return 'Node number ' + str(self.id) + 'at (' + str(self.x) + ',' + str(self.y) + ')\nAssociate with elements ' + ', '.join(map(str, self.ass_elms))

    def add_elm(self, elm, pos):
        """Add an element using this node, with this as the pos'th node"""
        for node in self.parent.elements[elm].nodes:
            if node != self.id:
                if node not in list(self.neighbors.keys()):
                    self.neighbors[node]=[elm]
                else:
                    self.neighbors[node].append(elm)
        self.ass_elms.append([elm,pos])

    def coords(self):
        """Convenience wrapper for coordinates"""
        return np.array([self.x, self.y])


class Element(object):
    """A single finite element of given type
    
    Attributes
    ----------
    kind : str
        The type of element (triangular or linear currently)
    eltypes : int
        The gmsh numeric descriptor of this type of element
    id : int
        The number of the element
    parent : :py:class:`Mesh`
        The mesh to which the element is associated
    nodes : of ints
        The mesh nodes belonging to this element
    cent : 2-tuple of floats
        The center of this element
    gpoints : of 3-tuples (weight,x,y)
        The gauss points with weights of the parent element
    bases : functions
        The basis functions of the element, ordered so the i-th is 1 on the i-th node
    dbases : of 2-tuples (dx,dy)
        The derivatives of the basis functions
    F : function
        The mapping from the parent element to this element
    """
    curr_id = 0

    @staticmethod
    def init_element_gmsh(params, parent=None):
        ntags = params[2]
        nodes = params[(3 + ntags):]
        kwargs = {}
        if ntags >= 1:
            kwargs['physents'] = params[3]
            if ntags >= 2:
                kwargs['geoents'] = params[4]
                if ntags >= 3:
                    kwargs['npartits'] = params[5]
        if params[1] == 1:
            return LineElement(nodes, ident=params[0], parent=parent, skwargs=kwargs)
        elif params[1] == 2:
            return TriangElement(nodes, ident=params[0], parent=parent, skwargs=kwargs)
        else:
            print('Unknown element type')

    def __str__(self):
        string = 'Element Number ' + str(self.id) + '\nType: ' + str(self.kind) + '(' + str(
            self.eltypes) + ')\nAssociated with nodes ' + ', '.join([str(node) for node in self.nodes]) + '\nAnd with physical element '
        string += str(self.physents) if hasattr(self, 'physents') else 'None'
        return string

    def pvecs(self):
        """Get x,y vectors for plotting
        
        Returns
        -------
        x : array
        y : array
        """
        ncoords = [[], []]
        for node in self.nodes:
            ncoords[0].append(self.parent.nodes[node].x)
            ncoords[1].append(self.parent.nodes[node].y)
        if len(self.nodes) > 2:
            ncoords[0].append(ncoords[0][0])
            ncoords[1].append(ncoords[1][0])
        return ncoords[0], ncoords[1]

    def xyvecs(self):
        """Return xy vectors for use with basis functions etc
        
        Returns
        -------
        nodes : list
            A of coordinate pairs for nodes; nodwise as opposed to :py:meth:`Element.pvecs`
        """
        nodes_return = []
        for node in self.nodes:
            nodes_return.append(
                [self.parent.nodes[node].x, self.parent.nodes[node].y])
        return nodes_return

    def _gpts(self):
        """A function to return the gauss points. I use 4 for 2d, 2 for 1d"""
        self.gpts=[(pt[0],self._F()(pt[1:3])) for pt in self.gpoints]


class TriangElement(Element):
    """A single triangular finite element
    
    A subclass of :py:class:`Element` with the same properties
    """
    kind = 'Triangular'
    eltypes = 2
    gpoints=[[-27.0/96.0, 1.0/3.0, 1.0/3.0], [25.0/96.0, 0.2, 0.6], [25.0/96.0, 0.6, 0.2], [25.0/96.0, 0.2, 0.2]] #gauss points with weights for parent element

    def _F(self):
        """Right triangle to element mapping"""
        if self.F is None:
            ps=self.xyvecs()
            self.F=lambda p: np.dot([[ps[1][0] - ps[0][0], ps[2][0] - ps[0][0]], [ps[1][1] - ps[0][1], ps[2][1] - ps[0][1]]],np.array(p).reshape(len(p), 1))+np.array([[ps[0][0]], [ps[0][1]]])
        return self.F


    def _Finv(self):
        """Map from element to right triangle at origin"""
        if self.Finv is None:
            ps = self.xyvecs()
            self.area=abs((ps[1][0] - ps[0][0])*( ps[2][1] - ps[0][1] )-(ps[2][0] - ps[0][0])*( ps[1][1] - ps[0][1] ))/2.0
            self.Finv=lambda p: solve(np.array([[ps[1][0] - ps[0][0], ps[2][0] - ps[0][0]], [ps[1][1] - ps[0][1], ps[2][1] - ps[0][1]]]), np.array(p).reshape(len(p), 1) - np.array([[ps[0][0]], [ps[0][1]]]))
        return self.Finv


    def _normal(self):
        """Dummy to be lazy"""
        pass


    def _b1(self, Finv):
        """Define basis function 1 using map from element to origin"""
        def b1(p):
            Fi = Finv(p)
            return 1-Fi[0]-Fi[1]
        return b1

    def _b2(self, Finv):
        """Define basis function 2 using map from element to origin"""
        def b1(p):
            Fi = Finv(p)
            return Fi[0]
        return b1

    def _b3(self, Finv):
        """Define basis function 3 using map from element to origin"""
        def b1(p):
            Fi = Finv(p)
            return Fi[1]
        return b1

    def __init__(self, nodes, ident, parent, skwargs):
        if not len(nodes) == 3:
            raise ValueError('Bad Triangle')
        if ident is not None:
            self.id = ident
            Element.curr_id = max(Element.curr_id, self.id)
        else:
            self.id = Element.curr_id
            Element.curr_id += 1
        self.F=None
        self.Finv=None
        self.parent = parent
        self.nodes = nodes
        self.kind = TriangElement.kind
        self.eltypes = TriangElement.eltypes
        for tag, val in list(skwargs.items()):
            setattr(self, tag, val)
        ps=self.xyvecs()
        self.cent=[(ps[0][0]+ps[1][0]+ps[2][0])/3.0,(ps[0][1]+ps[1][1]+ps[2][1])/3]

    def _bases(self):
        self._Finv()
        self._F()
        self.bases = [self._b1(self.Finv), self._b2(self.Finv), self._b3(self.Finv)]
        return self.bases

    def _dbases(self):
        pts=self.xyvecs()
        self.dbases = [[self.bases[0](np.array(pts[1]).reshape(len(pts[0]),1)+np.array([[1.0],[0.0]])),self.bases[0](np.array(pts[1]).reshape(len(pts[0]),1)+np.array([[0.0],[1.0]]))],[self.bases[1](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[1.0],[0.0]])),self.bases[1](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[0.0],[1.0]]))],[self.bases[2](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[1.0],[0.0]])),self.bases[2](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[0.0],[1.0]]))]]
        return self.dbases


class LineElement(Element):

    """A single line finite element
    
    A subclass of :py:class:`Element` with the same attributes.
    """
    kind = 'Line'
    eltypes = 1
    gpoints = [ [0.5, (1.0-1.0/np.sqrt(3.0))/2.0, 0], [0.5, (1.0+1.0/np.sqrt(3.0))/2.0, 0]]


    def _F(self):
        if self.F is None:
            ps=self.xyvecs()
            self.F = lambda p: np.array([ps[0][0]+p[0]*(ps[1][0]-ps[0][0]),ps[0][1]+p[0]*(ps[1][1]-ps[0][1])])
        return self.F

    def _Finv(self):
        pts=self.xyvecs()
        return lambda p: np.array([[pts[0][0]+(pts[0][0]-pts[1][0])*p[0]],[pts[0][1]+(pts[0][1]-pts[1][1])*p[0]]])

    def _b2(self, pts):
        if pts[1][0]==pts[0][0]:
            return lambda x: (x[1]-float(pts[0][1]))/(float(pts[1][1]) - pts[0][1])
        else:
            return lambda x: (x[0]-float(pts[0][0]))/(float(pts[1][0]) - pts[0][0])

    def _b1(self, pts):
        if pts[1][0]==pts[0][0]:
            return lambda x: (float(pts[1][1]) - x[1]) / (pts[1][1] - pts[0][1])
        else:
            return lambda x: (float(pts[1][0]) - x[0]) / (pts[1][0] - pts[0][0])

    def _bases(self, *args):
        pts = self.xyvecs()
        self._F()
        self.length=np.sqrt(np.sum([(pts[0][i]-pts[1][i])**2 for i in range(len(pts[0]))]))
        self.bases = [self._b1(pts), self._b2(pts)]
        return self.bases

    def _dbases(self):
        pts=self.xyvecs()
        self.dbases= [[self.bases[0](np.array(pts[1]).reshape(len(pts[1]),1)+np.array([[1.0],[0.0]])),self.bases[0](np.array(pts[1]).reshape(len(pts[1]),1)+np.array([[0.0],[1.0]]))],[self.bases[1](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[1.0],[0.0]])),self.bases[1](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[0.0],[1.0]]))]]
        return self.dbases

    def _normal(self):
        pts=self.xyvecs()
        self.normal=np.array([pts[0][1]-pts[1][1],pts[1][0]-pts[0][0]])/self.length
        return self.normal

    def __init__(self, nodes, ident, parent, skwargs):
        if not len(nodes) == 2:
            raise ValueError('Not a valid line')
        if ident is not None:
            self.id = ident
            Element.curr_id = max(Element.curr_id, self.id)
        else:
            self.id = Element.curr_id
            Element.curr_id += 1
        self.parent = parent
        self.nodes = nodes
        self.kind = LineElement.kind
        self.eltypes = LineElement.eltypes
        self.F=None
        ps = self.xyvecs()
        self.cent=[(ps[0][0]+ps[1][0])/2,(ps[0][1]+ps[1][1])/2]
        for tag, val in list(skwargs.items()):
            setattr(self, tag, val)


class Mesh:

    """A finite element mesh"""

    def __init__(self):
        self.elements = {}
        self.nodes = {}
        self.bases = {}
        self.eltypes = {}
        self.physents = {}
        self.npartits = {}
        self.numnodes = 0
        self.numels = 0


    def __str__(self):
        string = 'Mesh object\nNumber of nodes: ' + \
            str(self.numnodes) + '\nNumber of elements: ' + \
            str(self.numels) + '\nElement types: \n'
        for key in list(self.eltypes.keys()):
            string += str(len(self.eltypes[key])) + \
                ' elements of type ' + str(key) + '\n'
        string += str(len(list(self.physents.keys()))) + ' physical entities\n'
        if self.bases:
            string += 'Bases formed'
        else:
            string += 'No Bases Associated'
        return string


    def loadgmsh(self, fn):
        with open(fn, 'r') as f:
            flines = f.readlines()
        if not flines[0] == '$MeshFormat\n':
            print('Unrecognized msh file')
            return False
        self.types = list(map(float, flines[1].split()))
        self.numnodes = int(flines[4])
        self.nodes = {int(line[0]): Node(*list(map(float, line[1:4])), ident=int(line[0]),parent=self)
                      for line in map(str.split, flines[5:(5 + self.numnodes)])}
        if not flines[self.numnodes + 6] == '$Elements\n':
            print('Unrecognized msh file')
            return False
        self.numels = int(flines[self.numnodes + 7])
        self.elements = {int(line[0]): Element.init_element_gmsh(list(map(int, line)), parent=self) for line in map(
            str.split, flines[(8 + self.numnodes):(8 + self.numnodes + self.numels)])}
        for key in list(self.elements.keys()):
            for attr in ['eltypes', 'physents', 'geoents', 'npartits']:
                try:
                    param = getattr(self.elements[key], attr)
                except AttributeError:
                    pass
                try:
                    paramlist = getattr(self, attr)
                    if not param in paramlist:
                        paramlist[param] = []
                    paramlist[param].append(key)
                except AttributeError:
                    pass
            for pos,node in enumerate(self.elements[key].nodes):
                self.nodes[node].add_elm(key,pos)
        flines = None
        self.coords=np.r_[[node.coords()[0:2] for node in list(self.nodes.values())]]


    def CreateBases(self,gpts=True,normals=True):
        """Create the finite element basis functions"""
        self.bases = {}
        self.dbases = {}
        if gpts:
            for number, element in list(self.elements.items()):
                self.bases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._bases())}
                self.dbases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._dbases())}
                element._gpts()
                if normals:
                    element._normal()
        else:
            for number, element in list(self.elements.items()):
                self.bases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._bases())}
                self.dbases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._dbases())}
                if normals:
                    element._normal()


    def PlotBorder(self, show=False, writefile=None, axis=None, fignum=None):
        """Plot out the border of the mesh with different colored borders"""
        if fignum is not None:
            plt.figure(fignum)
        else:
            plt.figure()
        colors = ['b', 'k', 'r', 'c', 'g', 'm', 'darkred', 'darkgreen',
                  'darkslategray', 'saddlebrown', 'darkorange', 'darkmagenta', 'y']
        plot_these = np.sort(list(self.physents.keys()))[0:-1]
        plt_lines = {}
        for i, key in enumerate(plot_these):
            for j, element in enumerate(self.physents[key]):
                if j == 0:
                    plt_lines[key], = plt.plot(
                        *self.elements[element].pvecs(), color=colors[i], label='Border '+str(key))
                else:
                    plt.plot(*self.elements[element].pvecs(), color=colors[i])
        plt.legend(handles=list(plt_lines.values()))
        if axis is not None:
            plt.axis(axis)
        if show:
            plt.show()
        if writefile is not None:
            plt.savefig(writefile)


    def PlotMesh(self, show=False, writefile=None, axis=None, labels=None, fignum=None):
        """Plot out the whole interior mesh structure"""
        if fignum is not None:
            plt.figure(fignum)
        else:
            plt.figure()
        plot_these = list(self.physents.keys())
        plt_lines = {key: None for key in plot_these}
        for i, key in enumerate(plot_these):
            for element in self.physents[key]:
                plt_lines[key], = plt.plot(
                        *self.elements[element].pvecs(), color = 'b')
        if axis is not None:
            plt.axis(axis)
        if labels=='area':
            for eln in self.eltypes[2]:
                plt.text(self.elements[eln].cent[0],self.elements[eln].cent[1],'%1.1f' % self.elements[eln].area)
        elif labels=='number':
            for eln in self.eltypes[2]:
                plt.text(self.elements[eln].cent[0],self.elements[eln].cent[1],'%d' % eln)
        elif labels=='edge_el_num':
            for eln in self.eltypes[1]:
                plt.text(self.elements[eln].cent[0],self.elements[eln].cent[1],'%d' % eln)
                
        if show:
            plt.show()
        if writefile is not None:
            plt.savefig(writefile)


class Model:
    """A model, with associated mesh, BCs, and equations
    
    Parameters
    ----------
    Mesh : string or classes.mesh
       This can be a .msh or .shp file, or an already imported mesh
    
    Keyword Arguments
    -----------------
    td : bool
       If `True` this is a time dependent model. Defaults to steady state

    Attributes
    ----------
    time_dep : bool
        True if time dependent, false if steady-state
    mesh : :py:class:`Mesh`
        Points to the associated mesh
    dofs : int
        The number of degrees of freedom in the equation to solve. e.g. 2 for 2D velocity
    eqn : :py:class:`equations.Equation`
        The equation to solve, should be attached using :py:meth:`add_equation`
    BCs : dictionary
        The associated boundary conditions, should be attached using :py:meth:`add_BC` 
    """
    def __init__(self,*mesh,**kwargs):
        if mesh:
            if type(mesh[0])==str:
                f_type=splitext(mesh[0])[1]
                if f_type=='.msh':
                    self.mesh=Mesh()
                    self.mesh.loadgmsh(mesh[0])
                elif f_type=='.shp':
                    self.mesh=Mesh()
                    self.mesh.loadShpMsh(mesh[0])
                else:
                    raise TypeError('Unknown mesh file type, need msh or shp')
            elif type(mesh[0])==Mesh:
                self.mesh=mesh[0]
            else:
                raise TypeError('Mesh input not understood')
            self.mesh.CreateBases()
        self.eqn=[]

        # Do a bunch of rigamarole to allow a couple kwargs for linearity
        if 'td' in kwargs:
            if kwargs['td']:
                self.time_dep=True
            else:
                self.time_dep=False
        elif 'time_dep' in kwargs:
            if kwargs['time_dep']:
                self.time_dep=True
            else:
                self.time_dep=False
        elif 'steady' in kwargs:
            if kwargs['steady']:
                self.time_dep=False
            else:
                self.time_dep=True
        else:
            print('Defaulting to steady state model')
            self.time_dep=False


    def add_equation(self,eqn):
        """Add the equation to be solved

        Parameters
        ----------
        eqn : `equations.Equation`
           Equation to solve

        Raises
        ------
        TypeError
           If the equation is not of the proper type
        """

        try:
            if not Equation in type(eqn).__bases__:
                raise TypeError('Need equation of type equations.Equation')
        except AttributeError:
            raise TypeError('Need equation of type equations.Equation')
        self.eqn.append(eqn)
        self.dofs=eqn.dofs
        if eqn.lin:
            self.linear=True
        else:
            self.linear=False
        return None


    def add_BC(self,cond_type,target_edge,function,eqn=0):
        """Assign a boundary condition (has some tests)
        
        Parameters
        ----------
        cond_type : string
            Type of boundary condition, must be dirchlet or neumann
        target_edge : int
            The number of the boundary to which this is being assigned
        function : function
            What the value is on this boundary,
            must be specified if time dependent
        """
        # You can also just manually edit the self.BCs dictionary
        if not cond_type in ['neumann','dirichlet']:
            raise TypeError('Not a recognized boundary condition type')
        elif not target_edge in self.mesh.physents.keys():
            raise ValueError('Specified target edge does not exist')
        elif target_edge==max(self.mesh.physents.keys()):
            raise ValueError('Specified target is plane not edge')
        else:
            if not self.time_dep:
                try:
                    function(self.mesh.nodes[self.mesh.elements[self.mesh.physents[target_edge][0]].nodes[0]].coords())
                except:
                    raise TypeError('Not a usable function, must take vector input and time')
            else:
                try:
                    function(self.mesh.nodes[self.mesh.elements[self.mesh.physents[target_edge][0]].nodes[0]].coords(),0.0)
                except:
                    raise TypeError('Not a usable function, must take vector input and time')

        self.BCs[target_edge]=(cond_type,function)


    def makeIterate(self):
        """Prepare to solve.

        Returns
        -------
        model : LinearModel or NonLinearModel
           Determined by whether or not the model is linear
        """
        if self.linear:
            return LinearModel(self,dofs=self.eqn.dofs)
        else:
            return NonLinearModel(self,dofs=self.eqn.dofs)


class ModelIterate:
    """This object makes matrix, forms a solution, etc

       Parameters
       ----------
       model : classes.Model
           The model, with equations and boundary conditions
       eqn : :py:class:`equations.Equation`,optional
           The equation to solve, if it differs from that tied to the model
           e.g. in a time dependent model

       Keyword Arguments
       -----------------
       dofs : int,optional
           Number of degrees of freedom. Default to that associated with the equation
       """



    def __init__(self,model,*eqn,**kwargs):
        self.parent=model
        self.mesh=self.parent.mesh
        if eqn:
            self.eqn=eqn[0]
        else:
            self.eqn=self.parent.eqn
        if 'DOFs' in kwargs:
            self.dofs=kwargs['DOFs']
        elif 'dofs' in kwargs:
            self.dofs=kwargs['dofs']
        else:
            self.dofs=1
        if not type(self.dofs)==int:
            raise TypeError('Degrees of freedom must be an integer')


    def MakeMatrixEQ(self,max_nei=12,**kwargs):
        """Make the matrix form, max_nei is the most neighbors/element
        
        Parameters
        ----------
        max_nei : int,optional
           The maximum number of nodes/equations per element. Overestimate this. Defaults to 12.
        
        Keyword Arguments
        -----------------
        All keyword arguments are simply passed along to the equation you are trying to solve.
        This should include things like source terms, conductivities, or other arguments needed by the equation.
        """
        #cdef np.ndarray[cINT32, ndim=1] row, col
        #cdef np.ndarray[cDOUBLE, ndim=1] data,rhs

        if self.dofs==1:
            # The easy version, scalar variable to solve for

            # Empty vectors to make the sparse matrix
            rows=np.zeros(max_nei*self.mesh.numnodes,dtype=np.int16)
            cols=np.zeros(max_nei*self.mesh.numnodes,dtype=np.int16)
            data=np.zeros(max_nei*self.mesh.numnodes)

            # zero vector for the rhs
            rhs=np.zeros(self.mesh.numnodes)

            # count the number of non-zeros
            nnz=0

            for i,node1 in self.mesh.nodes.items():

                # Do the diagonal element
                rows[nnz]=i-1 
                cols[nnz]=i-1
                data[nnz],rhs[i-1]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,**kwargs)
                nnz += 1

                for j,node2_els in node1.neighbors.items():
                    # Do the off diagonals, do not assume symmetry
                    rows[nnz]=i-1
                    cols[nnz]=j-1
                    data[nnz]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],max_nei=max_nei,**kwargs)
                    nnz += 1

            # store what we have done
            self.matrix=csc_matrix((data,(rows,cols)),shape=(self.mesh.numnodes,self.mesh.numnodes))
            self.rhs=rhs
            return None

        elif self.dofs==2:
            # Set things up so we can do velocity

            # Empty vectors to accept the sparse info, make them large for cross terms
            malloc=max_nei*self.mesh.numnodes*self.dofs**2
            m=self.mesh.numnodes*self.dofs

            rows=np.zeros(malloc,dtype=np.int16)
            cols=np.zeros(malloc,dtype=np.int16)
            data=np.zeros(malloc)


            rhs=np.zeros(m)

            #Count how many entries we have
            nnz=0

            for i,node1 in self.mesh.nodes.items():
                # Order things u1,v1,u2,v2,...
                # Still loop in the same way, just be careful with indexing
                # set things up for the diagonal for the first argument
                rows[nnz]=2*(i-1) 
                cols[nnz]=2*(i-1)

                # for the second argument
                rows[nnz+1]=2*i-1
                cols[nnz+1]=2*i-1

                # for the cross-term between the two components
                rows[nnz+2]=2*(i-1)
                cols[nnz+2]=2*i-1

                # for the other cross-term
                rows[nnz+3]=2*i-1
                cols[nnz+3]=2*(i-1)

                # Lazy, no checking for correct return from equation but so it goes
                data[nnz],data[nnz+1],data[nnz+2],data[nnz+3],rhs[i-1],rhs[i]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,**kwargs)
                
                # increment things
                nnz += 4

                for j,node2_els in node1.neighbors.items():
                    # Do the off diagonals, do not assume symmetry

                    # The first component off-diagonal
                    rows[nnz]=2*(i-1)
                    cols[nnz]=2*(j-1)

                    # The second component off-diagonal
                    rows[nnz+1]=2*i-1
                    cols[nnz+1]=2*j-1

                    # for the cross-term between the two components
                    rows[nnz+2]=2*(i-1)
                    cols[nnz+2]=2*j-1

                    # for the other cross-term
                    rows[nnz+3]=2*i-1
                    cols[nnz+3]=2*(j-1)

                    # Again, we hope the return from this equation is good, dumb things are happening with i,j in the supplement, so these don't match
                    data[nnz],data[nnz+1],data[nnz+2],data[nnz+3]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],max_nei=max_nei,**kwargs)

                    # increment again
                    nnz += 4

            # set up our matrix for real
            self.matrix=csc_matrix((data,(rows,cols)),shape=(m,m))
            self.rhs=rhs
            return None

        else:
            raise ValueError('Cannnot do more than 2 dofs')

        
    def applyBCs(self,time=None,normal=True):
        """ Put the boundary conditions into the matrix equations

        Parameters
        ----------
        time : float,optional
           The time if the model is time-dependent, None otherwise. Defaults to None.
        normal : bool,optional
           Specifies if the flux is normal for a Neumann condition. Defaults to True.
        """
        BCs=self.parent.BCs
        Mesh=self.mesh
        dirichlet=[edgeval[0] for edgeval in BCs.items() if edgeval[1][0]=='dirichlet']
        neumann=[edgeval[0] for edgeval in BCs.items() if edgeval[1][0]=='neumann']
        b_funcs={edgeval[0]:edgeval[1][1] for edgeval in BCs.items()}
        edges=np.sort(list(Mesh.physents.keys()))[0:-1]
        listed_edges=np.sort(dirichlet+neumann)
        if not all(edges==listed_edges):
            for edge in listed_edges:
                if not edge in edges:
                    print('You a non existent border '+str(edge)+' in types')
                    print('Available borders are ',edges)
                    raise ValueError('Unknown border')
            else:
                print('Some border not specified in types, taking Neumann')
                print('Borders are ',edges,' listed are ',listed_edges)


        # minor checking which we will warn but ignore
        if not all(edges==np.sort(list(b_funcs.keys()))):
            print('Error with borders')
            for edge in list(b_funcs.keys()):
                if not edge in edges:
                    print('You a non existent border '+str(edge)+' in types')
                    print('Available borders are ',edges)
                    raise ValueError ('Unknown border')
            else:
                print('Some border not specified in types, taking equal to zero')
                print('Borders are ',edges,' listed are ',b_funcs.keys())

        # Ok, hopefully we have parse-able input now
        edge_nodes={} # Figure out which nodes are associated with each boundary
        for edge in edges:
            edge_nodes[edge]=np.zeros((2*len(Mesh.physents[edge]),),dtype=int)
            for i,edge_element in enumerate(Mesh.physents[edge]):
                edge_nodes[edge][2*i]=Mesh.elements[edge_element].nodes[0]
                edge_nodes[edge][2*i+1]=Mesh.elements[edge_element].nodes[1]
            edge_nodes[edge]=np.unique(edge_nodes[edge]) # no repeats


        for edge in neumann:
            try:
                if time is not None:
                    self._applyNeumann(edge_nodes[edge],b_funcs[edge],normal=normal,time=time)
                else:
                    self._applyNeumann(edge_nodes[edge],b_funcs[edge],normal=normal)
            except KeyError: # If we have no condition we are just taking 0 neumann
                pass

        for edge in dirichlet:
            try:
                if time is not None:
                    self._applyDirichlet(edge_nodes[edge],b_funcs[edge],time=time)
                else:
                    self._applyDirichlet(edge_nodes[edge],b_funcs[edge])
            except KeyError:
                self._applyDirichlet(edge_nodes[edge],lambda x:0) # We actually need to do something to implement a zero
                # maybe throw the error though?


    def _applyNeumann(self,edge_nodes,function,normal=True,flux=True,time=None): #TODO make non-normal stuff, non-flux  possible
        """Apply a natural boundary condition, must be normal"""
        if self.dofs==1:
            for node in edge_nodes:
                for j,els in self.mesh.nodes[node].neighbors.items():
                    if j in edge_nodes:
                        for k,el in enumerate(els):
                            if self.mesh.elements[el].kind=='Line':
                                if not flux:
                                    raise TypeError('You need to specify the BC as a flux (e.g. divide out k in diffusion)')
                                if time is not None:
                                    if normal:
                                        self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]),time)*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                    else:
                                        self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]),time),self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                else:
                                    if normal:
                                        self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                    else:
                                        self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1])),self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

        elif self.dofs==2:
             for node in edge_nodes:
                for j,els in self.mesh.nodes[node].neighbors.items():
                    if j in edge_nodes:
                        for k,el in enumerate(els):
                            if self.mesh.elements[el].kind=='Line':
                                if not flux:
                                    raise TypeError('You need to specify the BC as a flux (e.g. divide out k in diffusion)')
                                if time is not None:
                                    if normal: # Normal, time-dependent, 2dofs
                                        self.rhs[2*(node-1)] = self.rhs[2*(node-1)]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]),time)[0]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1])[1],time)*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

                                    else: # Non-normal, time-dependent, 2dofs
                                        self.rhs[2*(node-1)] = self.rhs[2*(node-1)]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]),time)[0],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]),time)[1],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                else:
                                    if normal: # Normal, steady state, 2dofs
                                        self.rhs[2*(node-1)] = self.rhs[2*(node-1)]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))[0]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))[1]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

                                    else: # Non-normal, steady state, 2dofs
                                        self.rhs[2*(node-1)] = self.rhs[2*(node-1)]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]))[0],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]))[1],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

        else:
            raise ValueError('Cannot do more than 2 dofs')


    def _applyDirichlet(self,edge_nodes,function,time=None):
        """Let's apply an essential boundary condition"""
        if self.dofs==1:
            for node in edge_nodes:
                self.matrix[node-1,node-1]=1.0
                if time is not None:
                    self.rhs[node-1]=function(self.mesh.nodes[node].coords(),time)
                else:
                    self.rhs[node-1]=function(self.mesh.nodes[node].coords())
                for j in self.mesh.nodes[node].neighbors.keys(): # Get the neighboring nodes
                    if not j in edge_nodes: 
                        # Check if this neighboring node is on the edge
                        self.rhs[j-1]=self.rhs[j-1]-self.matrix[j-1,node-1]*self.rhs[node-1]
                    self.matrix[node-1,j-1]=0.0
                    self.matrix[j-1,node-1]=0.0   
        elif self.dofs==2:
            for node in edge_nodes:
                # We now have 4 terms relating to this element itself
                self.matrix[2*(node-1),2*(node-1)]=1.0
                self.matrix[2*node-1,2*node-1]=1.0
                self.matrix[2*node-1,2*(node-1)]=0.0
                self.matrix[2*(node-1),2*node-1]=0.0

                # Set the values on the right hand side
                if time is not None:
                    self.rhs[2*(node-1)]=function(self.mesh.nodes[node].coords(),time)[0]
                    self.rhs[2*node-1]=function(self.mesh.nodes[node].coords(),time)[1]
                else:
                    self.rhs[2*(node-1)]=function(self.mesh.nodes[node].coords())[0]
                    self.rhs[2*node-1]=function(self.mesh.nodes[node].coords())[1]

                # zero out the off-diagonal elements to get the condition correct
                # and keep symmetry if we have it
                for j in self.mesh.nodes[node].neighbors.keys(): # Get the neighboring nodes
                    if not j in edge_nodes: 
                        # Check if this neighboring node is on the edge

                        # We have four elements to zero out
                        self.rhs[2*(j-1)]=self.rhs[2*(j-1)]-self.matrix[2*(j-1),2*(node-1)]*self.rhs[2*(node-1)]
                        self.rhs[2*j-1]=self.rhs[2*j-1]-self.matrix[2*j-1,2*node-1]*self.rhs[2*node-1]
                        # Cross-terms
                        self.rhs[2*(j-1)]=self.rhs[2*(j-1)]-self.matrix[2*(j-1),2*node-1]*self.rhs[2*node-1]
                        self.rhs[2*j-1]=self.rhs[2*j-1]-self.matrix[2*j-1,2*(node-1)]*self.rhs[2*(node-1)]

                    # zero out each of these, and also the symmetric part
                    # all u
                    self.matrix[2*(node-1),2*(j-1)]=0.0
                    self.matrix[2*(j-1),2*(node-1)]=0.0 
                    # all v 
                    self.matrix[2*node-1,2*j-1]=0.0
                    self.matrix[2*j-1,2*node-1]=0.0
                    # uv
                    self.matrix[2*(node-1),2*j-1]=0.0
                    self.matrix[2*j-1,2*(node-1)]=0.0
                    # vu
                    self.matrix[2*node-1,2*(j-1)]=0.0
                    self.matrix[2*(j-1),2*node-1]=0.0

        else:
            raise ValueError('Cannot do more than 2 dofs')


    def solveIt(self,method='BiCGStab',precond='LU',tolerance=1.0e-5):
        """Solve the matrix equation

        Parameters
        ----------
        method : string,optional
           Must be one of CG, BiCGStab, GMRES, and direct. Default is BiCGStab
        precond : string or LinearOperator,optional
           Can be LU, or you can feed the matrix preconditioner object, or it can 
           be None. Defaults to LU.
        tolerance : float
           Convergence tolerance of the iterative method. Defaults to 1.0e-5
           
        Returns
        -------
        self.sol : array
            The matrix solution
        """


        if not method=='direct':
            if precond=='LU':
                p=spilu(self.matrix, drop_tol=1.0e-5)
                M_x=lambda x: p.solve(x)
                M=LinearOperator((self.mesh.numnodes*self.dofs,self.mesh.numnodes*self.dofs),M_x)
            elif precond is not None:
                M=precond
        if method=='CG':
            if precond is not None:
                self.sol,info=cg(self.matrix,self.rhs,tol=tolerance,M=M)
            else:
                self.sol,info=cg(self.matrix,self.rhs,tol=tolerance)
            if info>0:
                warn('Conjugate gradient did not converge. Attempting BiCGStab')
                if precond is not None:
                    self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance,M=M)
                else:
                    self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance)
                if info>0:
                    raise ConvergenceError(method='CG and BiCGStab',iters=info)
        elif method=='BiCGStab':
            if precond is not None:
                self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance,M=M)
            else:
                self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance)
            if info>0:
                raise ConvergenceError(method=method,iters=info)
        elif method=='direct':
            self.sol=spsolve(self.matrix,self.rhs)
        elif method=='GMRES':
            if precond is not None:
                self.sol,info=gmres(self.matrix,self.rhs,tol=tolerance,M=M)
            else:
                self.sol,info=gmres(self.matrix,self.rhs,tol=tolerance)
            if info>0:
                raise ConvergenceError(method=method,iters=info)
        else:
            raise TypeError('Unknown solution method')
        return self.sol


    def plotSolution(self,target=None,nodewise=True,threeD=True,savefig=None,show=False,x_steps=20,y_steps=20,cutoff=5,savesol=False,figsize=(15,10)):
        """ Plot the solution to the differential equation

        Parameters
        ----------
        target : string,optional
           What variable to grid. Defaults to the solution to the diff-EQ
        nodewise : bool,optional
           Indicates that things should be plotted at the nodes (as opposed to elements). Defaults to true.

        threeD : bool,optional
           If True, plot as 3D tri_surf, otherwise grid and plot 2D. Defaults to True.
        savefig : string,optional
           If a string is supplied, save figure with that filename. Defaults to None.
        show : bool,optional
           If True display the result in a window. Defaults to False.
        x_steps : int,optional
           The number of pixels in x, defaults to 500
        y_steps : int,optional
           The number of pixels in y, defaults to 500
        cutoff_dist : float,optional
           If the mesh is concave, supply this number to exclude pixels greater than this distance from node.
        savesol : bool,optional
           If True, store the gridded solution in memory as self.sol. Defaults to False.
        figsize : tuple,optional
           The dimensions of the figure. Defaults to (15,10)

        Returns
        -------
        mat_sol : of arrays
            The gridded solution, returned as x,y,solution
        """


        if self.dofs==1:
            mat_sol=self.sparse2mat(target=target,nodewise=nodewise,x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
            if savesol:
                self.matsol=mat_sol
            fig=plt.figure(figsize=figsize)
            if threeD:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(self.mesh.coords[:,0],self.mesh.coords[:,1],Z=self.sol,cmap=cm.jet)
            else:
                ctr=plt.contourf(*mat_sol,levels=np.linspace(0.9*min(self.sol),1.1*max(self.sol),50))
                plt.colorbar(ctr)
            if savefig is not None:
                plt.savefig(savefig)
            if show:
                plt.show()
            return mat_sol

        elif self.dofs==2:

            # Do a quick check before we do the slow steps
            if savefig is not None:
                if not len(savefig)==self.dofs:
                    raise ValueError('savefig must be of strings same length as dofs')

            mat_sol=self.sparse2mat(target=target,nodewise=nodewise,x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
            if savesol:
                # we want to have the option of not re-computing
                self.matsol=mat_sol

            # Do the plotting
            for i,ms in enumerate(mat_sol[2:]):
                fig=plt.figure(figsize=figsize)
                if threeD:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_trisurf(self.mesh.coords[:,0],self.mesh.coords[:,1],Z=ms,cmap=cm.jet)
                else:
                    ctr=plt.contourf(mat_sol[0],mat_sol[1],ms,levels=np.linspace(0.9*min(self.sol),1.1*max(self.sol),50))
                    plt.colorbar(ctr)
                    plt.title('Solution component {:d}'.format(i))
                if savefig is not None:
                    plt.savefig(savefig[i])
            if show:
                plt.show()
            return mat_sol
           

    def sparse2mat(self, target=None, nodewise=True, x_steps=500, y_steps=500, cutoff_dist=2000.0):
        """Grid up the solution, with potentially concave data
        
        Parameters
        ----------
        target : string,optional
           What variable to grid. Defaults to the solution to the diff-EQ
        nodewise : bool,optional
           Indicates that things should be plotted at the nodes (as opposed to elements). Defaults to true.

        x_steps : int,optional
           The number of pixels in x, defaults to 500
        y_steps : int,optional
           The number of pixels in y, defaults to 500
        cutoff_dist : float,optional
           If the mesh is concave, supply this number to exclude pixels greater than this distance from node.
        """
        coords=self.mesh.coords
        if self.dofs==1:
            data=self.sol
            tx = np.linspace(np.min(np.array(coords[:,0])), np.max(np.array(coords[:,0])), x_steps)
            ty = np.linspace(np.min(coords[:,1]), np.max(coords[:,1]), y_steps)
            XI, YI = np.meshgrid(tx, ty)
            ZI = griddata(coords, data, (XI, YI), method='linear')
            tree = KDTree(coords)
            dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
            dist = dist.reshape(XI.shape)
            ZI[dist > cutoff_dist] = np.nan
            return [tx, ty, ZI]

        elif self.dofs==2:
            data1=self.sol[::2]
            data2=self.sol[1::2]
            tx = np.linspace(np.min(np.array(coords[:,0])), np.max(np.array(coords[:,0])), x_steps)
            ty = np.linspace(np.min(coords[:,1]), np.max(coords[:,1]), y_steps)
            XI, YI = np.meshgrid(tx, ty)
            ZI = griddata(coords, data1, (XI, YI), method='linear')
            ZI2 = griddata(coords, data2, (XI, YI), method='linear')
            tree = KDTree(coords)
            dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
            dist = dist.reshape(XI.shape)
            ZI[dist > cutoff_dist] = np.nan
            ZI2[dist > cutoff_dist] = np.nan
            return [tx, ty, ZI, ZI2]


class LinearModel(ModelIterate):
    """A Linear Model Iterate"""
    # Basically the same as a model iterate, add a method to solve things
    kind='Linear'
    def iterate(self,method='BiCGStab',precond='LU',tolerance=1.0e-5,max_nei=12,time=None,**kwargs):
        self.MakeMatrixEQ(max_nei=max_nei,**kwargs)
        self.applyBCs(time=time)
        if time is not None:
            if 'BDF1' in kwargs:
                self.matrix=kwargs['timestep']*self.matrix+diags(np.ones(self.mesh.numnodes),0)
                self.rhs=kwargs['timestep']*self.rhs+kwargs['prev']
            elif 'BDF2' in kwargs:
                self.matrix=self.matrix-diags()
            else:
                raise ValueError('Cannot do that timestepping stategy')
        sol=self.solveIt(method='BiCGStab',precond='LU',tolerance=1.0e-5)
        return sol


class NonLinearModel:
    """A class for performing the solves on a nonlinear model, with the same method names

    Parameters
    ----------
    model : :py:class:`Model`
       The model with mesh, BCs, equation, etc.
    dofs : int,optional
       The number of degrees of freedom of the variable for which we are solving.

    """
    

    kind='NonLinear'

    def __init__(self,model,dofs=1):
        self.model=model
        self.dofs=dofs

    
    def iterate(self,gradient,relaxation=1.0,nl_tolerance=1.0e-5,guess=None,nl_maxiter=50,method='BiCGStab',precond='LU',tolerance=1.0e-5,max_nei=16,time=None,abort_not_converged=False,**kwargs):
        """
        The method for performing the solution to the nonlinear model iterate

        Parameters
        ----------
        gradient : function
           This gets called at every iteration in order to update parameters used in the equation being solved
        relaxation : float,optional
           The amount to relax. Use less than 1 if you have convergence problems. Defaults to 1.0.
        nl_tolerance : float,optional
           When to declare things converged
        guess : array,optional
           An initial guess at the solution. If None, use all zeros. Defaults to None.
        nl_maxiter : int,optional
           Maximum number of nonlinear iterations. Defaults to 50.
        method : string,optional
           Solution method to use for the linear system. Defaults to BiCGStab. Done using :py:meth:`ModelIterate.solveIt`.
        precond : string,optional
           Preconditioning method for the linear system if solved iteratively. Defaults to ILU. Can also be a LinearOperator which does the solving using a preconditioning matrix or matrices.
        tolerance : float,optional
           Linear system convergence tolerance for iterative methods. Defaults to 1.0e-5
        max_nei : int,optional
           Maximum number of neighboring elements times dofs. Err large. Defaults to 16.
        time : float,optional
           The time in a time dependent model. None for steady state. Defaults to None.
        abort_not_converged : bool,optional
           If true, raise a :py:exc:`ConvergenceError` if non-linear iterations do not converge. Otherwise call it good enough. Defaults to False.
        
        Any Any keyword arguments are passed down to the equation we are solving, for example time dependent terms like sources or conductivity can be specified.

        Returns
        -------
        solution : array
           The solution to the nonlinear Equation.

        """

        # Make an initial guess at the velocities. Let's just use 0s by default
        if guess is not None:
            old=guess
        else:
            old=np.zeros(self.model.mesh.numnodes*self.dofs)

        try:
            for i in range(nl_maxiter):
                # Loop until we have converged to within the desired tolerance
                print( 'Nonlinear iterate {:d}:    '.format(i) , end=' ')
                kwargs['gradient']=gradient(self,old)

                self.linMod=LinearModel(self.model,dofs=self.dofs)
                new=self.linMod.iterate(method=method,precond=precond,tolerance=tolerance,max_nei=max_nei,time=time,**kwargs)
                if np.linalg.norm(new)>1.0e-15:
                    relchange=np.linalg.norm(new-old)/np.sqrt(float(self.model.mesh.numnodes))/np.linalg.norm(new)
                else:
                    relchange=np.linalg.norm(new-old)/np.sqrt(float(self.model.mesh.numnodes))
                print('Relative Change: {:f}'.format(relchange))

                #Check if we converged
                if relchange<nl_tolerance and i != 0:
                    break
                old[:]=relaxation*new+(1.0-relaxation)*old
            else: # Executed if we never break b/c of convergence
                if abort_not_converged:
                    raise ConvergenceError('Nonlinear solver failed within iteration count')
                else:
                    print('Nonlinear solver did not converge within desired tolerance, returning')
            self.sol=new
        except KeyboardInterrupt:
            try:
                self.sol=new
            except:
                raise ConvergenceError('You aborted before first iterate finished')
        return new


    def plotSolution(self, target=None, nodewise=True, threeD=True, savefig=None, show=False, x_steps=500, y_steps=500, cutoff=7000, savesol=False, vel=False, figsize=(15,10), clims=None):
        """ Plot the resulting nonlinear solution.

        See :py:meth:`ModelIterate.plotSolution` for explanation of parameters.
        """
        if nodewise:
            coords=self.model.mesh.coords
            if target is not None:
                if target in self.vars:
                    sol=self.vars[target]
                else:
                    sol=np.zeros(self.model.mesh.numnodes)
                    for node in self.model.mesh.nodes:
                        try:
                            sol[node.id]=getattr(node,target)
                        except AttributeError:
                            raise AttributeError('Neither model nor nodes has desired variable')
            else:
                sol=self.sol

        else:
            # Elementwise
            tri_els=self.model.mesh.eltypes[2]
            coords=np.zeros([len(tri_els),2])
            sol=np.zeros(len(tri_els))
            if type(getattr(self.model.mesh.elements[tri_els[0]],target))==float or type(getattr(self.model.mesh.elements[tri_els[0]],target))==np.float64:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    sol[i]=getattr(self.model.mesh.elements[element],target)
            elif type(getattr(self.model.mesh.elements[tri_els[0]],target))==list:
                 for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    sol[i]=getattr(self.model.mesh.elements[element],target)[0][1]
            else:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    try:
                        sol[i]=getattr(self.model.mesh.elements[element],target)(element.cent)
                    except:
                        raise RuntimeError('Problems with parsing function for plotting')


        if self.dofs==1 or target is not None:
            mat_sol=self.sparse2mat(target=target,nodewise=nodewise,x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
            if savesol:
                self.matsol=mat_sol
            fig=plt.figure(figsize=figsize)
            if threeD:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(coords[:,0],coords[:,1],Z=sol,cmap=cm.jet)
            else:
                if clims is None:
                    clims=[0.9*min(sol),1.1*max(sol)]
                ctr=plt.contourf(*mat_sol,levels=np.linspace(*clims,num=50))
                plt.colorbar(ctr)
            if savefig is not None:
                plt.savefig(savefig)
            if show:
                plt.show()
            return mat_sol

        elif self.dofs==2:

            # Do a quick check before we do the slow steps
            if savefig is not None:
                if not vel:
                    if not len(savefig)==self.dofs:
                        raise ValueError('savefig must be of strings same length as dofs')

            mat_sol=self.sparse2mat(x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
            if savesol:
                # we want to have the option of not re-computing
                self.matsol=mat_sol




            # Do the plotting
            if not vel:
                for i,ms in enumerate(mat_sol[2:]):
                    fig=plt.figure(figsize=figsize)
                    if threeD:
                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot_trisurf(self.model.mesh.coords[:,0],self.model.mesh.coords[:,1],Z=self.sol[i::2],cmap=cm.jet)
                    else:
                        ctr=plt.contourf(mat_sol[0],mat_sol[1],ms,levels=np.linspace(0.9*min(self.sol[i::2]),1.1*max(self.sol[i::2]),50))
                        plt.colorbar(ctr)
                        plt.title('Solution component {:d}'.format(i))
                    if savefig is not None:
                        plt.savefig(savefig[i])
            else:
                fig = plt.figure(figsize=figsize)
                if threeD:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_trisurf(self.model.mesh.coords[:,0],self.model.mesh.coords[:,1],Z=np.sqrt(self.sol[0::2]**2+self.sol[1::2]**2),cmap=cm.jet)
                else:
                    ctr=plt.contourf(mat_sol[0],mat_sol[1],np.sqrt(mat_sol[2]**2+mat_sol[3]**2),50)
                    plt.colorbar(ctr)
                if savefig is not None:
                    plt.savefig(savefig)

            if show:
                plt.show()
            return mat_sol
           

    def sparse2mat(self, target=None, nodewise=True, x_steps=500, y_steps=500, cutoff_dist=2000.0):
        """Grid up the solution, with potentially concave data
        
        Parameters
        ----------
        target : string,optional
           What variable to grid. Defaults to the solution to the diff-EQ
        nodewise : bool,optional
           Indicates that things should be plotted at the nodes (as opposed to elements). Defaults to true.
        x_steps : int,optional
           The number of pixels in x, defaults to 500
        y_steps : int,optional
           The number of pixels in y, defaults to 500
        cutoff_dist : float,optional
           If the mesh is concave, supply this number to exclude pixels greater than this distance from node.

        Returns
        -------
           Matrix Solution : list
              Matrix solution of the form x,y,solution_1,*solution_2 where solution_2 is only returned if the variable we are solving for is 2d.
        """
        if nodewise:
            coords=self.model.mesh.coords
            if target is not None:
                if target in self.vars:
                    data=self.vars[target]
                else:
                    data=np.zeros(self.model.mesh.numnodes)
                    for node in self.model.mesh.nodes:
                        try:
                            data[node.id]=getattr(node,target)
                        except AttributeError:
                            raise AttributeError('Neither model nor nodes has desired variable')


            elif self.dofs==2:
                data1=self.sol[::2]
                data2=self.sol[1::2]
                tx = np.linspace(np.min(np.array(coords[:,0])), np.max(np.array(coords[:,0])), x_steps)
                ty = np.linspace(np.min(coords[:,1]), np.max(coords[:,1]), y_steps)
                XI, YI = np.meshgrid(tx, ty)
                ZI = griddata(coords, data1, (XI, YI), method='linear')
                ZI2 = griddata(coords, data2, (XI, YI), method='linear')
                tree = KDTree(coords)
                dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
                dist = dist.reshape(XI.shape)
                ZI[dist > cutoff_dist] = np.nan
                ZI2[dist > cutoff_dist] = np.nan
                return [tx, ty, ZI, ZI2]
            else:
                raise ValueError('Too many dofs')

        else:
            # Elementwise
            tri_els=self.model.mesh.eltypes[2]
            coords=np.zeros([len(tri_els),2])
            data=np.zeros(len(tri_els))
            if type(getattr(self.model.mesh.elements[tri_els[0]],target))==float or type(getattr(self.model.mesh.elements[tri_els[0]],target))==np.float64:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    data[i]=getattr(self.model.mesh.elements[element],target)
            elif type(getattr(self.model.mesh.elements[tri_els[0]],target))==list:
                 for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    data[i]=getattr(self.model.mesh.elements[element],target)[0][1]
            else:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    try:
                        data[i]=getattr(self.model.mesh.elements[element],target)(element.cent)
                    except:
                        raise RuntimeError('Problems with parsing function for plotting')

        # The generic 1d stuff
        tx = np.linspace(np.min(np.array(coords[:,0])), np.max(np.array(coords[:,0])), x_steps)
        ty = np.linspace(np.min(coords[:,1]), np.max(coords[:,1]), y_steps)
        XI, YI = np.meshgrid(tx, ty)
        ZI = griddata(coords, data, (XI, YI), method='linear')
        tree = KDTree(coords)
        dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
        dist = dist.reshape(XI.shape)
        ZI[dist > cutoff_dist] = np.nan
        return [tx, ty, ZI]


class TimeDependentModel:
    """A time dependent model"""


    def __init__(self,model,timestep,n_steps,initial_condition,method='BDF1',lin_method='BiCGStab',precond='LU',lin_tolerance=1.0e-5):
        """We are doing this as a class to organize the results, but init does all computation"""
        if not type(model)==Model:
            raise TypeError('Model must be of class Model')
        self.model=model
        if not type(timestep)==float:
            raise TypeError('Timestep must be a float')
        self.timestep=timestep
        if not type(n_steps)==int:
            raise TypeError('Number of timesteps must be an integer')
        if not n_steps>0:
            raise ValueError('Number of timesteps must be strictly greater than 0')
        self.n_steps=n_steps
        if not lin_method in ['BiCGStab','GMRES','direct','CG']:
            raise ValueError('Not a supported linear solver')
        self.lin_method=lin_method
        if not precond in ['LU', None]:
            raise ValueError('Not a supported procondtioner')
        self.precond=precond
        if not type(lin_tolerance)==float:
            raise TypeError('Tolerance must be a float')
        self.lin_tolerance=lin_tolerance
        self.ic=initial_condition
        self.method=method
        self.sol=self.iterate()


    def __str__(self):
        return 'Time dependent model with '+str(self.n_steps)+' time steps'


    def iterate(self):
        """
        Solve the equation. I don't allow any choices as of now.

        Returns
        -------
        solution : list
           A of arrays of the node-wise solution at each timestep. The first entry is the initial condition.
        """
        sol=[np.array([self.ic(pt) for pt in self.model.mesh.coords])]
        if self.model.eqn.lin:
            iterate=LinearModel
        else:
            iterate=NonLinearModel
        if self.method=='BDF2':
            time=self.timestep
            equation=self.model.eqn
            model_iterate=iterate(self.model,equation)
            sol.append(model_iterate.iterate(method=self.lin_method,precond=self.precond,tolerance=self.lin_tolerance,time=time,BDF1=True,timestep=self.timestep,prev=sol[-1]))
            for i in range(2,self.n_steps):
                time=i*self.timestep
                equation=self.model.eqn
                model_iterate=iterate(self.model,equation)
                sol.append(model_iterate.iterate(method=self.lin_method,precond=self.precond,tolerance=self.lin_tolerance,time=time,timestep=self.timestep))
        elif self.method=='BDF1':
            for i in range(1,(self.n_steps+1)):
                time=i*self.timestep
                print('Timestep {:d}, real time {:f}'.format(i,time))
                equation=self.model.eqn
                model_iterate=iterate(self.model,equation)
                sol.append(model_iterate.iterate(method=self.lin_method,precond=self.precond,tolerance=self.lin_tolerance,time=time,BDF1=True,timestep=self.timestep,prev=sol[-1]))
        else:
            raise ValueError('Not a supported timestepping method. Use BDF2.')
        return sol


    def plotSolution(self,iterate=-1,threeD=True,savefig=None,show=False,x_steps=20,y_steps=20,cutoff=5,savesol=False,figsize=(15,10)):
        """ Plot the solution at a some poduring the run

        For parameter options see :py:meth:`ModelIterate.plotSolution`. In addition, supports
        Parameters
        ----------
        iterate : int,optional
           The timestep we want to plot. Defaults to -1 (the final one).
        """
        mat_sol=self.sparse2mat(iterate, x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
        if savesol:
            self.matsol=mat_sol
        fig=plt.figure(figsize=figsize)
        if threeD:
            ax = p3.Axes3D(fig) 
            ax.plot_trisurf(self.model.mesh.coords[:,0],self.model.mesh.coords[:,1],Z=self.sol[iterate],cmap=cm.jet)
        else:
            ctr=plt.contourf(*mat_sol,levels=np.linspace(0.9*min(self.sol),1.1*max(self.sol[iterate]),50))
            plt.colorbar(ctr)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        return mat_sol


    def _plotAnimate(self,num,ax):
        iterate=0
        while iterate<=self.n_steps:
            iterate += 1
            yield ax.plot_trisurf(self.model.mesh.coords[:,0],self.model.mesh.coords[:,1],Z=self.sol[iterate],cmap=cm.jet)


    def animate(self,show=False,save=None):
        """ Animate the solution

        Parameters
        ----------
        show : bool,optional
           If True, display the plot
        save : string,optional
           In not None, save animation as the given name. Uses ffmpeg writer. Only mp4 extension is tested.
        """
        fig=plt.figure()
        ax=p3.Axes3D(fig)
        ax.set_xlim3d([min(self.model.mesh.coords[:,0]),max(self.model.mesh.coords[:,0])])
        ax.set_ylim3d([min(self.model.mesh.coords[:,1]),max(self.model.mesh.coords[:,1])])
        ax.set_zlim3d([0.9*min(self.sol[0]),1.1*max(self.sol[0])])
        class nextPlot:
            def __init__(self,outer):
                self.iterate=0
                self.outer=outer
            def __call__(self,num):
                self.iterate+=1
                ax.clear()
                ax.set_xlim3d([min(self.outer.model.mesh.coords[:,0]),max(self.outer.model.mesh.coords[:,0])])
                ax.set_ylim3d([min(self.outer.model.mesh.coords[:,1]),max(self.outer.model.mesh.coords[:,1])])
                ax.set_zlim3d([0.9*min(self.outer.sol[0]),1.1*max(self.outer.sol[0])])
                return ax.plot_trisurf(self.outer.model.mesh.coords[:,0],self.outer.model.mesh.coords[:,1],Z=self.outer.sol[self.iterate-1],cmap=cm.jet)
        np=nextPlot(self)
        an=mpl_an.FuncAnimation(fig, np, self.n_steps-1, interval=5000, blit=False)
        Writer = mpl_an.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='David Lilien'), bitrate=1800)
        if save is not None:
            an.save(save,writer=writer)
        if show:
            plt.show()
        return None


    def sparse2mat(self, iterate, x_steps=500, y_steps=500, cutoff_dist=2000.0):
        """Grid up the solution

        Same as :py:meth:`ModelIterate.sparse2mat` except takes a parameter to specify the timestep.
        Parameters
        ----------
        iterate : int
           The timestep to grid up
        """
        coords=self.model.mesh.coords
        data=self.sol[iterate]
        tx = np.linspace(np.min(np.array(coords[:,0])), np.max(np.array(coords[:,0])), x_steps)
        ty = np.linspace(np.min(coords[:,1]), np.max(coords[:,1]), y_steps)
        XI, YI = np.meshgrid(tx, ty)
        ZI = griddata(coords, data, (XI, YI), method='linear')
        tree = KDTree(coords)
        dist, _ = tree.query(np.c_[XI.ravel(), YI.ravel()], k=1)
        dist = dist.reshape(XI.shape)
        ZI[dist > cutoff_dist] = np.nan
        return [tx, ty, ZI]


class ConvergenceError(Exception):
    """Error for bad iterative method result
    
    Parameters
    ----------
    method : string,optional
        The iterative method which caused the error
    iters : int,optional
        The iteration number at failure
    """


    def __init__(self,method=None,iters=None):
        self.method=method
        self.iters=iters


    def __str__(self):
        return 'Method '+self.method+' did not converge at iteration '+str(self.iters)


def main():
    import equations
    mo=Model('524_project/testmesh.msh')
    mo.add_equation(equations.diffusion())
    mo.add_BC('dirichlet',1,lambda x: 10.0)
    mo.add_BC('neumann',2,lambda x:-1.0) # 'dirichlet',2,lambda x: 10.0)
    mo.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
    mo.add_BC('neumann',4,lambda x:0.0)
    m=LinearModel(mo)
    m.iterate()


    admo=Model('524_project/testmesh.msh')
    admo.add_equation(equations.advectionDiffusion())
    admo.add_BC('dirichlet',1,lambda x: 15.0)
    admo.add_BC('neumann',2,lambda x:0.0) # 'dirichlet',2,lambda x: 10.0)
    admo.add_BC( 'dirichlet',3,lambda x: 5.0)
    admo.add_BC('neumann',4,lambda x:0.0)
    am=LinearModel(admo)
    am.iterate(v=lambda x:np.array([1.0,0.0]))

    mod=Model('524_project/testmesh.msh',td=True)
    mod.add_equation(equations.diffusion())
    mod.add_BC('dirichlet',1,lambda x,t: 26.0)
    mod.add_BC('neumann',2,lambda x,t:0.0) # 'dirichlet',2,lambda x: 10.0)
    mod.add_BC( 'dirichlet',3,lambda x,t: 26.0)
    mod.add_BC('neumann',4,lambda x,t:0.0)
    #mi=TimeDependentModel(mod,10.0,2,lambda x:1+(x[0]-5)**2)
    #mi.animate(show=False,save='decay.mp4')
    return am #m,am,mi

    
    


if __name__ == '__main__':
    #main()
    import cProfile

    def do_cprofile(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        return profiled_func

    @do_cprofile
    def upwinding():
        import equations
        """test upwinding"""
        class k:
            def __init__(self,vel,k_old,alpha):
                self.vel=vel
                self.k=k_old
                self.alpha=alpha
            def __call__(self,pt):
                v=self.vel(pt)
                return self.k(pt)+self.alpha*0.5/2.0*np.outer(v,v)/max(1.0e-8,np.linalg.norm(v))

        alpha=3.0
        k_old=lambda x:np.array([[1.0, 0.0],[0.0, 1.0]])
        vel=lambda x: np.array([1000.0,0.0])
        k_up=k(vel,k_old,alpha)

        admo=Model('524_project/testmesh.msh')
        admo.add_equation(equations.advectionDiffusion())
        admo.add_BC('dirichlet',1,lambda x: 15.0)
        admo.add_BC('neumann',2,lambda x:0.0) # 'dirichlet',2,lambda x: 10.0)
        admo.add_BC( 'dirichlet',3,lambda x: 5.0)
        admo.add_BC('neumann',4,lambda x:0.0)
        am=LinearModel(admo)
        am.iterate(v=vel,k=k_up)
        #am.plotSolution(savefig='figs/upwinded.eps',show=True)
        am.plotSolution()
        plt.title('v=1000, upwinded')
        plt.savefig('figs/upwinded1000.eps')



    upwinding()
