#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
Define a mesh classes that have info I might want about finite element meshes
"""

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import bicgstab,cg,spsolve,gmres,spilu,LinearOperator
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_an
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from warnings import warn
from equationsFEM import Equation
from os.path import splitext
from scipy.sparse import csc_matrix,diags
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata
Axes3D # Avoid the warning
bc=0


class Node:

    """A node with x,y,z coordinates"""
    _curr_id = 0

    def __init__(self, x, y, z=0.0, ident=None, parent=None):
        self.ass_elms = []
        self.neighbors = {} # Dictionary of nodes and the connecting elements
        self.parent = parent
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if ident is not None:
            self.id = ident
            Node._curr_id = max(Node._curr_id, self.id)
        else:
            self.id = Node._curr_id
            Node._curr_id += 1

    def __str__(self):
        return 'Node number ' + str(self.id) + 'at (' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')\nAssociate with elements ' + ', '.join(map(str, self.ass_elms))

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
        return np.array([self.x, self.y, self.z])

    def form_basis(self):
        """Write something to note which basis functions are associated?"""
        pass


class Element(object):

    """A single finite element, of given type"""
    _curr_id = 0

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
        """Return x,y vectors for plotting"""
        ncoords = [[], []]
        for node in self.nodes:
            ncoords[0].append(self.parent.nodes[node].x)
            ncoords[1].append(self.parent.nodes[node].y)
        if len(self.nodes) > 2:
            ncoords[0].append(ncoords[0][0])
            ncoords[1].append(ncoords[1][0])
        return ncoords[0], ncoords[1]

    def xyvecs(self):
        """Return xy vectors for use with basis functions etc"""
        nodes_return = []
        for node in self.nodes:
            nodes_return.append(
                [self.parent.nodes[node].x, self.parent.nodes[node].y])
        return nodes_return

    def _gpts(self):
        """A function to return the gauss points. I use 4 for 2d, 2 for 1d"""
        self.gpts=[(pt[0],self._Finv()(pt[1:3])) for pt in self.gpoints]


class TriangElement(Element):

    """A single triangular finite element"""
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
            print('Bad Triangle')
            return None
        if ident is not None:
            self.id = ident
            Element._curr_id = max(Element._curr_id, self.id)
        else:
            self.id = Element._curr_id
            Element._curr_id += 1
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

    """A single line finite element"""
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
            print('Bad Line')
            return None
        if ident is not None:
            self.id = ident
            Element._curr_id = max(Element._curr_id, self.id)
        else:
            self.id = Element._curr_id
            Element._curr_id += 1
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
        self.coords=np.r_[[node.coords()[0:-1] for node in list(self.nodes.values())]]

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
                element._gpts()
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
        plt_lines = {key: None for key in plot_these}
        for i, key in enumerate(plot_these):
            for j, element in enumerate(self.physents[key]):
                if j == 0:
                    plt_lines[key], = plt.plot(
                        *self.elements[element].pvecs(), color=colors[i], label=str(key))
                else:
                    plt.plot(*self.elements[element].pvecs(), color=colors[i])
        plt.legend(
            list(plt_lines.values()), ['Border ' + num for num in map(str, plot_these)])
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
    """A steady state model, with associated mesh, BCs, and equations"""
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
        self.BCs={}

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
        try:
            if not Equation in type(eqn).__bases__:
                raise TypeError('Need equation of type equationsFEM.Equation')
        except AttributeError:
            raise TypeError('Need equation of type equationsFEM.Equation')
        self.eqn=eqn
        if eqn.lin:
            self.linear=True
        else:
            self.linear=False
        return None


    def add_BC(self,cond_type,target_edge,function=lambda x:0.0):
        """Assign a boundary condition, has some checking"""
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
        if self.linear:
            return LinearModel(self)
        else:
            return NonLinearModel(self)


class ModelIterate:
    """This object makes matrix, forms a solution, etc"""


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

    

    def MakeMatrixEQ(self,max_nei=12,parkwargs={}):
        """Make the matrix form, max_nei is the most neighbors/element"""
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
                data[nnz],rhs[i-1]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,kwargs=parkwargs)
                nnz += 1

                for j,node2_els in node1.neighbors.items():
                    # Do the off diagonals, do not assume symmetry
                    rows[nnz]=i-1
                    cols[nnz]=j-1
                    data[nnz]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],max_nei=max_nei,kwargs=parkwargs)
                    nnz += 1

            # store what we have done
            self.matrix=csc_matrix((data,(rows,cols)),shape=(self.mesh.numnodes,self.mesh.numnodes))
            self.rhs=rhs
            return None

        elif self.dofs==2:
            # Set things up so we can do velocity

            # Empty vectors to accept the sparse info, make them large for cross terms
            rows=np.zeros(max_nei*self.mesh.numnodes*self.dofs**2,dtype=np.int16)
            cols=np.zeros(max_nei*self.mesh.numnodes*self.dofs**2,dtype=np.int16)
            data=np.zeros(max_nei*self.mesh.numnodes*self.dofs**2)

            #Vector for the rhs
            rhs=np.zeros(self.mesh.numnodes*self.dofs)

            #Count how many entries we have
            nnz=0

            for i,node1 in self.mesh.nodes.items():
                # Order things u1,v1,u2,v2,...
                # Still loop in the same way, just be careful with indexing
                # set things up for the diagonal for the first argument
                rows[nnz]=i-1 
                cols[nnz]=i-1

                # for the second argument
                rows[nnz+1]=i
                cols[nnz+1]=i

                # for the cross-term between the two components
                rows[nnz+2]=i
                cols[nnz+2]=i-1

                # for the other cross-term
                rows[nnz+3]=i-1
                cols[nnz+3]=i

                # Lazy, no checking for correct return from equation but so it goes
                data[nnz],data[nnz+1],data[nnz+2],data[nnz+3],rhs[i-1],rhs[i]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,kwargs=parkwargs)
                
                # increment things
                nnz += 4

                for j,node2_els in node1.neighbors.items():
                    # Do the off diagonals, do not assume symmetry

                    # The first component off-diagonal
                    rows[nnz]=i-1
                    cols[nnz]=j-1

                    # The second component off-diagonal
                    rows[nnz+1]=i
                    cols[nnz]=j

                    # for the cross-term between the two components
                    rows[nnz+2]=i
                    cols[nnz+2]=j-1

                    # for the other cross-term
                    rows[nnz+3]=i-1
                    cols[nnz+3]=j

                    # Again, we hope the return from this equation is good
                    data[nnz],data[nnz+1],data[nnz+2],data[nnz+3]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],max_nei=max_nei,kwargs=parkwargs)

                    # increment again
                    nnz += 4

            # set up our matrix for real
            self.matrix=csc_matrix((data,(rows,cols)),shape=(self.mesh.numnodes*self.dofs,self.mesh.numnodes*self.dofs))
            self.rhs=rhs
            return None

        else:
            raise ValueError('Cannnot do more than 2 dofs')


        
    def applyBCs(self,time=None):
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
                    print('You list a non existent border '+str(edge)+' in types')
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
                    print('You list a non existent border '+str(edge)+' in types')
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
                    self.applyNeumann(edge_nodes[edge],b_funcs[edge],normal=True,time=time)
                else:
                    self.applyNeumann(edge_nodes[edge],b_funcs[edge],normal=True)
            except KeyError: # If we have no condition we are just taking 0 neumann
                pass

        for edge in dirichlet:
            try:
                if time is not None:
                    self.applyDirichlet(edge_nodes[edge],b_funcs[edge],time=time)
                else:
                    self.applyDirichlet(edge_nodes[edge],b_funcs[edge])
            except KeyError:
                self.applyDirichlet(edge_nodes[edge],lambda x:0) # We actually need to do something to implement a zero
                # maybe throw the error though?


    def applyNeumann(self,edge_nodes,function,normal=True,flux=True,time=None): #TODO make non-normal stuff, non-flux  possible
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
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]),time)[0]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node] = self.rhs[2*node]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1])[1],time)*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

                                    else: # Non-normal, time-dependent, 2dofs
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]),time)[0],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node] = self.rhs[2*node]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]),time)[1],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                else:
                                    if normal: # Normal, steady state, 2dofs
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))[0]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node] = self.rhs[2*node]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))[1]*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

                                    else: # Non-normal, steady state, 2dofs
                                        self.rhs[2*node-1] = self.rhs[2*node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]))[0],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                                        self.rhs[2*node] = self.rhs[2*node]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1]))[1],self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])

        else:
            raise ValueError('Cannot do more than 2 dofs')


    def applyDirichlet(self,edge_nodes,function,time=None):
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
                # We now have 4 terms relating to this element itself
                self.matrix[2*node-1,2*node-1]=1.0
                self.matrix[2*node,2*node]=1.0
                self.matrix[2*node,2*node-1]=0.0
                self.matrix[2*node-1,2*node]=0.0

                # Set the values on the right hand side
                if time is not None:
                    self.rhs[2*node-1]=function(self.mesh.nodes[node].coords(),time)[0]
                    self.rhs[2*node]=function(self.mesh.nodes[node].coords(),time)[1]
                else:
                    self.rhs[2*node-1]=function(self.mesh.nodes[node].coords())[0]
                    self.rhs[2*node]=function(self.mesh.nodes[node].coords())[1]

                # zero out the off-diagonal elements to get the condition correct
                # and keep symmetry if we have it
                for j in self.mesh.nodes[node].neighbors.keys(): # Get the neighboring nodes
                    if not j in edge_nodes: 
                        # Check if this neighboring node is on the edge

                        # We have four elements to zero out
                        self.rhs[2*j-1]=self.rhs[2*j-1]-self.matrix[2*j-1,2*node-1]*self.rhs[2*node-1]
                        self.rhs[2*j]=self.rhs[2*j]-self.matrix[2*j,2*node]*self.rhs[2*node]
                        # Cross-terms
                        self.rhs[2*j-1]=self.rhs[2*j-1]-self.matrix[2*j-1,2*node]*self.rhs[2*node]
                        self.rhs[2*j]=self.rhs[2*j]-self.matrix[2*j,2*node-1]*self.rhs[2*node-1]

                    # zero out each of these, and also the symmetric part
                    # all u
                    self.matrix[2*node-1,2*j-1]=0.0
                    self.matrix[2*j-1,2*node-1]=0.0 
                    # all v 
                    self.matrix[2*node,2*j]=0.0
                    self.matrix[2*j,2*node]=0.0
                    # uv
                    self.matrix[2*node-1,2*j]=0.0
                    self.matrix[2*j,2*node-1]=0.0
                    # vu
                    self.matrix[2*node,2*j-1]=0.0
                    self.matrix[2*j-1,2*node]=0.0

        else:
            raise ValueError('Cannot do more than 2 dofs')


    def solveIt(self,method='BiCGStab',precond='LU',tolerance=1.0e-5):
        """Do some linear algebra"""
        if not method=='direct':
            if precond=='LU':
                p=spilu(self.matrix, drop_tol=1.0e-5)
                M_x=lambda x: p.solve(x)
                M=LinearOperator((self.mesh.numnodes,self.mesh.numnodes),M_x)
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



    def plotSolution(self,threeD=True,savefig=None,show=False,x_steps=20,y_steps=20,cutoff=5,savesol=False,figsize=(15,10)):
        if self.dofs==1:
            mat_sol=self.sparse2mat(x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
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
                    raise ValueError('savefig must be list of strings same length as dofs')

            mat_sol=self.sparse2mat(x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
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
           


    def sparse2mat(self, x_steps=500, y_steps=500, cutoff_dist=2000.0):
        """Grid up some sparse, potentially concave data"""
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
        self.MakeMatrixEQ(max_nei=max_nei,parkwargs=kwargs)
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
    """A class for performing the solves on a nonlinear model, with the same method names"""
    kind='NonLinear'


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
        self.sol=self.solveIt()


    def __str__(self):
        return 'Time dependent model with '+str(self.n_steps)+' time steps'


    def solveIt(self):
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

    def plotAnimate(self,num,ax):
        iterate=0
        while iterate<=self.n_steps:
            iterate += 1
            yield ax.plot_trisurf(self.model.mesh.coords[:,0],self.model.mesh.coords[:,1],Z=self.sol[iterate],cmap=cm.jet)

    def animate(self,show=False,save=None):
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
        """Grid up some sparse, potentially concave data"""
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
    """Error for bad iterative method result"""


    def __init__(self,method=None,iters=None):
        self.method=method
        self.iters=iters


    def __str__(self):
        return 'Method '+self.method+' did not converge at iteration '+str(self.iters)


def main():
    import equationsFEM
    mo=Model('testmesh.msh')
    mo.add_equation(equationsFEM.diffusion())
    mo.add_BC('dirichlet',1,lambda x: 10.0)
    mo.add_BC('neumann',2,lambda x:-1.0) # 'dirichlet',2,lambda x: 10.0)
    mo.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
    mo.add_BC('neumann',4,lambda x:0.0)
    m=LinearModel(mo)
    m.iterate()


    admo=Model('testmesh.msh')
    admo.add_equation(equationsFEM.advectionDiffusion())
    admo.add_BC('dirichlet',1,lambda x: 15.0)
    admo.add_BC('neumann',2,lambda x:0.0) # 'dirichlet',2,lambda x: 10.0)
    admo.add_BC( 'dirichlet',3,lambda x: 5.0)
    admo.add_BC('neumann',4,lambda x:0.0)
    am=LinearModel(admo)
    am.iterate(v=lambda x:np.array([10.0,0.0]))

    mod=Model('testmesh.msh',td=True)
    mod.add_equation(equationsFEM.diffusion())
    mod.add_BC('dirichlet',1,lambda x,t: 26.0)
    mod.add_BC('neumann',2,lambda x,t:0.0) # 'dirichlet',2,lambda x: 10.0)
    mod.add_BC( 'dirichlet',3,lambda x,t: 26.0)
    mod.add_BC('neumann',4,lambda x,t:0.0)
    mi=TimeDependentModel(mod,10.0,60,lambda x:1+(x[0]-5)**2)
    mi.animate(show=False,save='decay.mp4')
    return m,am,mi


if __name__ == '__main__':
    main()
