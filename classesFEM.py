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
from scipy.sparse.linalg import bicgstab,cg,spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from warnings import warn
from os.path import splitext
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata
Axes3D # Avoid the warning



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


class model:
    """A steady state model, with associated mesh, BCs, and equations"""
    def __init__(self,*mesh):
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
    

    def add_equation(self,eqn):
        self.eqn=eqn


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
            try:
                function(self.mesh.nodes[self.mesh.elements[self.mesh.physents[target_edge][0]].nodes[0]].coords())
            except:
                raise TypeError('Not a usable function, must take vector input and time')
        self.BCs[target_edge]=(cond_type,function)


class ModelIterate:
    """This object makes matrix, forms a solution, etc"""


    def __init__(self,model,*eqn):
        self.parent=model
        self.mesh=self.parent.mesh
        if eqn:
            self.eqn=eqn
        else:
            self.eqn=self.parent.eqn
    

    def MakeMatrixEQ(self,max_nei=12,**kwargs):
        """Make the matrix form, max_nei is the most neighbors/element"""
        # We can ignore trailing zeros as long as we allocate space
        # I.e. go big with max_nei
        #TODO fix equation handling...
        #TODO automatic max_nei
        rows=np.zeros(max_nei*self.mesh.numnodes,dtype=np.int16)
        cols=np.zeros(max_nei*self.mesh.numnodes,dtype=np.int16)
        data=np.zeros(max_nei*self.mesh.numnodes)
        rhs=np.zeros(self.mesh.numnodes)
        nnz=0
        for i,node1 in self.mesh.nodes.items():
            rows[nnz]=i-1 #TODO
            cols[nnz]=i-1 #TODO
            data[nnz],rhs[i-1]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,kwargs=kwargs) #TODO fix indexing, bases
            nnz += 1
            for j,node2_els in node1.neighbors.items():
                rows[nnz]=i-1 #TODO
                cols[nnz]=j-1 #TODO
                data[nnz]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],max_nei=max_nei,kwargs=kwargs) #TODO fix indexing, bases, the 1!!!!!
                nnz += 1
        self.matrix=csc_matrix((data,(rows,cols)),shape=(self.mesh.numnodes,self.mesh.numnodes)) #TODO fix indexing
        self.rhs=rhs
        return None

        
    def applyBCs(self):
        Mesh=self.mesh
        dirichlet=[edgeval[0] for edgeval in self.parent.BCs.items() if edgeval[1][0]=='dirichlet']
        neumann=[edgeval[0] for edgeval in self.parent.BCs.items() if edgeval[1][0]=='neumann']
        b_funcs={edgeval[0]:edgeval[1][1] for edgeval in self.parent.BCs.items()}
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
                self.applyNeumann(edge_nodes[edge],b_funcs[edge],normal=True)
            except KeyError: # If we have no condition we are just taking 0 neumann
                pass

        for edge in dirichlet:
            try:
                self.applyDirichlet(edge_nodes[edge],b_funcs[edge])
            except KeyError:
                self.applyDirichlet(edge_nodes[edge],lambda x:0) # We actually need to do something to implement a zero
                # maybe throw the error though?


    def applyNeumann(self,edge_nodes,function,normal=True,flux=True): #TODO make non-normal stuff, non-flux  possible
        """Apply a natural boundary condition, must be normal"""
        for node in edge_nodes:
            for j,els in self.mesh.nodes[node].neighbors.items():
                if j in edge_nodes:
                    for k,el in enumerate(els):
                        if self.mesh.elements[el].kind=='Line':
                            if not flux:
                                raise TypeError('You need to specify the BC as a flux (e.g. divide out k in diffusion)')
                            if normal:
                                self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*function(self.mesh.elements[el].F(gpt[1:-1]))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])
                            else:
                                self.rhs[node-1] = self.rhs[node-1]- np.sum([self.mesh.elements[el].length*gpt[0]*(np.dot(function(self.mesh.elements[el].F(gpt[1:-1])),self.elements[el].normal))*self.mesh.elements[el].bases[self.mesh.elements[el].nodes.index(node)](self.mesh.elements[el].F(gpt[1:-1])) for gpt in self.mesh.elements[el].gpoints])



    def applyDirichlet(self,edge_nodes,function):
        """Let's apply an essential boundary condition"""
        for node in edge_nodes:
            self.matrix[node-1,node-1]=1.0
            self.rhs[node-1]=function(self.mesh.nodes[node].coords())
            for j in self.mesh.nodes[node].neighbors.keys(): # Get the neighboring nodes
                if not j in edge_nodes: 
                    # Check if this neighboring node is on the edge
                    self.rhs[j-1]=self.rhs[j-1]-self.matrix[j-1,node-1]*self.rhs[node-1]
                self.matrix[node-1,j-1]=0.0
                self.matrix[j-1,node-1]=0.0   


    def solveIt(self,method='BiCGStab',precond=None,tolerance=1.0e-5):
        """Do some linear algebra"""
        if method=='CG':
                self.sol,info=cg(self.matrix,self.rhs,tol=tolerance)
                if info>0:
                    warn('Conjugate gradient did not converge. Attempting BiCGStab')
                    self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance)
                    if info>0:
                        raise ConvergenceError(method='CG and BiCGStab',iters=info)
        elif method=='BiCGStab':
            self.sol,info=bicgstab(self.matrix,self.rhs,tol=tolerance)
            if info>0:
                raise ConvergenceError(method=method,iters=info)
        elif method=='direct':
            self.sol=spsolve(self.matrix,self.rhs)
        else:
            raise TypeError('Unknown solution method')
        return self.sol


    def plotSolution(self,threeD=True,savefig=None,show=False,x_steps=20,y_steps=20,cutoff=5,savesol=False,figsize=(15,10)):
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


    def sparse2mat(self, x_steps=500, y_steps=500, cutoff_dist=2000.0):
        """Grid up some sparse, potentially concave data"""
        coords=self.mesh.coords
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


class ConvergenceError(Exception):
    """Error for bad iterative method result"""
    def __init__(self,method=None,iters=None):
        self.method=method
        self.iters=iters
    def __str__(self):
        return 'Method '+self.method+' did not converge at iteration '+str(self.iters)


def main():
    import equationsFEM
    mod=model('testmesh.msh')
    mod.add_equation(equationsFEM.diffusion)
    mod.add_BC('dirichlet',1,lambda x: 10.0)
    mod.add_BC('neumann',2,lambda x:-1.0) # 'dirichlet',2,lambda x: 10.0)
    mod.add_BC( 'dirichlet',3,lambda x: abs(x[1]-5.0)+5.0)
    mod.add_BC('neumann',4,lambda x:0.0)
    mi=ModelIterate(mod)
    mi.MakeMatrixEQ()#f=lambda x:(5.0-abs(x[0]-5.0))*(5.0-abs(x[1]-5.0)),k=lambda x:10.0)
    mi.applyBCs()
    mi.solveIt(method='CG')
    return mi


if __name__ == '__main__':
    main()
