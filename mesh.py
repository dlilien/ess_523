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
import matplotlib.pyplot as plt



def main():
    """A callable version for debugging"""
    tm = Mesh()
    tm.loadgmsh('testmesh.msh')
    tm.CreateBases()
    plt.savefig('spy.eps')
    return tm


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
                if node not in self.neighbors.keys():
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
            print 'Unknown element type'

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
            print 'Bad Triangle'
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
        for tag, val in skwargs.items():
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
        self.bases = [self._b1(pts), self._b2(pts)]
        return self.bases

    def _dbases(self):
        pts=self.xyvecs()
        self.dbases= [[self.bases[0](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[1.0],[0.0]])),self.bases[0](np.array(pts[0]).reshape(len(pts[0]),1)+np.array([[0.0],[1.0]]))],[self.bases[1](np.array(pts[1]).reshape(len(pts[1]),1)+np.array([[1.0],[0.0]])),self.bases[1](np.array(pts[1]).reshape(len(pts[1]),1)+np.array([[0.0],[1.0]]))]]
        return self.dbases

    def __init__(self, nodes, ident, parent, skwargs):
        if not len(nodes) == 2:
            print 'Bad Line'
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
        for tag, val in skwargs.items():
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
        for key in self.eltypes.keys():
            string += str(len(self.eltypes[key])) + \
                ' elements of type ' + str(key) + '\n'
        string += str(len(self.physents.keys())) + ' physical entities\n'
        if self.bases:
            string += 'Bases formed'
        else:
            string += 'No Bases Associated'
        return string

    def loadgmsh(self, fn):
        with open(fn, 'r') as f:
            flines = f.readlines()
        if not flines[0] == '$MeshFormat\n':
            print 'Unrecognized msh file'
            return False
        self.types = map(float, flines[1].split())
        self.numnodes = int(flines[4])
        self.nodes = {int(line[0]): Node(*map(float, line[1:4]), ident=int(line[0]),parent=self)
                      for line in map(str.split, flines[5:(5 + self.numnodes)])}
        if not flines[self.numnodes + 6] == '$Elements\n':
            print 'Unrecognized msh file'
            return False
        self.numels = int(flines[self.numnodes + 7])
        self.elements = {int(line[0]): Element.init_element_gmsh(map(int, line), parent=self) for line in map(
            str.split, flines[(8 + self.numnodes):(8 + self.numnodes + self.numels)])}
        for key in self.elements.keys():
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
        self.coords=np.r_[[node.coords()[0:-1] for node in self.nodes.values()]]

    def CreateBases(self,gpts=True):
        """Create the finite element basis functions"""
        self.bases = {}
        self.dbases = {}
        if gpts:
            for number, element in self.elements.items():
                self.bases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._bases())}
                self.dbases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._dbases())}
                element._gpts()
        else:
            for number, element in self.elements.items():
                self.bases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._bases())}
                self.dbases[number] = {
                    i: fnctn for i, fnctn in enumerate(element._dbases())}

    def PlotBorder(self, show=False, writefile=None, axis=None, fignum=None):
        """Plot out the border of the mesh with different colored borders"""
        if fignum is not None:
            plt.figure(fignum)
        else:
            plt.figure()
        colors = ['b', 'k', 'r', 'c', 'g', 'm', 'darkred', 'darkgreen',
                  'darkslategray', 'saddlebrown', 'darkorange', 'darkmagenta', 'y']
        plot_these = np.sort(self.physents.keys())[0:-1]
        plt_lines = {key: None for key in plot_these}
        for i, key in enumerate(plot_these):
            for j, element in enumerate(self.physents[key]):
                if j == 0:
                    plt_lines[key], = plt.plot(
                        *self.elements[element].pvecs(), color=colors[i], label=str(key))
                else:
                    plt.plot(*self.elements[element].pvecs(), color=colors[i])
        plt.legend(
            plt_lines.values(), ['Border ' + num for num in map(str, plot_these)])
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
        plot_these = self.physents.keys()
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
        if show:
            plt.show()
        if writefile is not None:
            plt.savefig(writefile)


if __name__ == '__main__':
    main()
