#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
Break off the meshing stuff into a separate file because of clutter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


class Node:

    """An individual node on a mesh

    Attributes
    ----------
    ass_elms : list
       Elements surrounding this node
    neighbors : dict
       Dictionary of Node:[elements] where the elements are shared between the nodes
    parent : Mesh
       The mesh with which this node is associated
    phys_vars : dict
       A container for physical variables where they won't pollute the namespace
    x : float
       The x coordinate
    y : float
       The y coordinate
    id : int
       The number of this element
    """
    curr_id = 0

    def __init__(self, x, y, z=0.0, ident=None, parent=None):
        self.ass_elms = []
        self.neighbors = {}  # Dictionary of nodes and the connecting elements
        self.parent = parent  # The mesh with which this node is associated
        # Put physical variables here so namespace isn't polluted
        self.phys_vars = {}
        self.x = x
        self.y = y
        if ident is not None:
            self.id = ident
            Node.curr_id = max(Node.curr_id, self.id)
        else:
            self.id = Node.curr_id
            Node.curr_id += 1

    def __str__(self):
        return 'Node number ' + str(self.id) + 'at (' + str(self.x) + ',' + str(
            self.y) + ')\nAssociate with elements ' + ', '.join(map(str, self.ass_elms))

    def add_elm(self, elm, pos):
        """Add an element using this node, with this as the pos'th node"""
        for node in self.parent.elements[elm].nodes:
            if node != self.id:
                if node not in list(self.neighbors.keys()):
                    self.neighbors[node] = [elm]
                else:
                    self.neighbors[node].append(elm)
        self.ass_elms.append([elm, pos])

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
            return LineElement(
                nodes,
                ident=params[0],
                parent=parent,
                skwargs=kwargs)
        elif params[1] == 2:
            return TriangElement(
                nodes,
                ident=params[0],
                parent=parent,
                skwargs=kwargs)
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
        self.gpts = [(pt[0], self._F()(pt[1:3])) for pt in self.gpoints]


class TriangElement(Element):

    """A single triangular finite element

    A subclass of :py:class:`Element` with the same properties
    """
    kind = 'Triangular'
    eltypes = 2
    gpoints = [[-27.0 / 96.0, 1.0 / 3.0, 1.0 / 3.0],
                 [25.0 / 96.0, 0.2, 0.6],
                 [25.0 / 96.0, 0.6, 0.2],
                 [25.0 /96.0, 0.2, 0.2]]

    def _F(self):
        """Right triangle to element mapping"""
        if self.F is None:
            ps = self.xyvecs()
            self.F = lambda p: np.dot([[ps[1][0] - ps[0][0],
                                        ps[2][0] - ps[0][0]],
                                       [ps[1][1] - ps[0][1],
                                        ps[2][1] - ps[0][1]]],
                                      np.array(p).reshape(len(p),
                                                          1)) + np.array([[ps[0][0]],
                                                                          [ps[0][1]]])
        return self.F

    def _Finv(self):
        """Map from element to right triangle at origin"""
        if self.Finv is None:
            ps = self.xyvecs()
            self.area = abs((ps[1][0] - ps[0][0]) * (ps[2][1] - ps[0][1]) 
                           - (ps[2][0] - ps[0][0]) * (ps[1][1] - ps[0][1])) / 2.0
            self.Finv = lambda p: solve(
                    np.array([[ps[1][0] - ps[0][0], ps[2][0] - ps[0][0]],
                                [ps[1][1] - ps[0][1], ps[2][1] - ps[0][1]]]),
                    np.array(p).reshape(len(p), 1) 
                        - np.array([[ps[0][0]], [ps[0][1]]]))
        return self.Finv

    def _normal(self):
        """Dummy to be lazy"""
        pass

    def _b1(self, Finv):
        """Define basis function 1 using map from element to origin"""
        def b1(p):
            Fi = Finv(p)
            return 1 - Fi[0] - Fi[1]
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
        self.F = None
        self.Finv = None
        self.parent = parent
        self.nodes = nodes
        self.kind = TriangElement.kind
        self.eltypes = TriangElement.eltypes
        self.phys_vars = {}
        for tag, val in list(skwargs.items()):
            setattr(self, tag, val)
        ps = self.xyvecs()
        self.cent = [
            (ps[0][0] + ps[1][0] + ps[2][0]) / 3.0,
            (ps[0][1] + ps[1][1] + ps[2][1]) / 3]

    def _bases(self):
        self._Finv()
        self._F()
        self.bases = [
            self._b1(
                self.Finv), self._b2(
                self.Finv), self._b3(
                self.Finv)]
        return self.bases

    def _dbases(self):
        pts = self.xyvecs()
        self.dbases = [[self.bases[0](np.array(pts[1]).reshape(len(pts[0]), 1)
                            + np.array([[1.0], [0.0]])), 
                        self.bases[0](np.array(pts[1]).reshape(len(pts[0]), 1) 
                            + np.array([[0.0], [1.0]]))], 
                        [self.bases[1](np.array(pts[0]).reshape(len(pts[0]), 1) 
                            + np.array([[1.0], [0.0]])), 
                         self.bases[1](np.array(pts[0]).reshape(len(pts[0]), 1)
                            + np.array([[0.0], [1.0]]))],
                         [self.bases[2](np.array(pts[0]).reshape(len(pts[0]), 1) 
                            + np.array([[1.0], [0.0]])),
                          self.bases[2](np.array(pts[0]).reshape(len(pts[0]), 1) 
                            + np.array([[0.0], [1.0]]))]]
        return self.dbases


class LineElement(Element):

    """A single line finite element

    A subclass of :py:class:`Element` with the same attributes.
    """
    kind = 'Line'
    eltypes = 1
    gpoints = [[0.5, (1.0 - 1.0 / np.sqrt(3.0)) / 2.0, 0],
               [0.5, (1.0 + 1.0 / np.sqrt(3.0)) / 2.0, 0]]

    def _F(self):
        if self.F is None:
            ps = self.xyvecs()
            self.F = lambda p: np.array(
                [ps[0][0] + p[0] * (ps[1][0] - ps[0][0]),
                 ps[0][1] + p[0] * (ps[1][1] - ps[0][1])])
        return self.F

    def _Finv(self):
        pts = self.xyvecs()
        return lambda p: np.array([[pts[0][0] + (pts[0][0] - pts[1][0]) * p[0]], 
                                    [pts[0][1] + (pts[0][1] - pts[1][1]) * p[0]]])

    def _b2(self, pts):
        if pts[1][0] == pts[0][0]:
            return lambda x: (x[1] - float(pts[0][1])) / \
                (float(pts[1][1]) - pts[0][1])
        else:
            return lambda x: (x[0] - float(pts[0][0])) / \
                (float(pts[1][0]) - pts[0][0])

    def _b1(self, pts):
        if pts[1][0] == pts[0][0]:
            return lambda x: (
                float(pts[1][1]) - x[1]) / (pts[1][1] - pts[0][1])
        else:
            return lambda x: (
                float(pts[1][0]) - x[0]) / (pts[1][0] - pts[0][0])

    def _bases(self, *args):
        pts = self.xyvecs()
        self._F()
        self.length = np.sqrt(
            np.sum([(pts[0][i] - pts[1][i])**2 for i in range(len(pts[0]))]))
        self.bases = [self._b1(pts), self._b2(pts)]
        return self.bases

    def _dbases(self):
        pts = self.xyvecs()
        self.dbases = [[self.bases[0](np.array(pts[1]).reshape(len(pts[1]), 1)
                            + np.array([[1.0], [0.0]])),
                        self.bases[0](np.array(pts[1]).reshape(len(pts[1]), 1)
                            + np.array([[0.0], [1.0]]))],
                       [self.bases[1](np.array(pts[0]).reshape(len(pts[0]), 1)
                            + np.array([[1.0], [0.0]])), 
                        self.bases[1](np.array(pts[0]).reshape(len(pts[0]), 1)
                            + np.array([[0.0], [1.0]]))]]
        return self.dbases

    def _normal(self):
        pts = self.xyvecs()
        self.normal = np.array(
            [pts[0][1] - pts[1][1], pts[1][0] - pts[0][0]]) / self.length
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
        self.phys_vars = {}
        self.F = None
        ps = self.xyvecs()
        self.cent = [(ps[0][0] + ps[1][0]) / 2, (ps[0][1] + ps[1][1]) / 2]
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
        self.phys_vars = {}
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
        self.nodes = {int(line[0]): Node(*list(map(float,
                                                   line[1:4])),
                                         ident=int(line[0]),
                                         parent=self) for line in map(str.split,
                                                                      flines[5:(5 + self.numnodes)])}
        if not flines[self.numnodes + 6] == '$Elements\n':
            print('Unrecognized msh file')
            return False
        self.numels = int(flines[self.numnodes + 7])
        self.elements = {
                int(line[0]):
                Element.init_element_gmsh(
                    list(map(int,line)),
                    parent=self)
                for line in map(str.split,flines[
                    (8 + self.numnodes):
                    (8 + self.numnodes +
                        self.numels)])}
        for key in list(self.elements.keys()):
            for attr in ['eltypes', 'physents', 'geoents', 'npartits']:
                try:
                    param = getattr(self.elements[key], attr)
                except AttributeError:
                    pass
                try:
                    paramlist = getattr(self, attr)
                    if param not in paramlist:
                        paramlist[param] = []
                    paramlist[param].append(key)
                except AttributeError:
                    pass
            for pos, node in enumerate(self.elements[key].nodes):
                self.nodes[node].add_elm(key, pos)
        flines = None
        self.coords = np.r_[[node.coords()[0:2]
                             for node in list(self.nodes.values())]]

    def CreateBases(self, gpts=True, normals=True):
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
        colors = [
            'b',
            'k',
            'r',
            'c',
            'g',
            'm',
            'darkred',
            'darkgreen',
            'darkslategray',
            'saddlebrown',
            'darkorange',
            'darkmagenta',
            'y',
            'b',
            'k',
            'r',
            'c',
            'g',
            'm',
            'darkred',
            'darkgreen',
            'darkslategray',
            'saddlebrown',
            'darkorange',
            'darkmagenta',
            'y']
        plot_these = np.sort(list(self.physents.keys()))[0:-1]
        plt_lines = {}
        for i, key in enumerate(plot_these):
            for j, element in enumerate(self.physents[key]):
                if j == 0:
                    plt_lines[key], = plt.plot(
                        *self.elements[element].pvecs(), color=colors[i], label='Border ' + str(key))
                else:
                    plt.plot(*self.elements[element].pvecs(), color=colors[i])
        plt.legend(handles=list(plt_lines.values()))
        if axis is not None:
            plt.axis(axis)
        if show:
            plt.show()
        if writefile is not None:
            plt.savefig(writefile)

    def PlotMesh(
            self,
            show=False,
            writefile=None,
            axis=None,
            labels=None,
            fignum=None):
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
                    *self.elements[element].pvecs(), color='b')
        if axis is not None:
            plt.axis(axis)
        if labels == 'area':
            for eln in self.eltypes[2]:
                plt.text(
                    self.elements[eln].cent[0],
                    self.elements[eln].cent[1],
                    '%1.1f' %
                    self.elements[eln].area)
        elif labels == 'number':
            for eln in self.eltypes[2]:
                plt.text(
                    self.elements[eln].cent[0],
                    self.elements[eln].cent[1],
                    '%d' %
                    eln)
        elif labels == 'edge_el_num':
            for eln in self.eltypes[1]:
                plt.text(
                    self.elements[eln].cent[0],
                    self.elements[eln].cent[1],
                    '%d' %
                    eln)

        if show:
            plt.show()
        if writefile is not None:
            plt.savefig(writefile)
