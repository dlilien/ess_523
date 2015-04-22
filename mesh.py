#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
Define a mesh classes that have info I might want about finite element meshes
"""

import numpy as np
from scipy.linalg import solve,qr,svd,lstsq

def main():
    """A callable version for debugging"""
    m=Mesh()
    m.loadgmsh('roughshear_bl.msh')
    m.CreateBases()
    return m

class Node:
    """A node with x,y,z coordinates"""
    _curr_id=0
    def __init__(self,x,y,z=0.0,ident=None):
        self.ass_elms=[]
        self.x=float(x)
        self.y=float(y)
        self.z=float(z)
        if ident is not None:
            self.id=ident
            Node._curr_id=max(Node._curr_id,self.id)
        else:
            self.id=Node._curr_id
            Node._curr_id+=1
    def add_elm(self,elm):
        self.ass_elms.append(elm)
    def coords(self):
        return [self.x,self.y,self.z]
    def __str__(self):
        return 'Node number '+str(self.id)+'at ('+str(self.x)+','+str(self.y)+','+str(self.z)+')\nAssociate with elements '+', '.join(map(str,self.ass_elms))

class Element(object):
    """A single finite element, of given type"""
    _curr_id=0
    @staticmethod
    def init_element_gmsh(params,parent=None):
        ntags=params[2]
        nodes=params[(3+ntags):]
        kwargs={}
        if ntags>=1:
            kwargs['physents']= params[3]
            if ntags>=2:
                kwargs['geoents']=params[4]
                if ntags>=3:
                    kwargs['npartits']=params[5]
        if params[1]==1:
            return LineElement(nodes,ident=params[0],parent=parent,skwargs=kwargs)
        elif params[1]==2:
            return TriangElement(nodes,ident=params[0],parent=parent,skwargs=kwargs)
        else:
            print 'Unknown element type'
    def __str__(self):
        string= 'Element Number '+str(self.id)+'\nType: '+str(self.kind)+'('+str(self.eltypes)+')\nAssociated with nodes '+', '.join([str(node) for node in self.nodes])+'\nAnd with physical element '
        string+=str(self.physents) if hasattr(self,'physents') else 'None'
        return string
    def pvecs(self):
        """Return x,y vectors for plotting"""
        ncoords=[[],[]]
        for node in self.nodes:
            ncoords[0].append(self.parent.nodes[node].x)
            ncoords[1].append(self.parent.nodes[node].y)
        if len(self.nodes)>2:
            ncoords[0].append(ncoords[0][0])
            ncoords[1].append(ncoords[1][0])
        return ncoords[0],ncoords[1]
          
class TriangElement(Element):
    """A single triangular finite element"""
    kind='Triangular'
    eltypes=2
    
    @classmethod
    def _F(cls,p1,p2,p3):
        """This guy is useless, but whatever"""
        try:
            return lambda p: np.dot(np.array([[p2[0]-p1[0],p3[0]-p1[0]],[p2[1]-p1[1],p3[1]-p1[1]]]),np.array(p).reshape((len(p),1))-np.array([[p1[0]],[p1[1]]]))
        except TypeError:
            print 'Bad points for triangular element'
            return None

    @classmethod
    def _Finv(cls,p1,p2,p3):
        """Map from element to right triangle at origin"""
        try:
            return lambda p: solve(np.array([[p2[0]-p1[0],p3[0]-p1[0]],[p2[1]-p1[1],p3[1]-p1[1]]]),np.array(p).reshape((len(p),1))-np.array([[p1[0]],[p1[1]]]))
        except TypeError:
            print 'Bad points for triangular element'
            return None

    @classmethod
    def _b1(cls,Finv):
        """Define basis function 1 using map from element to origin"""
        def b1(p):
            Fi=Finv(p)
            return 1-Fi[0]-Fi[1]
        return b1

    @classmethod
    def _b2(cls,Finv):
        """Define basis function 2 using map from element to origin"""
        def b1(p):
            Fi=Finv(p)
            return Fi[0]
        return b1

    @classmethod
    def _b3(cls,Finv):
        """Define basis function 3 using map from element to origin"""
        def b1(p):
            Fi=Finv(p)
            return Fi[1]
        return b1

    def __init__(self,nodes,ident,parent,skwargs):
        if not len(nodes)==3:
            print 'Bad Triangle'
            return None
        if ident is not None:
            self.id=ident
            Element._curr_id=max(Element._curr_id,self.id)
        else:
            self.id=Element._curr_id
            Element._curr_id+=1
        self.parent=parent
        self.nodes=nodes
        self.kind=TriangElement.kind
        self.eltypes=TriangElement.eltypes
        for tag,val in skwargs.items():
            setattr(self,tag,val)
    def _bases(self,*args):
        Fi=TriangElement._Finv(*args)
        return [TriangElement._b1(Fi),TriangElement._b2(Fi),TriangElement._b2(Fi)]


class LineElement(Element):
    """A single line finite element"""
    kind='Line'
    eltypes=1

    @classmethod
    def _b1(cls,x1,x2):
        return lambda x: x/(float(x2)-x1)
    @classmethod
    def _b2(cls,x1,x2):
        return lambda x: (float(x2)-x)/(x2-x1)
    def _bases(self,*args):
        return [LineElement._b1(args[0][0],args[1][0]),LineElement._b2(args[0][0],args[0][0])]

    def __init__(self,nodes,ident,parent,skwargs):
        if not len(nodes)==2:
            print 'Bad Line'
            return None
        if ident is not None:
            self.id=ident
            Element._curr_id=max(Element._curr_id,self.id)
        else:
            self.id=Element._curr_id
            Element._curr_id+=1
        self.parent=parent
        self.nodes=nodes
        self.kind=LineElement.kind
        self.eltypes=LineElement.eltypes
        for tag,val in skwargs.items():
            setattr(self,tag,val)

class Mesh:
    """A finite element mesh"""
    def __init__(self):
        self.elements={}
        self.nodes={}
        self.bases={}
        self.eltypes={}
        self.physents={}
        self.npartits={}
        self.numnodes=0
        self.numels=0
    def __str__(self):
        string='Mesh object\nNumber of nodes: '+str(self.numnodes)+'\nNumber of elements: '+str(self.numels)+'\nElement types: \n'
        for key in self.eltypes.keys():
            string+=str(len(self.eltypes[key]))+' elements of type '+str(key)+'\n'
        string+=str(len(self.physents.keys()))+' physical entities\n'
        if self.bases:
            string+='Bases formed'
        else:
            strin+='No Bases Associated'
        return string
    def loadgmsh(self,fn):
        with open(fn,'r') as f:
            flines=f.readlines()
        if not flines[0]=='$MeshFormat\n':
            print 'Unrecognized msh file'
            return False
        self.types=map(float,flines[1].split())
        self.numnodes=int(flines[4])
        self.nodes={int(line[0]): Node(*map(float,line[1:4]), ident=int(line[0])) for line in map(str.split,flines[5:(5+self.numnodes)])}
        if not flines[self.numnodes+6]=='$Elements\n':
            print 'Unrecognized msh file'
            return False
        self.numels=int(flines[self.numnodes+7])
        self.elements={int(line[0]): Element.init_element_gmsh(map(int,line),parent=self) for line in map(str.split,flines[(8+self.numnodes):(8+self.numnodes+self.numels)])}
        for key in self.elements.keys():
            for attr in ['eltypes','physents','geoents','npartits']:
                try:
                    param=getattr(self.elements[key],attr)
                except AttributeError:
                    pass
                try:
                    paramlist=getattr(self,attr)
                    if not param in paramlist:
                        paramlist[param]=[]
                    paramlist[param].append(key)
                except AttributeError:
                    pass
            for node in self.elements[key].nodes:
                self.nodes[node].add_elm(key)
        flines=None
    def CreateBases(self):
        """Create the finite element basis functions"""
        self.bases={}
        for number,element in self.elements.items():
            ps=[self.nodes[i].coords() for i in element.nodes]
            self.bases[number]={i:fnctn for i,fnctn in enumerate(element._bases(*ps))}
            

class Mesh2D(Mesh):
    kind='2D'


if __name__=='__main__':
    main()
