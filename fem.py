#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
A class to do operations on meshes (form fem matrix, solve things?)
"""

import numpy as np
from scipy.sparse import csc_matrix
import equationsFEM
import mesh
import matplotlib.pyplot as plt


def MakeMatrix(mesh,eqn=equationsFEM.area,max_nei=8):
    """Make the matrix form, max_nei is the most neighbors/element"""
    # We can ignore trailing zeros as long as we allocate space
    # I.e. go big with max_nei
    #TODO fix equation handling...
    #TODO automatic max_nei
    rows=np.zeros(max_nei*mesh.numnodes,dtype=np.int16)
    cols=np.zeros(max_nei*mesh.numnodes,dtype=np.int16)
    data=np.zeros(max_nei*mesh.numnodes)
    nnz=0
    for i,node1 in mesh.nodes.items():
        rows[nnz]=i-1 #TODO
        cols[nnz]=i-1 #TODO
        data[nnz]=eqn(areas=[mesh.elements[elm[0]].area for elm in node1.ass_elms if mesh.elements[elm[0]].eltypes==2],bases=None,dbases=None,gpoints=None) #TODO fix indexing, bases
        nnz += 1
        for j,node2_els in node1.neighbors.items():
            rows[nnz]=i-1 #TODO
            cols[nnz]=j-1 #TODO
            data[nnz]=eqn(areas=[mesh.elements[nei_el].area for nei_el in node2_els if mesh.elements[nei_el].eltypes==2],bases=None,dbases=None,gpoints=None) #TODO fix indexing, bases
            nnz += 1
    mesh.matrix=csc_matrix((data,(rows,cols)),shape=(mesh.numnodes,mesh.numnodes)) #TODO fix indexing

def main():
    """A callable version for debugging"""
    tm = mesh.Mesh()
    tm.loadgmsh('testmesh.msh')
    tm.CreateBases()
    MakeMatrix(tm)
    plt.savefig('spy.eps')
    return tm

def checkBases(me):
    """Just a quick check that the basis functions are zero and 1 where they should be"""
    badn=0
    for element in me.elements.values():
        for i,pt in enumerate(element.xyvecs()):
            if abs(element.bases[i](pt)-1)>1e-15:
                print 'Bad element ',element.id,'basis ',i
                badn+=1
    if badn==0:
        print 'Bases are good'



if __name__=='__main__':
    main()



