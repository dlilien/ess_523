#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.

"""
A class to do operations on Meshes (form fem matrix, solve things?)
"""

import numpy as np
from scipy.sparse import csc_matrix
import equationsFEM
import mesh
import matplotlib.pyplot as plt
from warnings import warn


def MakeMatrixEQ(Mesh,eqn=equationsFEM.diffusion,max_nei=8,**kwargs):
    """Make the matrix form, max_nei is the most neighbors/element"""
    # We can ignore trailing zeros as long as we allocate space
    # I.e. go big with max_nei
    #TODO fix equation handling...
    #TODO automatic max_nei
    rows=np.zeros(max_nei*Mesh.numnodes,dtype=np.int16)
    cols=np.zeros(max_nei*Mesh.numnodes,dtype=np.int16)
    data=np.zeros(max_nei*Mesh.numnodes)
    rhs=np.zeros(Mesh.numnodes)
    nnz=0
    for i,node1 in Mesh.nodes.items():
        rows[nnz]=i-1 #TODO
        cols[nnz]=i-1 #TODO
        data[nnz],rhs[i-1]=eqn(i,i,[(elm[0],Mesh.elements[elm[0]]) for elm in node1.ass_elms if Mesh.elements[elm[0]].eltypes==2],max_nei=max_nei,rhs=True,kwargs=kwargs) #TODO fix indexing, bases
        nnz += 1
        for j,node2_els in node1.neighbors.items():
            rows[nnz]=i-1 #TODO
            cols[nnz]=j-1 #TODO
            data[nnz]=eqn(i,j,[(1,Mesh.elements[nei_el]) for nei_el in node2_els if Mesh.elements[nei_el].eltypes==2],max_nei=max_nei) #TODO fix indexing, bases, the 1!!!!!
            nnz += 1
    Mesh.matrix=csc_matrix((data,(rows,cols)),shape=(Mesh.numnodes,Mesh.numnodes)) #TODO fix indexing
    Mesh.rhs=rhs
    return None

def applyBCs(Mesh,dirichlet=[],neumann=[],b_funcs={}):
    """Go in and modify the matrix and rhs to comply with BCs"""
    # Just wrap Neumann and Dirichlet methods and add some error checking

    # Let's do some checking/warning about our inputs
    edges = np.sort(Mesh.physents.keys())[0:-1]
    listed_edges=np.sort(dirichlet+neumann)
    if not all(edges==listed_edges):
        warn('Error with borders')
        for edge in listed_edges:
            if not edge in edges:
                print 'You list a non existent border '+str(edge)+' in types'
                print 'Available borders are ',edges
                raise ValueError('Unknown border')
        else:
            print 'Some border not specified in types, taking Neumann'
            print 'Borders are ',edges,' listed are ',listed_edges


    # minor checking which we will warn but ignore
    if not all(edges==np.sort(b_funcs.keys())):
        warn('Error with borders')
        for edge in b_funcs.keys():
            if not edge in edges:
                print 'You list a non existent border '+str(edge)+' in types'
                print 'Available borders are ',edges
                raise ValueError ('Unknown border')
        else:
            print 'Some border not specified in types, taking equal to zero'
            print 'Borders are ',edges,' listed are ',b_funcs.keys()

    # Ok, hopefully we have parse-able input now
    for edge in dirichlet:
        try:
            applyDirichlet(Mesh,edge,b_funcs[edge])
        except KeyError:
            applyDirichlet(Mesh,edge,lambda x:0)

    for edge in neumann:
        try:
            applyNeumann(Mesh,edge,b_funcs[edge])
        except KeyError:
            pass

def applyDirichlet(Mesh,edge,function):
    pass

def applyNeumann(Mesh,edge,function):
    pass




def main():
    """A callable version for debugging"""
    tm = mesh.Mesh()
    tm.loadgmsh('testMesh.msh')
    tm.CreateBases()
    MakeMatrixEQ(tm,f=lambda x:1.0)
    plt.spy(tm.matrix)
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



