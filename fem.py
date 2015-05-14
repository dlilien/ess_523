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
from scipy.sparse.linalg import bicgstab,cg,spsolve
import equationsFEM
import mesh
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from warnings import warn
Axes3D


def main():
    """A callable version for debugging"""
    tm = mesh.Mesh()
    tm.loadgmsh('testmesh.msh')
    #tm.loadgmsh('roughshear_bl.msh')
    tm.CreateBases()
    MakeMatrixEQ(tm,f=lambda x:0.0,k=lambda x:1.00)
    applyBCs(tm,dirichlet=[1,3],neumann=[2,4],b_funcs={4:lambda x: 1.0,2:lambda x: 1.0,1:lambda x:1.0, 3:lambda x: (abs(x[1]-5.0)+5.0)})
    solveIt(tm,method='CG')
    plotSolution(tm,show=True,savesol=True)
    #plotSolution(tm, show=True, x_steps=500, y_steps=500, cutoff=3000.0)
    return tm


def MakeMatrixEQ(Mesh,eqn=equationsFEM.diffusion,max_nei=12,**kwargs):
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
            data[nnz]=eqn(i,j,[(nei_el,Mesh.elements[nei_el]) for nei_el in node2_els if Mesh.elements[nei_el].eltypes==2],max_nei=max_nei,kwargs=kwargs) #TODO fix indexing, bases, the 1!!!!!
            nnz += 1
    Mesh.matrix=csc_matrix((data,(rows,cols)),shape=(Mesh.numnodes,Mesh.numnodes)) #TODO fix indexing
    Mesh.rhs=rhs
    return None


def applyBCs(Mesh,dirichlet=[],neumann=[],b_funcs={}):
    """Go in and modify the matrix and rhs to comply with BCs"""
    # Just wrap Neumann and Dirichlet methods and add some error checking

    # Let's do some checking/warning about our inputs
    edges = np.sort(list(Mesh.physents.keys()))[0:-1]
    listed_edges=np.sort(dirichlet+neumann)
    if not all(edges==listed_edges):
        warn('Error with borders')
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
        warn('Error with borders')
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

    for edge in dirichlet:
        try:
            applyDirichlet(Mesh,edge_nodes[edge],b_funcs[edge])
        except KeyError:
            applyDirichlet(Mesh,edge_nodes[edge],lambda x:0) # We actually need to do something to implement a zero
            # maybe throw the error though?

    for edge in neumann:
        try:
            applyNeumann(Mesh,edge_nodes[edge],b_funcs[edge],normal=True)
        except KeyError: # If we have no condition we are just taking 0 neumann
            pass


def applyDirichlet(Mesh,edge_nodes,function):
    """Let's apply an essential boundary condition"""
    for node in edge_nodes:
        Mesh.matrix[node-1,node-1]=1.0
        Mesh.rhs[node-1]=function(Mesh.nodes[node].coords())
        for j in Mesh.nodes[node].neighbors.keys(): # Get the neighboring nodes
            if not j in edge_nodes: # Check if this neighboring node is on the edge
                Mesh.rhs[j-1]=Mesh.rhs[j-1]-Mesh.matrix[j-1,node-1]*Mesh.rhs[node-1]
            Mesh.matrix[node-1,j-1]=0.0
            Mesh.matrix[j-1,node-1]=0.0


def applyNeumann(Mesh,edge_nodes,function,normal=True): #TODO make non-normal stuff possible
    """Apply a natural boundary condition, must be normal"""
    for node in edge_nodes:
        ints=np.zeros((2,))
        i=0
        for j,els in Mesh.nodes[node].neighbors.items():
            print(node,j,els)
            if j in edge_nodes:
                for el in els:
                    if Mesh.elements[el].kind=='Line':
                        print(node,' connected to ',j,' with edge ',el)
                        self_ind=Mesh.elements[el].index(node)
                        nei_ind=Mesh.elements[el].index(j)
                        ints[i]=el#TODO
                        i+=1



def solveIt(Mesh,method='BiCGStab',precond=None,tolerance=1.0e-5):
    """Do some linear algebra"""
    if method=='CG':
            Mesh.sol,info=cg(Mesh.matrix,Mesh.rhs,tol=tolerance)
            if info>0:
                warn('Conjugate gradient did not converge. Attempting BiCGStab')
                Mesh.sol,info=bicgstab(Mesh.matrix,Mesh.rhs,tol=tolerance)
                if info>0:
                    raise ConvergenceError(method='CG and BiCGStab',iters=info)
    elif method=='BiCGStab':
        Mesh.sol,info=bicgstab(Mesh.matrix,Mesh.rhs,tol=tolerance)
        if info>0:
            raise ConvergenceError(method=method,iters=info)
    elif method=='direct':
        Mesh.sol=spsolve(Mesh.matrix,Mesh.rhs)
    else:
        raise TypeError('Unknown solution method')
    return Mesh.sol


def checkBases(me):
    """Just a quick check that the basis functions are zero and 1 where they should be"""
    badn=0
    for element in list(me.elements.values()):
        for i,pt in enumerate(element.xyvecs()):
            if abs(element.bases[i](pt)-1)>1e-15:
                print('Bad element ',element.id,'basis ',i)
                badn+=1
    if badn==0:
        print('Bases are good')


def plotSolution(Mesh,threeD=True,savefig=None,show=False,x_steps=20,y_steps=20,cutoff=5,savesol=False,figsize=(15,10)):
    mat_sol=sparse2mat(Mesh.coords,Mesh.sol,x_steps=x_steps,y_steps=y_steps,cutoff_dist=cutoff)
    if savesol:
        Mesh.matsol=mat_sol
    fig=plt.figure(figsize=figsize)
    if threeD:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(Mesh.coords[:,0],Mesh.coords[:,1],Z=Mesh.sol,cmap=cm.jet)
    else:
        ctr=plt.contourf(*mat_sol,levels=np.linspace(0.9*min(Mesh.sol),1.1*max(Mesh.sol),50))
        plt.colorbar(ctr)
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    return mat_sol


def sparse2mat(coords,data, x_steps=500, y_steps=500, cutoff_dist=2000.0):
    """Grid up some sparse, potentially concave data"""
    from scipy.spatial import cKDTree as KDTree
    from scipy.interpolate import griddata
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


if __name__=='__main__':
    main()



