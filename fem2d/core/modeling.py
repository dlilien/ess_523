#! /usr/bin/env python
#cython: embedsignature=True
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 David Lilien <dal22@uw.edu>
#
# Distributed under terms of the MIT license.


"""Define a number of different classes which collectively constitute a finite element solver

All these definitions are contained in the file modeling.py

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
>>> m=mo.makeIterate()
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
from scipy.sparse.linalg import bicgstab,cg,spsolve,gmres,spilu,LinearOperator
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_an
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from warnings import warn
from .equations import Equation
from .meshes import Mesh
from os.path import splitext
from scipy.sparse import csc_matrix,diags
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata
from collections import OrderedDict
import numpy as np
Axes3D # Avoid the warning

class Model:
    """A model, with associated mesh, and equations
    
    Parameters
    ----------
    Mesh : string or meshes.mesh
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
    eqn : list of :py:class:`equations.Equation`
        The equations to solve, should be attached using :py:meth:`add_equation`
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
        self.eqn=eqnlist()

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


    def add_equation(self,eqn,name=None,number=None,before_all=False,after_all=False):
        """Add the equation to be solved

        Parameters
        ----------
        eqn : `equations.Equation`
           Equation to solve
        name : str,optional
           Name which will overwrite the equation name

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
        if name is None:
            name=eqn.name
        self.eqn.setitem(name,eqn,number=number,before_all=before_all,after_all=after_all,diffEQ=True)
        return None


    def add_function(self,fnctn,name='Dummy',number=None,before_all=False,after_all=False):
        """Add a function to be called

        Parameters
        ----------
        eqn : `equations.Function`
           Function to call
        name : str,optional
           Name which will overwrite the equation name

        Raises
        ------
        TypeError
           If the equation is not of the proper type
        """

        self.eqn.setitem(name,fnctn,number=number,before_all=before_all,after_all=after_all,diffEQ=False)
        return None


    def add_BC(self,cond_type,target_edge,function,eqn_name=None):
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
        if eqn_name is not None:
            self.eqn[eqn_name].BCs[target_edge]=(cond_type,function)
        else:
            self.eqn[self.eqn.numbers[0]].BCs[target_edge]=(cond_type,function)


    def add_IC(self,function,eqn_name=None):
        """Associate an initial condition with an equation.

        Parameters
        ----------
        function : function
           The formula for the initial condition.
        eqn_name : str,optional
           The equation name with which to associate this IC. Default is None (the first).

        Raises
        ------
        ValueError if the model is steady state
        TypeError if the function is not callable on a coordinate input
        """

        if not self.time_dep:
            raise ValueError('Initial condition cannot apply to steady state model')
        try:
            function(self.mesh.nodes[1].coords())
        except TypeError:
            raise TypeError('Function must accept x,y coordinate as input')
        if eqn_name is not None:
            self.eqn[eqn_name].IC=function
        else:
            self.eqn[list(self.eqn.keys())[0]]=function


    def makeIterate(self,name=None):
        """Prepare to solve. Multi purpose, if num is given returns the nth equation (0 based index)

        Parameters
        ----------
        name : str,optional
           If not None, return an iterate of equation with that name
        
        Returns
        -------
        model : LinearModel or NonLinearModel or Multimodel
           Determined by linearity and number of equations
        """
        if name is not None:
            try:
                if Equation in type(self.eqn[name]).__bases__:
                    pass
                if self.eqn[name].lin:
                    return LinearModel(self,eqn_name=name)
                else:
                    return NonLinearModel(self,eqn_name=name)
            except AttributeError:
                return Dummy(self,f_name=name)

        elif len(self.eqn)==1:
            if list(self.eqn.values())[0].lin:
                return LinearModel(self)
            else:
                return NonLinearModel(self)
        else:
            return MultiModel(self)


class Dummy:
    """Mimic the iterate method so we can use non-diffeq stuff"""
    def __init__(self,model,f_name):
        self.model=model
        self.f_name=f_name
        self.f=self.model.eqn[self.f_name]
        self.eqn_name=f_name

    def iterate(self,**kwargs):
        self.f(self.model.mesh,self.model,kwargs['sol'])


class eqnlist:
    
    def __init__(self):
        self.mainDict={}
        self.keyPairs=[]
        self.numbers=[]
        self.f_numbers=[]
        self.de_numbers=[]

    def __getitem__(self,key):
        if type(key)==int:
            try:
                return self.mainDict[[keyname for keynum,keyname in self.keyPairs if keynum==key][0]]
            except IndexError:
                raise KeyError('Key number does not exist')
        elif type(key)==str:
            return self.mainDict[key]

    def __len__(self):
        return len(self.numbers)

    def setitem(self,key,value,number=None,before_all=False,after_all=False,diffEQ=True):
        """Number overrides before/after all

        If no ordering info is supplied, the next largest number counting up from 0 is used.

        Last entries in those categories claim precedence.
        before_all entries are numbered with negative indices descending from -1.
        after_all ascends from 1000.
        """
        if not type(key)==str:
            raise TypeError('Key',key,'must be a string')
        if number is not None:
            if number in self.numbers:
                raise AttributeError('Cannot overwrite existing equation number')
        elif before_all:
            number=-1
            while number in self.numbers:
                number-=1
        elif after_all:
            number=1000
            while number in self.numbers:
                number+=1
        else:
            if len(self.numbers)==0:
                number = 0
            else:
                number=max(key for key in self.numbers if key<1000)+1
        self.mainDict[key]=value
        self.keyPairs.append((number,key))
        self.numbers.append(number)
        if diffEQ:
            self.de_numbers.append(number)
        else:
            self.f_numbers.append(number)

    def items(self):
        return self.mainDict.items()

    def values(self):
        return self.mainDict.values()


class ModelIterate:
    """This object makes matrix, forms a solution, etc

       Parameters
       ----------
       model : modeling.Model
           The model, with equations and boundary conditions
       eqn : :py:class:`equations.Equation`,optional
           The equation to solve, if it differs from that tied to the model
           e.g. in a time dependent model

       Keyword Arguments
       -----------------
       dofs : int,optional
           Number of degrees of freedom. Default to that associated with the equation
       eqn_name : str,optional
           The name of the equation to solve if there are multiple, if eqn is not given.
       """



    def __init__(self,model,*eqn,**kwargs):
        self.parent=model
        self.mesh=self.parent.mesh
        if eqn:
            self.eqn=eqn[0]
        else:
            if 'eqn_name' in kwargs and (kwargs['eqn_name'] is not None):
                self.eqn_name=kwargs['eqn_name']
            else:
                self.eqn_name=[name for num,name in self.parent.eqn.keyPairs if num==self.parent.eqn.numbers[0]][0]
            self.eqn=self.parent.eqn[self.eqn_name]


    def MakeMatrixEQ(self,**kwargs):
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

        if self.eqn.dofs==1:
            # The easy version, scalar variable to solve for

            # Empty vectors to make the sparse matrix
            rows=np.zeros(self.eqn.max_nei*self.mesh.numnodes,dtype=np.int16)
            cols=np.zeros(self.eqn.max_nei*self.mesh.numnodes,dtype=np.int16)
            data=np.zeros(self.eqn.max_nei*self.mesh.numnodes)

            # zero vector for the rhs
            rhs=np.zeros(self.mesh.numnodes)

            # count the number of non-zeros
            nnz=0

            for i,node1 in self.mesh.nodes.items():

                # Do the diagonal element
                rows[nnz]=i-1 
                cols[nnz]=i-1
                data[nnz],rhs[i-1]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],rhs=True,**kwargs)
                nnz += 1

                for j,node2_els in node1.neighbors.items():
                    # Do the off diagonals, do not assume symmetry
                    rows[nnz]=i-1
                    cols[nnz]=j-1
                    data[nnz]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],**kwargs)
                    nnz += 1

            # store what we have done
            self.matrix=csc_matrix((data,(rows,cols)),shape=(self.mesh.numnodes,self.mesh.numnodes))
            self.rhs=rhs
            return None

        elif self.eqn.dofs==2:
            # Set things up so we can do velocity

            # Empty vectors to accept the sparse info, make them large for cross terms
            malloc=self.eqn.max_nei*self.mesh.numnodes*self.eqn.dofs**2
            m=self.mesh.numnodes*self.eqn.dofs

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
                data[nnz],data[nnz+1],data[nnz+2],data[nnz+3],rhs[i-1],rhs[i]=self.eqn(i,i,[(elm[0],self.mesh.elements[elm[0]]) for elm in node1.ass_elms if self.mesh.elements[elm[0]].eltypes==2],rhs=True,**kwargs)
                
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
                    data[nnz],data[nnz+1],data[nnz+2],data[nnz+3]=self.eqn(i,j,[(nei_el,self.mesh.elements[nei_el]) for nei_el in node2_els if self.mesh.elements[nei_el].eltypes==2],**kwargs)

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
        BCs=self.eqn.BCs
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
        if self.eqn.dofs==1:
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

        elif self.eqn.dofs==2:
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
        if self.eqn.dofs==1:
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
        elif self.eqn.dofs==2:
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


    def solveIt(self):
        """Solve the matrix equation

        Returns
        -------
        self.sol : array
            The matrix solution
        """


        if not self.eqn.method=='direct':
            if self.eqn.precond=='LU':
                p=spilu(self.matrix, drop_tol=1.0e-5)
                M_x=lambda x: p.solve(x)
                M=LinearOperator((self.mesh.numnodes*self.eqn.dofs,self.mesh.numnodes*self.eqn.dofs),M_x)
            elif self.eqn.precond is not None:
                M=self.eqn.precond
        if self.eqn.method=='CG':
            if self.eqn.precond is not None:
                self.sol,info=cg(self.matrix,self.rhs,tol=self.eqn.lin_tolerance,M=M)
            else:
                self.sol,info=cg(self.matrix,self.rhs,tol=self.eqn.lin_tolerance)
            if info>0:
                warn('Conjugate gradient did not converge. Attempting BiCGStab')
                if self.eqn.precond is not None:
                    self.sol,info=bicgstab(self.matrix,self.rhs,tol=self.eqn.lin_tolerance,M=M)
                else:
                    self.sol,info=bicgstab(self.matrix,self.rhs,tol=self.eqn.lin_tolerance)
                if info>0:
                    raise ConvergenceError(method='CG and BiCGStab',iters=info)
        elif self.eqn.method=='BiCGStab':
            if self.eqn.precond is not None:
                self.sol,info=bicgstab(self.matrix,self.rhs,tol=self.eqn.lin_tolerance,M=M)
            else:
                self.sol,info=bicgstab(self.matrix,self.rhs,tol=self.eqn.lin_tolerance)
            if info>0:
                raise ConvergenceError(method=self.eqn.method,iters=info)
        elif self.eqn.method=='direct':
            self.sol=spsolve(self.matrix,self.rhs)
        elif self.eqn.method=='GMRES':
            if self.eqn.precond is not None:
                self.sol,info=gmres(self.matrix,self.rhs,tol=self.eqn.lin_tolerance,M=M)
            else:
                self.sol,info=gmres(self.matrix,self.rhs,tol=self.eqn.lin_tolerance)
            if info>0:
                raise ConvergenceError(method=self.eqn.method,iters=info)
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


        if self.eqn.dofs==1:
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

        elif self.eqn.dofs==2:

            # Do a quick check before we do the slow steps
            if savefig is not None:
                if not len(savefig)==self.eqn.dofs:
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
        if self.eqn.dofs==1:
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

        elif self.eqn.dofs==2:
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
    def iterate(self,time=None,**kwargs):
        self.MakeMatrixEQ(max_nei=self.eqn.max_nei,**kwargs)
        self.applyBCs(time=time)
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = [name for num,name in self.parent.eqn.keyPairs if num==self.parent.eqn.numbers[0]][0]
        if time is not None:
            if 'BDF1' in kwargs:
                self.matrix=kwargs['timestep']*self.matrix+diags(np.ones(self.mesh.numnodes),0)
                self.rhs=kwargs['timestep']*self.rhs+np.array(kwargs['td_sol'][-1][name])
            elif 'BDF2' in kwargs:
                self.matrix=self.matrix-diags()
            else:
                raise ValueError('Cannot do that timestepping stategy')
        sol=self.solveIt()
        return {self.eqn_name:sol}


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

    def __init__(self,model,eqn_name=None):
        self.model=model
        if eqn_name is None:
            self.eqn_name=list(model.eqn.values)[0]
        else:
            self.eqn_name=eqn_name
        self.eqn=model.eqn[self.eqn_name]


    def iterate(self,gradient,time=None,abort_not_converged=False,**kwargs):
        """
        The method for performing the solution to the nonlinear model iterate

        Parameters
        ----------
        gradient : function
           This gets called at every iteration in order to update parameters used in the equation being solved
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
        if self.eqn.guess is not None:
            old=self.eqn.guess
        else:
            old=np.zeros(self.model.mesh.numnodes*self.eqn.dofs)

        try:
            for i in range(self.eqn.nl_maxiter):
                # Loop until we have converged to within the desired tolerance
                print( self.eqn_name,'Nonlinear iterate {:d}:    '.format(i) , end=' ')
                kwargs['gradient']=gradient(self,old)

                self.linMod=LinearModel(self.model,dofs=self.eqn.dofs,eqn_name=self.eqn_name)
                new=self.linMod.iterate(time=time,**kwargs)[self.eqn_name]
                if np.linalg.norm(new)>1.0e-15:
                    relchange=np.linalg.norm(new-old)/np.sqrt(float(self.model.mesh.numnodes))/np.linalg.norm(new)
                else:
                    relchange=np.linalg.norm(new-old)/np.sqrt(float(self.model.mesh.numnodes))
                print('Relative Change: {:f}'.format(relchange))

                #Check if we converged
                if relchange<self.eqn.nl_tolerance and i != 0:
                    break
                old=self.eqn.relaxation*new[:]+(1.0-self.eqn.relaxation)*old[:]
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
        return {self.eqn_name:new}


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
            if type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==float or type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==np.float64:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    sol[i]=self.model.mesh.elements[element].phys_vars[target]
            elif type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==list:
                 for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    sol[i]=self.model.mesh.elements[element].phys_vars[target][0][1]
            else:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    try:
                        sol[i]=self.model.mesh.elements[element].phys_vars[target](element.cent)
                    except:
                        raise RuntimeError('Problems with parsing function for plotting')


        if self.eqn.dofs==1 or target is not None:
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

        elif self.eqn.dofs==2:

            # Do a quick check before we do the slow steps
            if savefig is not None:
                if not vel:
                    if not len(savefig)==self.eqn.dofs:
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


            elif self.eqn.dofs==2:
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
            if type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==float or type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==np.float64:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    data[i]=self.model.mesh.elements[element].phys_vars[target]
            elif type(self.model.mesh.elements[tri_els[0]].phys_vars[target])==list:
                 for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    data[i]=self.model.mesh.elements[element].phys_vars[target][0][1]
            else:
                for i,element in enumerate(tri_els):
                    coords[i,:]=self.model.mesh.elements[element].cent
                    try:
                        data[i]=self.model.mesh.elements[element].phys_vars[target](element.cent)
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


class MultiModel:
    """A class for solving models with multiple equations associated. Leave the other guys alone for now.

    """
    kind='NonLinear'

    def __init__(self,model):
        self.model=model

    def iterate(self,gradient,ss_maxiter=50,time=None,abort_not_converged=False,**kwargs):
        """
        The method for performing the solution to the nonlinear model iterate

        Parameters
        ----------
        gradient : dictionary of functions
           This gets called at every iteration in order to update parameters used in the equation being solved.
        ss_maxiter : int
           Maximum number of steady-state iterations. Default 50.
        time : float,optional
           The time in a time dependent model. None for steady state. Defaults to None.
        abort_not_converged : bool,optional
           If true, raise a :py:exc:`ConvergenceError` if non-linear iterations do not converge. Otherwise call it good enough. Defaults to False.
        
        Any Any keyword arguments are passed down to the equation we are solving, for example time dependent terms like sources or conductivity can be specified.

        Returns
        -------
        solution : list of arrays
           The solutions to the nonlinear Equations

        """

        # Make a container to hold the solutions
        # We are going to have slots for every equation, even if some are linear
        self.sol={name:np.zeros(self.model.mesh.numnodes*eqn.dofs) for name,eqn in self.model.eqn.items() if [eqnum for eqnum,eqname in self.model.eqn.keyPairs if eqname==name][0] in self.model.eqn.de_numbers}
        relchange={name:1.0 for name in self.sol}
        ss_relchange=relchange.copy()
        self.models=OrderedDict([([name for eqnnum,name in self.model.eqn.keyPairs if eqnnum==num][0],self.model.makeIterate([name for eqnnum,name in self.model.eqn.keyPairs if eqnnum==num][0])) for num in sorted(self.model.eqn.numbers)])

        for k in range(0,ss_maxiter):
            for name,model in self.models.items():
                if [eqnum for eqnum,eqname in self.model.eqn.keyPairs if eqname==name][0] in self.model.eqn.de_numbers:
                    if self.model.eqn[name].lin:
                        print('SS iteration',str(k+1),'for linear',name)
                        new=model.iterate(time=time,**kwargs)
                    else:
                        print('SS iteration',str(k+1),'for nonlinear',name)
                        new=model.iterate(gradient[name],time=time,**kwargs)
                        model.eqn.guess=new[name].copy()
                    if k != 0:
                        ss_relchange[name]=np.linalg.norm(self.sol[name]-new[name])/np.sqrt(float(self.model.mesh.numnodes))/np.linalg.norm(new[name])
                    self.sol[name]=new[name].copy()
                    print(name,'Steady State change',ss_relchange[name])
                else:
                    model.iterate(sol=self.sol.copy())
            if np.all([ss_relchange[name]<eqn.ss_tolerance for name,eqn in self.model.eqn.items() if [eqnum for eqnum,eqname in self.model.eqn.keyPairs if eqname==name][0] in self.model.eqn.de_numbers]) and k !=0:
                print('Steady State convergence reached at iteration',str(k+1))
                break
            print('Steady State iteration',str(k+1),'completed')
        else:
            if abort_not_converged:
                raise ConvergenceError('Steady state solver failed within iteration count')
            else:
                print('Steady state solver did not converge within desired tolerance, returning')
        return self.sol


class TimeDependentModel:
    """A time dependent model"""


    def __init__(self,model,timestep,n_steps,method='BDF1'):
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
           A of arrays of the node-wise solution for each equation at each timestep. The first entry is the initial condition. I.e. if you query sol[i][j][k] you get the solution at the ith timestep, jth equation, kth node.
        """
        sol=[{name:np.array([eqn.IC(pt) for pt in self.model.mesh.coords]) for name,eqn in self.model.eqn.items() if [eqnum for eqnum,eqname in self.model.eqn.keyPairs if eqname==name][0] in self.model.eqn.de_numbers}]
        if self.method=='BDF2':
            time=self.timestep
            sol.append(self.model.makeIterate().iterate(time=time,BDF1=True,timestep=self.timestep,td_soln=sol))
            for i in range(2,self.n_steps):
                time=i*self.timestep
                sol.append(self.model.makeIterate().iterate(time=time,timestep=self.timestep,td_sol=sol))
        elif self.method=='BDF1':
            for i in range(1,(self.n_steps+1)):
                time=i*self.timestep
                print('Timestep {:d}, real time {:f}'.format(i,time))
                sol.append(self.model.makeIterate().iterate(time=time,BDF1=True,timestep=self.timestep,td_sol=sol))
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
        #ax.set_zlim3d([0.9*min(self.sol[0]),1.1*max(self.sol[0])])
        class nextPlot:
            def __init__(self,outer):
                self.iterate=0
                self.outer=outer
            def __call__(self,num):
                self.iterate+=1
                ax.clear()
                ax.set_xlim3d([min(self.outer.model.mesh.coords[:,0]),max(self.outer.model.mesh.coords[:,0])])
                ax.set_ylim3d([min(self.outer.model.mesh.coords[:,1]),max(self.outer.model.mesh.coords[:,1])])
                #ax.set_zlim3d([0.9*min(self.outer.sol[0]),1.1*max(self.outer.sol[0])])
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

