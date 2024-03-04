
********************************************************************************

                   Definition of the little Dirac operator

********************************************************************************

The little Dirac operator was introduced in

  M. Luescher: "Local coherence and deflation of the low quark modes
                in lattice QCD", JHEP 0707 (2007) 081

Here the data structures used to represent the operator are described.


Definition
----------

The deflation subspace is spanned by the fields (*b).s[1],..,(*b).s[Ns] on the
blocks b of the DFL_BLOCKS block grid. When the subspace is created, the basis
fields are orthonormalized on each block. The restriction of the Wilson-Dirac
operator Dw+i*mu*gamma_5 to the deflation subspace is referred to as the
little Dirac operator. It is completely specified by the matrix elements

  A_{n,k;m,l}=(v_{n,k},(Dw+i*mu*gamma_5*I_eo)*v_{m,l})

where v_{n,1},v_{n,2},..,v_{n,Ns} are the basis vectors on the block with
index n. The operator I_eo is 1 on the even sites of the lattice and
0 or 1 on the odd sites depending on whether the even-odd flag is set
or not (see flags/lattice.c).


Matrix arrays
-------------

The DFL_BLOCKS block grid consists of the sublattices of the even and odd
blocks (see dfl/dfl_geometry.c). In each local lattice, there are nb blocks,
half of which are even and half odd. The number nbb of blocks on the exterior
boundary of the local lattice also divides into equal numbers of even and odd
blocks.

The matrix A_{n,k;m,l} decomposes into the block-diagonal part and the
block-hopping part. Each of these parts may be stored in the form of
one-dimensional arrays of complex Ns x Ns matrices according to

  Ablk[n][Ns*k+l] = (v_{n,k},(Dw+i*mu*gamma_5*I_eo)*v_{n,l}),

  Ahop[8*n+ifc][Ns*k+l] = (v_{n,k},Dw*v_{inn[n][ifc],l}),

where the index n=0,..,nb-1 runs through all blocks, ifc=0,..,7 labels the 8
coordinate directions -0,+0,..,-3,+3 and inn[n][ifc] is the index of the block
in direction ifc of the block with index n (see dfl/dfl_geometry.c).

More explicitly, the hopping terms are given by

 -0.5*(v_{n,k},U*(1-gamma_mu)*v_{inn[n][ifc],l})  if ifc=2*mu+1,

 -0.5*(v_{n,k},U^dag*(1+gamma_mu)*v_{inn[n][ifc],l})  if ifc=2*mu,

where U stands for the link variables in direction mu=0,..,3 required to make
the scalar products gauge invariant.


Even-odd preconditioning
------------------------

The little operator acts on complex fields phi_{n,k} according to

  (A*phi)_{n,k}=sum_{m=0}^{nb-1} sum_{l=1}^Ns A_{n,k;m,l}*phi_{m,l}.

As in the case of the block-grid geometry arrays, the block index n is such
that the first nb/2 blocks are the even ones. The little operator accordingly
has a 2x2 block form with diagonal operators Aee,Aoo and off-diagonal ones
Aeo,Aoe.

The even-odd preconditioned little operator Ahat acts on fields supported on
the even blocks. It is related to the little operator A through

  Ahat=1-Aee^(-1)*Aeo*Aoo^(-1)*Aoe.

Ablk^(-1) and the products Ablk^(-1)*Ahop may be precomputed and are stored in
the static memory of the relevant module.


Data structures
---------------

The single- and double-precision arrays representing the little Dirac operator
are collected in the structures Aw_t and Aw_dble_t (see include/little.h). The
elements of these structures are

 Ns,nb
 Ablk[nb][Ns*Ns]
 Ahop[8*nb][Ns*Ns]

In the case of the even-odd preconditioned operator, the same structure
is used, but the stored matrices are now Ablk^(-1) and Ablk^(-1)*Ahop.


Little-little Dirac operator
----------------------------

The deflation subspace is constructed by projecting Ns global spinor fields to
the blocks of the DFL_BLOCKS grid. These global fields are linear combinations
of the basis fields v_{n,k} and span a subspace of dimension Ns within the
deflation subspace (which has dimension nb*Ns).

The even-odd preconditioned little Dirac operator may be deflated using the
restriction of the global modes to the even blocks as the deflation modes.
These fields (which are also contained in the deflation subspace) are referred
to as the little modes, and the restriction of the even-odd preconditioned
little Dirac operator to the space spanned by them as the little-little Dirac
operator. Its action is completely specified by its matrix elements in the
space of the little modes, i.e. by a complex Ns x Ns matrix (the program
set_ltl_modes() orthonormalizes the little modes before the little-little
Dirac operator is calculated).

The single-precision little modes md_k (k=1,..,Ns) and Awhat*md_k are stored
in the first and second half of the first Ns fields returned by vflds(). The
double-precision fields are stored in the same way in the Ns fields returned
by vdflds().


Boundary conditions
-------------------

In the case of boundary conditions of type 0,1 and 2, the hopping terms Ahop
that go across the boundaries of the lattice at global time 0 and NPROC0*L0-1
are equal to zero.

The programs in this directory compute the little Dirac operator always in the
same way and thus effectively as if periodic boundary conditions were imposed
in the time direction. Since the quark fields vanish at time 0 when boundary
conditions of type 0,1 or 2 are chosen, the programs however obtain the
correct result (i.e. vanishing hopping terms across the lattice boundaries)
also in these cases.
