#!/usr/bin/env python3

import numpy as np
from scipy.stats import unitary_group

def random_pop_list(n_sym, n_elec, HFdim):
    """Randomly populate n_sym buckets with n_elec objects, with max HFdim in each bucket"""
    N=0
    iteration=0
    pop_list=np.zeros(n_sym,dtype=int)
    while N<n_elec:
      isym=np.random.randint(0,n_sym)
      if pop_list[isym]<HFdim:
        pop_list[isym]+=1
        N+=1
      iteration+=1
    return pop_list


def gen_projector(pd,choice,sp_energy):
  """   
  Inputs:
    -sp_energy [ik1,ik2,tau,a]
    -choice has the following options
      -'empty': empty projector
      -'full': full projector
      -'CN': fill valence bands of BM model
      -'average': half-occupy of all active bands
      -'average central': fully fill (empty) remote valence (conduction) bands, and half-occupy central bands
      -'random': completely random (can have different populations in different sectors)
      -'BM': fill lowest-energy states of BM model up to filling factor 
       (may have missing electron if there are an odd number of electrons)
  Output:
    -P [ik1,ik2,s,tau*a,taup*b] (N1,N2,2,4*nact,4*nact)
  
  The total number of orbitals is 
    n_hilbert = 8*N1*N2*nact
  The total number electrons at filling is
    n_elec = N1*N2*(4*nact+filling)
  Symmetry sectors are (ik1,ik2,s), and the total number of them is
    n_sym = 2*N1*N2
    
  ! For the order of indices in the projector: 
  we use the convention where P[ik1,ik2,s,alpha,beta] = <c^\dagger_{alpha,s}(ik1,ik2) c_{beta,s}((ik2,ik2))> !
  """
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active'];filling=pd['filling']
  np.random.seed(pd["seed"])
  HFdim=4*nact #dimension of HF hamiltonian matrix (number of possible orbitals per conserved set of indicies)
  P=np.zeros((N1,N2,2,4*nact,4*nact),dtype=complex)
  n_elec = round(N1*N2*(4*nact+filling))
  n_sym = 2*N1*N2
  n_mband = 2*(2*nact)
  print("  chosen projector choice : "+choice)
  print("  filling factor : " + str(filling))
  print("  number of electrons in system : "+str(n_elec))
  print("  number of mBZ mtm and spins  : " + str(n_sym))
  print("  number of moire bands per mBZ mtm and spin : " + str(n_mband))
  
  if choice == 'empty':
    None
  
  elif choice == 'full':
    for ik1 in range(N1):
      for ik2 in range(N2):
        for s in range(2):
          P[ik1,ik2,s,...]=np.eye(HFdim)
  
  elif choice == 'CN':
    for tau in range(2):
      for a in range(nact):
        orb=a+tau*2*nact
        P[:,:,:,orb,orb]=1
            
  elif choice == 'average':
    for tau in range(2):
      for a in range(2*nact):
        orb=a+tau*2*nact
        P[:,:,:,orb,orb]=1/2
  
  elif choice == 'average central':
    Pex=np.zeros((N1,N2,2,2,2*nact,2,2*nact),dtype=complex)
    for tau in range(2):
      for a in range(nact-1):
        Pex[:,:,:,tau,a,tau,a]=1
      for a in range(2):
        Pex[:,:,:,tau,nact + a -1,tau,nact + a -1]=1/2
    P=np.reshape(Pex,(N1,N2,2,2*2*nact,2*2*nact))

  elif choice == 'random':
    #determine populations of each sector randomly
    n_list=random_pop_list(n_sym, n_elec,HFdim)
    for isym,norb in enumerate(n_list):
      s=isym%2
      ik2=(isym//2)%(N2)
      ik1=isym//(2*N2)
      P_diag=np.zeros((HFdim,HFdim),dtype=complex)
      P_diag[:norb,:norb]=np.eye(norb)
      U = unitary_group.rvs(HFdim)
      P[ik1,ik2,s,...]=np.einsum("ab,bc,dc->ad",U,P_diag,U.conj(),optimize=True)
  
  elif choice == 'BM':
    Pex=np.zeros((N1,N2,2,2,2*nact,2,2*nact),dtype=complex)
    for index in sp_energy.reshape((-1)).argsort()[:n_elec//2]:
      a=index%(2*nact)
      tau=(index//(2*nact))%(2) 
      ik2=(index//(4*nact))%(N2)
      ik1=(index//(N2*4*nact))
      Pex[ik1,ik2,:,tau,a,tau,a]=1
    P=np.reshape(Pex,(N1,N2,2,2*2*nact,2*2*nact))

  else:
    raise Exception("no valid choice selected for gen_projector")
    
  #check validity and filling of projector
  herm_check=np.max(np.abs(P-np.transpose(P.conj(),(0,1,2,4,3))))
  idem_check=np.max(np.abs(np.einsum("kKsab,kKsbc->kKsac",P,P)-P))
  if herm_check>1e-11: 
    print("!  P is NOT Hermitian!")
  if idem_check>1e-11:
    print("!  P is NOT idempotent!")    
  n_elec=np.real(np.einsum("kKsaa->",P))
  print("  P has filling factor %.3f" %((n_elec-4*nact*N1*N2)/(N1*N2)))
  return P
