#!/usr/bin/env python3

import numpy as np
import time

def gen_H_SP(pd,sp_energy):
  """
  Process one-body terms and generate one-body Hamiltonian
  Inputs
    -sp_energy [ik1,ik2,tau,a]
  Outputs
    -H_SP [ik1,ik2,s,tau*a,taup*b] 
  """  
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active']
  kron=np.eye(2)
  kronband=np.eye(2*nact)
  idspin=np.ones(2)
  
  H_SP=np.einsum("ab,kKta->kKtab",kronband,sp_energy)
  H_SP=np.einsum("s,tT,kKtab->kKstaTb",idspin,kron,H_SP,optimize=True)
  H_SP=np.reshape(H_SP,(N1,N2,2,4*nact,4*nact))
  return H_SP

def gen_M_tVE(pd,form,intFT):
  """
  Process terms for interactions
  Inputs
    -form [ik1,ik2,iq1,iq2,G1+NG1,G2+NG2,tau,a,b]
    -intFT [iq1,iq2,G1+NG1,G2+NG2]
  Outputs
    -form [ik1,ik2,iq1,iq2,G1,G2,tau,a,b]
    -M [ik1,ik2,iq1,iq2,G1,G2,tau,a,b]
    -tVE [ik1*ik2*a*d,iq1*iq2*b*c,tau*taup]
  """
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active'];NG1=pd['NG1'];NG2=pd['NG2']  
  #check if form is complex
  if np.max(np.abs(np.imag(form)))>1e-9:
    iscomplex = True
  else:
    iscomplex = False
  
  #generate M
  M=np.einsum("qQgG,kKqQgGtab->kKqQgGtab",intFT,form.conj(),optimize=True)
  
  if pd['exchange']==False:
    VE=np.zeros((N1*N2*4*nact**2,N1*N2*4*nact**2,4),dtype=complex)
    print("!  Exchange part of interaction not included!")    
    return form,M,VE
  
  
  #generate VE
  #index order is pre-empted for memory-efficient reshaping
  #[ik1,ik2,a,d,iq1,iq2,b,c,tau,taup]
  st=time.time()
  if iscomplex==True:
    VE=np.empty((N1,N2,2*nact,2*nact,N1,N2,2*nact,2*nact,2,2),dtype=complex)
  if iscomplex==False:
    VE=np.empty((N1,N2,2*nact,2*nact,N1,N2,2*nact,2*nact,2,2),dtype=float)  
  for ik1 in range(N1):
    VE[ik1,...]=np.einsum("KqQgGtab,KqQgGTdc->KadqQbctT",
                                form[ik1,...],
                                M[ik1,...],optimize=True).copy()
  print("  VE generated : %.4f seconds" % (time.time() - st))
  
  #roll to get tVE
  st=time.time()
  for ik1 in range(N1):
    VE[ik1,:,...]=np.roll(
      VE[ik1,:,...],ik1,axis=3)
  for ik2 in range(N2):
    VE[:,ik2,...]=np.roll(
      VE[:,ik2,...],ik2,axis=4)    
  #reshape tVE 
  #[ik1*ik2*a*d,iq1*iq2*b*c,tau*taup]
  VE.shape=(N1*N2*4*nact**2,N1*N2*4*nact**2,4)
  
  print("  tVE generated : %.4f seconds" % (time.time() - st))
  
  return form,M,VE
  
def calc_E(pd,P,P_ref,sp_energy,HFham_D,HFham_E):
  """
  Inputs:
    -P,P_ref [ik1,ik2,s,tau*a,taup*b] 
    -sp_energy [ik1,ik2,tau,a]
    -HFham_D/E [ik1,ik2,s,tau*a,taup*b] 
  Output:
    -[E_tot,E_kin,E_D,E_E]
  """
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active']
  Psplit=np.reshape(P,(N1,N2,2,2,2*nact,2,2*nact)).copy() #[ik1,ik2,s,tau,a,taup,b]
  E_kin=np.einsum("kKta,kKstata->",sp_energy,Psplit,optimize=True) 
  E_D=0.5*np.einsum("kpsAB,kpsAB",HFham_D,P-P_ref,optimize=True) #Hartree energy
  E_E=0.5*np.einsum("kpsAB,kpsAB",HFham_E,P-P_ref,optimize=True) #Fock energy
  if np.abs(np.imag(E_kin+E_D+E_E))>1e-9:
    print("!  Imaginary HF energy detected!")
  return np.real([E_kin+E_D+E_E, E_kin, E_D, E_E])
  
def calc_fock_matrix(pd,P,form,M,tVE,timeit=False):
  """
  Inputs:
    -P (this should have P_ref subtracted according to interaction scheme)
     [ik1,ik2,s,tau*a,taup*b] 
     (N1,N2,2,4*nact,4*nact)
    -form
     [ik1,ik2,iq1,iq2,G1,G2,tau,a,b]
     (N1,N2,N1,N2,2*NG,2*NG,2,2*nact,2*nact)
    -M  (this is form.conj()*Vint)
     [ik1,ik2,iq1,iq2,G1,G2,tau,a,b]
     (N1,N2,N1,N2,2*NG,2*NG,2,2*nact,2*nact)     
    -tVE (rolled and reshaped version of VE)
     [ik1*ik2*a*d,iq1*iq2*b*c,tau*taup]
     (N1*N2*4*nact**2,N1*N2*4*nact**2,4)
     which is a reshape of the more natural:
     [ik1,ik2,tq1,tq2,tau,taup,a,b,c,d]
     (N1,N2,N1,N2,2,2,2*nact,2*nact,2*nact,2*nact)
  Output:
    -HFham_D and HFham_E
     [ik1,ik2,s,tau*a,taup*b] 
     (N1,N2,2,4*nact,4*nact)
  """
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active']
  kron=np.eye(2)
  #prepare Pr[ik1,ik2,s,tau,a,taup,b], which is rolled and reshaped version of P
  Pr=np.reshape(P,(N1,N2,2,2,2*nact,2,2*nact)).copy()

  #Hartree contribution
  st=time.time()
  PrH=np.sum(Pr,axis=2) #[ik1,ik2,tau,a,taup,b]
  PrH=np.diagonal(PrH,axis1=2,axis2=4) #[ik1,ik2,a,b,tau]
  MPrH=np.einsum("kKgGtab,kKbat->gG",M[:,:,0,0,...],PrH,optimize=True) #[G1,G2]
  HFham_D=np.einsum("kKgGtab,gG->kKabt",form[:,:,0,0,...],MPrH,optimize=True) #[ik1,ik2,a,b,tau]
  HFham_D=np.einsum("tT,s,kKabt->kKstaTb",kron,np.ones(2),HFham_D,optimize=True)   
  HFham_D=np.reshape(HFham_D,(N1,N2,2,4*nact,4*nact))
  timeD=time.time()-st
  
  # Fock contribution (this includes the minus sign prefactor expected for the Fock contraction)
  st=time.time()
  Pr=np.transpose(Pr,(2,0,1,6,4,5,3))
  Pr=np.reshape(Pr,(2,N1*N2*4*nact**2,4))
  HFham_E=-np.einsum("abc,sbc->asc",tVE,Pr,optimize=True)
  HFham_E=np.reshape(HFham_E,(N1,N2,2*nact,2*nact,2,2,2))  #[ik1,ik2,a,b,s,tau,taup]     
  HFham_E=np.transpose(HFham_E,axes=(0,1,4,5,2,6,3)) #[ik1,ik2,s,tau,a,taup,b]
  HFham_E=np.reshape(HFham_E,(N1,N2,2,4*nact,4*nact))
  timeE=time.time()-st
  
  # check hermiticity
  if np.max(np.abs(HFham_D+HFham_E-np.transpose((HFham_D+HFham_E).conj(),(0,1,2,4,3))))>1e-9:
    print("!  Fock matrix is not Hermitian!") 
    
  if timeit==True:
    return [timeD,timeE], HFham_D + HFham_E
  return HFham_D,HFham_E
 
def aufbau(pd,HFham,timeit=False):
  """
  Input:
    -HFham [ik1,ik2,s,au*a,taup*b] 
  Output: 
    -P [ik1,ik2,s,tau*a,taup*b]
    -HFeigs [ik1*ik2*s*HFstate]
    -fill_index [filled ik1*ik2*s*HFstate]
    -evecs [ik1*ik2*s,tau*a,HFstate]
    
  ! For the order of indices in the projector: 
  we use the convention where P[ik1,ik2,s,alpha,beta] = <c^\dagger_{alpha,s}(ik1,ik2) c_{beta,s}(ik2,ik2)> !
  """
  N1=pd['N1'];N2=pd['N2'];N=N1*N2;nact=pd['n_active'];filling=pd['filling']
  HFdim=4*nact
  n_sym=N1*N2*2
  n_elec = round(N1*N2*(4*nact+filling))
  HFeigs=np.zeros((n_sym,HFdim)) #[ik1*ik2*s,HFstate]
  evecs=np.zeros((n_sym,HFdim,HFdim),dtype=complex) #[ik1*ik2*s,tau*a,HFstate]
  
  #diagonalize Hamiltonian
  st=time.time()
  for ik1 in range(N1):
    for ik2 in range(N2):
      for s in range(2):
        HFeigs[ik1*2*N2+ik2*2+s,...],evecs[ik1*2*N2+ik2*2+s,...]=np.linalg.eigh(HFham[ik1,ik2,s,...])
  timediag=time.time()-st

  #construct projector 
  st=time.time()          
    
  #determine filled states based on spin configuration 
  fill_index = np.reshape(HFeigs,(-1)).argsort()[:n_elec]  

  fill_mask = np.zeros((n_sym*HFdim))
  fill_mask[fill_index] = 1
  fill_mask = np.reshape(fill_mask, (n_sym, HFdim))
  P = np.einsum("san, sbn, sn -> sab" , evecs.conj(), evecs, fill_mask, optimize = True)
  P=np.reshape(P,(N1,N2,2,HFdim,HFdim))
  timeproj=time.time()-st
  
  if timeit==True:
    return [timediag,timeproj]
  return P, np.reshape(HFeigs,(-1)), fill_index, evecs
