#!/usr/bin/env python3
"""
This code does not support interaction schemes where the frozen valence (conduction) bands are not fully occupied (empty) in the reference density matrix.

Spinless time-reversal is used to construct the BM eigenvalues/eigenvectors in valley K' from those in valley K.
The code must be generalized to handle situations where time-reversal is broken explicitly (by e.g. a parallel magnetic field).
"""

import numpy as np
import time


from constants import *


# =============================================================================
# Basic generation functions
# =============================================================================

def gen_RLVs(pd):
  """
  Generates twist/strain transform, and moire RLVs.
  
  Inputs:
    
  Outputs:    
    M1[x or y,x or y] (2,2)
    M2[x or y,x or y] (2,2)
    b1[x or y] (2)
    b2[x or y] (2)
    Etens1[x/y,x/y] (2,2)
    Etens2[x/y,x/y] (2,2)
  """
  def rot(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

  theta=np.pi/180*pd['theta'];varphi=np.pi/180*pd['varphi'];strain=pd['strain']
  
  Etens1 = rot(-varphi)@np.diag([-strain/2, Poisson*strain/2])@rot(varphi)
  Etens2 = - Etens1

  M1 = rot(theta/2) + Etens1
  M2 = rot(-theta/2) + Etens2

  #MG already has kD scaled out. b's therefore have k_theta scaled out
  b1=np.dot(np.linalg.inv(M1)-np.linalg.inv(M2),bG2-bG1)/(2*np.sin(theta/2))
  b2=np.dot(np.linalg.inv(M1)-np.linalg.inv(M2),bG1)/(2*np.sin(theta/2))

  return M1,M2,b1,b2,Etens1,Etens2

def gen_moire_hamiltonian(pd,ik):
  """
  Generates BM hamiltonian for valley K.
  
  Inputs:
    ik[ik1,ik2] (N1,N2)
    
  Outputs:
    H_BM[basis1,basis2] (4*Ng1*Ng2,4*Ng1*Ng2)
      ik1 takes values 0,...,N1-1.
      g1 takes values -Ng1,...,Ng1-1.
      The origin is the X-point---the centre of the hexagon with K1 and K2 on 
      its left corners.
      The order of basis elements in H_BM is nested as:
        {species,g2,g1} (so g1 is the outermost block, and species is the fastest index)
        where species is 1A,1B,2A,2B.
  """
  N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2'];wAA=pd['wAA']
  wAB=pd['wAB'];theta=np.pi/180*pd['theta']
  ktheta=2*kD*np.sin(theta/2)
  M1,M2,b1,b2,Etens1,Etens2=gen_RLVs(pd)
  b1s=b1/N1;b2s=b2/N2; #discretized momentum spacing in the 1BZ
  ik1 = ik[0]; ik2 = ik[1]
  k=(ik1)*b1s+(ik2)*b2s; #momentum in 1BZ (k_theta is always scaled out of all momenta)  
  T1=np.array([[wAA, wAB],[wAB, wAA]])
  T2=np.array([[wAA, wAB * omega],[wAB * omegac, wAA]])
  T3=np.array([[wAA, wAB * omegac],[wAB * omega, wAA]]) 

  #generate Hamiltonian
  Ng=4*Ng1*Ng2 #number of RLVs (g's); because say g1 goes from -Ng1 to Ng1-1
  Hkin=np.zeros((4*Ng,4*Ng),dtype=complex) #factor of four is sublat/layer
  Hhop=np.zeros((4*Ng,4*Ng),dtype=complex)    
  for g1 in np.arange(-Ng1,Ng1):
    for g2 in np.arange(-Ng2,Ng2):
      indexg = 4*((g1+Ng1)*(2*Ng2)+(g2+Ng2))   
      #generate kinetic part -- 1BZ mtm is measured from Gamma_M point
      d1=-1/3*b1+1/3*b2 #position of K1 relative to X point
      A1=Beta/2/CCa*np.array([Etens1[0,0]-Etens1[1,1],-2*Etens1[0,1]])
      q1=ktheta*(k+g1*b1+g2*b2-d1)-A1
      q1 = np.dot(M1, q1)
      t1 = vhbar*(q1[0]*sx + q1[1]*syc)

      d2=-2/3*b1-1/3*b2 #position of K2 relative to X point
      A2=Beta/2/CCa*np.array([Etens2[0,0]-Etens2[1,1],-2*Etens2[0,1]])
      q2=ktheta*(k+g1*b1+g2*b2-d2)-A2
      q2 = np.dot(M2, q2)
      t2 = vhbar*(q2[0]*sx + q2[1]*syc)
      #species order is 1A,1B,2A,2B
      Hkin[indexg:indexg+2,indexg:indexg+2]=t1
      Hkin[indexg+2:indexg+4,indexg+2:indexg+4]=t2
      #generate hopping terms -- explicitly construct 2 --> 1 hopping. Get 1 --> 2 by h.c.
      #from 2, g-mtm either stays put, or increments by b1+b2 or b2
      Hhop[indexg:indexg+2,indexg+2:indexg+4]=T1
      if g1<Ng1-1 and g2<Ng2-1:
        indexp = indexg+4+8*Ng2
        Hhop[indexp:indexp + 2,indexg+2:indexg+4]=T2
      if g2<Ng2-1:
        indexp = indexg + 4
        Hhop[indexp:indexp + 2,indexg+2:indexg+4]=T3

  return Hkin+Hhop+np.conjugate(np.transpose(Hhop))

def gen_coeff(pd):
  """
  Solves BM model and generates array for BM coeffients and energies.
  The encapsulating array of plane waves for coeff is rhombus-shaped, but the 
  allowed region is determined by circles about each Dirac point.
  
  Inputs:
  
  Outputs:
    coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,band,species] (N1,N2,2*Ng1,2*Ng2,2,2*n_active,4)
    sp_energy[ik1,ik2,tau,band] (N1,N2,2,2*n_active)
  """
  N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2']
  n_active=pd['n_active']
  M1,M2,b1,b2,Etens1,Etens2=gen_RLVs(pd)
  coeff = np.zeros((N1,N2,2*Ng1,2*Ng2,2,2*n_active,4),dtype=complex) #[ik1,ik2,g1+Ng1,g2+Ng2,tau,band,species]
  sp_energy = np.zeros((N1,N2,2,2*n_active)) #[ik1,ik2,tau,band]

  avgbands=0  
  #determine cutoff circle radius R [if unstrained, then R=1/2*(3*Ng1-2)]
  #the user may edit the code to specify an alternative choice of cutoff
  X=2/3*b1+1/3*b2
  Y=1/3*b1-1/3*b2
  RX1=np.linalg.norm((Ng1*b1-X)-b2*np.dot(Ng1*b1-X,b2)/np.dot(b2,b2))
  RX2=np.linalg.norm((Ng1*b2-X)-b1*np.dot(Ng1*b2-X,b1)/np.dot(b1,b1))  
  RY1=np.linalg.norm((Ng1*b1-Y)-b2*np.dot(Ng1*b1-Y,b2)/np.dot(b2,b2))
  RY2=np.linalg.norm((-Ng1*b2-Y)-b1*np.dot(-Ng1*b2-Y,b1)/np.dot(b1,b1))
  R=np.min((RX1,RX2,RY1,RY2))
  print("  coeff circular cutoff in units of b1 : %.2f" %(R/np.linalg.norm(b1)))
  print("  coeff circular cutoff in units of b2 : %.2f" %(R/np.linalg.norm(b2)))  
  for ik1 in range(N1):
    for ik2 in range(N2):
      tau = 0
      stau=1-2*tau # +1 for valley K, -1 for valley K'
      ham = gen_moire_hamiltonian(pd,(ik1,ik2)) #matrix index is species(1A,1B,2A,2B)+4*(g2+Ng2)+8*Ng2*(g1+Ng1)
      #figure out which plane waves fall within circular cutoff
      #sub_index is [ham index]
      sub_index=np.array([],dtype=int)
      for g1 in np.arange(-Ng1,Ng1):
        for g2 in np.arange(-Ng2,Ng2):
          indexg = 4*(g2+Ng2)+8*Ng2*(g1+Ng1)

          #layer 1
          Q=(ik1)*b1/N1+(ik2)*b2/N2+g1*b1+g2*b2+stau*1/3*b1-stau*1/3*b2
          modQ=np.sqrt(Q[0]**2+Q[1]**2)
          if modQ<R-0.00001: #0.00001 is to exclude momentum points that lie exactly at R
            sub_index=np.append(sub_index,np.arange(indexg,indexg + 2))
          #layer 2
          Q=(ik1)*b1/N1+(ik2)*b2/N2+g1*b1+g2*b2+stau*2/3*b1+stau*1/3*b2
          modQ=np.sqrt(Q[0]**2+Q[1]**2)
          if modQ<R-0.00001: 
            sub_index=np.append(sub_index,np.arange(indexg+2,indexg+4))

      cbandnum=np.size(sub_index)//2
      avgbands+=cbandnum*2
      sub_ham = ham[sub_index][:,sub_index] #take subblock of Hamiltonian within the circular cutoffs
      eigvals,eigvecs=np.linalg.eigh(sub_ham)
      sp_energy[ik1,ik2,tau,:]=eigvals[cbandnum-n_active:cbandnum+n_active]
      for index, value in enumerate(sub_index):
        species=value%4
        g2=(value//4)%(2*Ng2)-Ng2
        g1=value//(4*2*Ng2)-Ng1
        coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,:,species]=eigvecs[index,cbandnum-n_active:cbandnum+n_active]  

  #fix C2T gauge (Caution that this is insufficient in the presence of degeneracies. The HF runs correctly but form factors will be complex.)
  coeff_K=np.reshape(coeff[:,:,:,:,tau,:,:],(N1,N2,2*Ng1,2*Ng2,2*n_active,2,2)) #reshape species to layer/sub  
  coeff_C2T=np.zeros_like(coeff_K,dtype=complex)
  for sub in range(2):
    coeff_C2T[...,sub]=np.conj(coeff_K[...,1-sub])
  U_C2T=np.einsum("kKgGals,kKgGbls->kKab",np.conj(coeff_K),coeff_C2T,optimize=True)

  for ik1 in range(N1):
    for ik2 in range(N2):
      for a in range(2*n_active):
        sum_phase = -np.angle(U_C2T[ik1, ik2, a, a])
        coeff[ik1,ik2,:,:,tau,a,:]=np.exp(-1j*1/2*sum_phase)*coeff[ik1,ik2,:,:,tau,a,:].copy()

  #generate the other valley using time-reversal
  #this means that U_T does not need to be computed
  for ik1 in range(N1):
    for ik2 in range(N2):
      sp_energy[ik1, ik2, 1, :] = sp_energy[(-ik1)%N1,(-ik2)%N2, 0, :]
      for g1 in np.arange(-Ng1,Ng1):
        for g2 in np.arange(-Ng2,Ng2):
          gp1=-g1+(-ik1)//N1
          gp2=-g2+(-ik2)//N2
          if gp1>=-Ng1 and gp2>=-Ng2 and gp1<Ng1 and gp2<Ng2:
            coeff[ik1,ik2,g1+Ng1,g2+Ng2,1,:,:]=np.conj(coeff[(-ik1)%N1,(-ik2)%N2,gp1+Ng1,gp2+Ng2,0 ,:,:])
   
  return coeff,sp_energy


def gen_interaction(pd):
  """
  Generates array of interaction FTs.
  
  Inputs:
    
  Outputs:
    intFT[ik1,ik2,G1+NG1,G2+NG2]  (N1,N2,2*NG1,2*NG2)
      Here, momenta correspond to momentum transfers, so they are measured w.r.t. "global mtm origin".
      Factor of 1/area is included here, where area is the total system area
      The relative permittivity epsr is not introduced here. It is left as an input parameter to the HF code.
  """
  dsc=pd['dsc'];N1=pd['N1'];N2=pd['N2'];NG1=pd['NG1'];NG2=pd['NG2'];
  theta=np.pi/180*pd['theta'];gates=pd['gates']
  b1,b2=gen_RLVs(pd)[2:4]
  #calculate circular cutoff radius R [if unstrained, then R=3/2*NG1]
  #the user may edit the code to specify an alternative choice of cutoff
  R1=NG1*np.linalg.norm(b1-b2*np.dot(b1,b2)/np.dot(b2,b2))
  R2=NG1*np.linalg.norm(b2-b1*np.dot(b1,b2)/np.dot(b1,b1))
  R=np.min((R1,R2))
  print("  intFT circular cutoff in units of b1 : %.2f" %(R/np.linalg.norm(b1)))
  print("  intFT circular cutoff in units of b2 : %.2f" %(R/np.linalg.norm(b2)))    
  if gates not in ['dual','single']:
    raise Exception('invalid gate configuration')
  epsr=1 #the permittivity is introduced in the HF code, so we just set epsr=1 here
  U=echarge**2/(2*epsilon0*epsr)/echarge #in eV
  area=N1*N2*(4*np.pi**2)/(np.abs(b1[0]*b2[1]-b1[1]*b2[0]))/(2*kD*np.sin(theta/2))**2 #total real-space area of system
  intFT=np.zeros((N1,N2,2*NG1,2*NG2))
  for ik1 in np.arange(0,N1):
    for ik2 in np.arange(0,N2):
      for G1 in np.arange(-NG1,NG1):
        for G2 in np.arange(-NG2,NG2):
          Q=ik1*b1/N1+ik2*b2/N2+G1*b1+G2*b2
          modQ = np.linalg.norm(Q)
          if modQ<R-0.00001:
            #check if we include V(q=0)
            if ik1==0 and ik2==0 and G1==0 and G2==0:
              if pd['include_q=0']:
                print("  intFT includes q=0 contributions!")
                if gates=='dual':
                  intFT[ik1,ik2,G1+NG1,G2+NG2]=U*dsc/area
                elif gates=='single':
                  intFT[ik1,ik2,G1+NG1,G2+NG2]=2*U*dsc/area
              else:
                print("  intFT does not include q=0 contributions!")
                intFT[ik1,ik2,G1+NG1,G2+NG2]=0
            else:
              q=kD*2*np.sin(theta/2)*Q #reintroduce k_theta to get mtm in SI
              modq = np.linalg.norm(q)
              if gates=='dual':     
                intFT[ik1,ik2,G1+NG1,G2+NG2]=U*np.tanh(modq*dsc)/(modq)/area
              elif gates=='single':
                intFT[ik1,ik2,G1+NG1,G2+NG2]=U*(1-np.exp(-2*dsc*modq))/(modq)/area 
  return intFT
    


# =============================================================================
# Symmetry Transforms
# =============================================================================
def C2T_symmetry(pd,coeff):
  """
  Input:
    -coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,a,species]
  Output:
    -U[ik1,ik2,tau,a,b]
  """  
  N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2']
  n_active=pd['n_active'];n_bands=n_active  
  coeff=np.reshape(coeff,(N1,N2,2*Ng1,2*Ng2,2,2*n_bands,2,2)) #reshape species to layer and sub  
  coeff_C2T=np.zeros_like(coeff,dtype=complex)
  for sub in range(2):
    coeff_C2T[...,sub]=np.conj(coeff[...,1-sub])
  U_C2T=np.einsum("kKgGtals,kKgGtbls->kKtab",np.conj(coeff),coeff_C2T,optimize=True)
  return U_C2T

def C2_symmetry(pd,coeff):
  """
  Input:
    -coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,a,species]
  Output:
    -U[ik1,ik2,tau,a,b]
  """  
  N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2']
  n_active=pd['n_active'];n_bands=n_active
  coeff=np.reshape(coeff,(N1,N2,2*Ng1,2*Ng2,2,2*n_bands,2,2)) #reshape species to layer and sub  
  coeff_C2=np.zeros_like(coeff,dtype=complex)
  for ik1 in range(N1):
    for ik2 in range(N2):
      for g1 in np.arange(-Ng1,Ng1):
        for g2 in np.arange(-Ng2,Ng2):
          for tau in range(2):
            for sub in range(2):
              gp1=-g1+(-ik1)//N1
              gp2=-g2+(-ik2)//N2
              if gp1>=-Ng1 and gp2>=-Ng2 and gp1<Ng1 and gp2<Ng2:
                coeff_C2[ik1,ik2,g1+Ng1,g2+Ng2,tau,:,:,sub]=coeff[(-ik1)%N1,(-ik2)%N2,gp1+Ng1,gp2+Ng2,1-tau,:,:,1-sub]
  U_C2=np.einsum("kKgGtals,kKgGtbls->kKtab",coeff_C2.conj(),coeff,optimize=True)
  return U_C2


def C3_symmetry(pd,coeff):
  """
  Input:
    -coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,a,species]
  Output:
    -U[ik1,ik2,tau,a,b]
  """   
  N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2']
  n_active=pd['n_active'];n_bands=n_active
  phi=2*np.pi/3
  coeff=np.reshape(coeff,(N1,N2,2*Ng1,2*Ng2,2,2*n_bands,2,2)) #reshape species to layer and sub  
  coeff_C3=np.zeros_like(coeff,dtype=complex)
  for ik1 in range(N1):
    for ik2 in range(N2):
      for g1 in np.arange(-Ng1,Ng1):
        for g2 in np.arange(-Ng2,Ng2):
          for tau in range(2):
            for sub in range(2):
              #layer 1
              gp1=-g2+(-ik2)//N1
              gp2=g1-g2+(1-2*tau)+(ik1-ik2)//N2
              ip1=(-ik2)%N1
              ip2=(ik1-ik2)%N2
              if gp1>=-Ng1 and gp2>=-Ng2 and gp1<Ng1 and gp2<Ng2:            
                coeff_C3[ik1,ik2,g1+Ng1,g2+Ng2,tau,:,0,sub]=np.exp(1j*phi*(1-2*tau)*(1-2*sub))*coeff[ip1,ip2,gp1+Ng1,gp2+Ng2,tau,:,0,sub]              
              #layer 2
              gp1=-g2-(1-2*tau)+(-ik2)//N1
              gp2=g1-g2+(ik1-ik2)//N2
              ip1=(-ik2)%N1
              ip2=(ik1-ik2)%N2
              if gp1>=-Ng1 and gp2>=-Ng2 and gp1<Ng1 and gp2<Ng2:          
                coeff_C3[ik1,ik2,g1+Ng1,g2+Ng2,tau,:,1,sub]=np.exp(1j*phi*(1-2*tau)*(1-2*sub))*coeff[ip1,ip2,gp1+Ng1,gp2+Ng2,tau,:,1,sub]              
  U_C3=np.einsum("kKgGtals,kKgGtbls->kKtab",np.conj(coeff_C3),coeff,optimize=True)
  return U_C3   

def get_symmetry(pd,coeff):
  """
  Input:
    -coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,a,species]
  Outputs:
    -U[ik1,ik2,tau,a,b]
  """
  U_C2T=C2T_symmetry(pd,coeff) #[ik1,ik2,tau,a,b]
  U_C2=C2_symmetry(pd,coeff) #[ik1,ik2,tau,a,b]
  U_C3=C3_symmetry(pd,coeff) #[ik1,ik2,tau,a,b]
  
  symchecks = symmetry_check(pd,U_C2T,U_C2,U_C3)
  
  return U_C2T,U_C2,U_C3, symchecks

def symmetry_check(pd,U_C2T,U_C2,U_C3):
  """
  Note that symmetry checks may fail if the BM bandstructure has degeneracy and the choice of active
  bands only includes a subset of the states in this degenerate space.
  Inputs:
    -U[ik1,ik2,tau,a,b]
  """
  n_act = pd["n_active"]

  U0=np.zeros_like(U_C2T)
  U0[:,:,:]=np.eye(2*n_act)
  print("---> Symmetry checks on coeff")
  #C2T
  V=np.einsum("kKtab,kKtcb->kKtac",U_C2T,np.conj(U_C2T), optimize=True)
  C2Tcheck=np.max(np.abs(V-U0))
  print("C2T: %.2e" %C2Tcheck)
  #C2
  V=np.einsum("kKtab,kKtcb->kKtac",U_C2,np.conj(U_C2), optimize=True)
  # C2check=np.max(np.abs(V-U0))
  C2check=np.max(np.abs((V-U0))[:, :, 1, :, :])
  print("C2: %.2e" %C2check)  
  #C3
  V=np.einsum("kKtab,kKtcb->kKtac",U_C3,np.conj(U_C3), optimize=True)
  C3check=np.max(np.abs(V-U0))
  print("C3: %.2e" %C3check) 

  return np.array([C2Tcheck,C2check,C3check])


# =============================================================================
# Matrix element generation functions
# =============================================================================

def gen_form_factors(pd,c,cp):
  
  """
  Generates the intravalley form factors (lambdas) given input c and cp 
  (obtained e.g. by part of coeff from gen_coeff).
  Outputs array form[ik1,ik2,iq1,iq2,G1+NG1,G2+NG2], which is the quantity:
    <u(k)|u(k+q+G)> in periodic gauge.
  c is associated with the bra, while cp is associated with the ket.
  (ik1,ik2) represents a mBZ momentum.
  q=(iq1/N1+G1)*b1+(iq2/N2+G2)*b2 represents a momentum transfer.
  
  Note that we do not explicitly impose a cutoff on q in the form factor,
  because it will be implicitly truncated by the q-cutoff in the interaction V(q).

  [For legacy reasons, we first generate 
  form[ik1,ik2,ikp1,ikp2,G1+NG1,G2+NG2,...]=<u(k)|u(k'-G)>
  before converting to 
  form[ik1,ik2,iq1,iq2,G1+NG1,G2+NG2,...]=<u(k)|u(k+q+G)>]
  
  Inputs:
    c,cp[ik1,ik2,g1+Ng1,g2+Ng2,species] (N1,N2,2*Ng1,2*Ng2,4)
    
  Outputs:
    form[ik1,ik2,iq1,iq2,G1+NG1,G2+NG2] (N1,N2,N1,N2,2*NG1,2*NG2)
  """
  NG1=pd['NG1'];NG2=pd['NG2'];N1=pd['N1'];N2=pd['N2'];Ng1=pd['Ng1'];Ng2=pd['Ng2']
  if NG1 > 2*Ng1 -1 or NG2 > 2*Ng2 - 1:
    raise Exception("NG should be smaller than 2*Ng.") 
  form=np.zeros((N1,N2,N1,N2,2*(NG1+1),2*(NG2+1)),dtype=complex) #OLD form factors
  for G1 in np.arange(-NG1-1,NG1+1):
    for G2 in np.arange(-NG2-1,NG2+1):
        cp_copy = np.roll(cp, (G1,G2), axis = (2,3))
        if G1 > 0:
            cp_copy[:,:,0:G1,:,:] = 0
        if G1 < 0:
            cp_copy[:,:,2*Ng1+G1:,:,:] = 0
        if G2 > 0:
            cp_copy[:,:,:,0:G2,:] = 0
        if G2 < 0:
            cp_copy[:,:,:,2*Ng2+G2:,:] = 0
        form[:,:,:,:,G1+NG1+1,G2+NG2+1]=np.einsum("abefz,cdefz->abcd",np.conj(c),cp_copy,optimize=True)
  #convert to form factor convention <u(k)|u(k+q+G)>
  form_new=np.zeros((N1,N2,N1,N2,2*NG1,2*NG2),dtype=complex)
  for ik1 in range(N1):
    for ik2 in range(N2):
      for iq1 in range(N1):
        for iq2 in range(N2):
          form_new[ik1,ik2,iq1,iq2,:,:]=form[ik1,ik2,(ik1+iq1)%N1,(ik2+iq2)%N2,
                                             2-(ik1+iq1)//N1:2*NG1+2-(ik1+iq1)//N1,
                                             2-(ik2+iq2)//N2:2*NG2+2-(ik2+iq2)//N2]
  form_new=np.flip(form_new,axis=(4,5))
  return form_new

# =============================================================================
# Main Functions
# =============================================================================

def singleP(pd):
  """
  Main function that generates quantities required for self-consistent HF.

  pd is a dictionary that contains all the input parameters
  
  Flowchart:
  -Solve continuum model to obtain coeff, sp_energy, U_C2T
  -Generate interaction FT
  
  Saved quantities [index structure] (dimensions):
    intFT[ik1,ik2,G1+NG1,G2+NG2] (N1,N2,2*NG1,2*NG2)
    coeff[ik1,ik2,g1+Ng1,g2+Ng2,tau,band,species] (N1,N2,2*Ng1,2*Ng2,2,2*n_active,4) 
    sp_energy[ik1,ik2,tau,band] (N1,N2,2,2*n_active) 
    U_C2T[ik1,ik2,tau,a,b] (N1,N2,2,2*n_active,2*n_active)

  Comments:
    -ik1 takes values in 0,...,N1-1
    -g1 takes values in -Ng1,...,Ng1-1
    -For example, the momentum of a plane wave state is k=(ik1/N1+g1)*b1+(ik2/N2+g2)*b2
     gen_coeff implements a circular-based cutoff on the allowed momenta of plane wave states
    -G1 takes values in -NG1,...,NG1-1
    -For example, a momentum transfer can be expressed as q=(ik1/N1+G1)*b1+(ik2/N2+G2)*b2
     gen_interaction implmenets a circular cutoff on the allowed momentum transfers
    -tau=0,1 indexes valley K,K' respectively
    -band (also referred to as a or b) takes values in 0,...,2*n_active-1, and indexes the active bands from lowest to highest BM energy
    -species=0,1,2,3 indexes the 1A,1B,2A,2B layer/sublattice respectively
  """
  
  #generate Bloch coefficients and kinetic energies of the BM model
  loc_time = time.time()
  coeff,sp_energy = gen_coeff(pd)  
  
  #obtain symmetry representations  
  U_C2T,U_C2,U_C3, symchecks = get_symmetry(pd,coeff)
  
  #generate intFT
  intFT=gen_interaction(pd)
  
  #save quantities needed
  np.save("coeff.npy",coeff) #[ik1,ik2,g1+Ng1,g2+Ng2,tau,band,species]
  np.save("sp_energy.npy",sp_energy) #[ik1,ik2,tau,band]
  np.save("intFT.npy",intFT) 
  np.save("U_C2T.npy",U_C2T)
  #np.save("U_C2.npy",U_C2)
  #np.save("U_C3.npy",U_C3)
  np.save("symchecks.npy",symchecks)

  
