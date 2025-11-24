#!/usr/bin/env python3

import numpy as np
import time
import os

from constants import *
import routines as routines
import projectors as projectors
import measure as measure
from singleParticle import gen_form_factors

# =============================================================================
# HF
# =============================================================================

def mainP(pd):
  """
  """
  N1=pd['N1'];N2=pd['N2'];NG1=pd['NG1'];NG2=pd['NG2'];n_active=pd['n_active']
  HF_itermax=pd['HF_itermax'];HF_tolerance=pd['HF_tolerance']
  measure_period=pd['measure_period']
  boost1=pd['boost1'];boost2=pd['boost2']

  outdir = os.getcwd()
  os.chdir(pd['intdir'])
  #load arrays from output of singleParticle.py
  coeff=np.load("coeff.npy") #[ik1,ik2,g1+Ng1,g2+Ng2,tau,a,species]
  sp_energy=np.load("sp_energy.npy") #[ik1,ik2,tau,a]
  U_C2T=np.load("U_C2T.npy") #[ik1,ik2,tau,a,b]
  intFT=np.load("intFT.npy")/pd['epsr'] #[ik1,ik2,G1+NG1,G2+NG2]
  
  #generate form factors
  form = np.zeros((N1,N2,N1,N2,2*NG1,2*NG2,2,2*n_active,2*n_active),dtype=complex) #[ik1,ik2,ip1,ip2,G1+NG1,G2+NG2,tau,band1,band2]
  for band1 in range(2*n_active):
    for band2 in range(2*n_active):
      form[:,:,:,:,:,:,0,band1,band2]=gen_form_factors(pd,coeff[...,0,band1,:],coeff[...,0,band2,:])
      form[:,:,:,:,:,:,1,band1,band2]=gen_form_factors(pd,coeff[...,1,band1,:],coeff[...,1,band2,:])
  if np.max(np.abs(np.imag(form)))>1e-9:   
    print("! Complex intravalley form factors detected! form will be kept complex.")
    print("  %.2e" %(np.max(np.abs(np.imag(form)))))    
  else:
    form=np.real(form)
  
  os.chdir(outdir)

  #boost quantities in valley Kp 
  #so the boosted k actually corresponds to k in valley K and k+q in valley Kp
  if boost1!=0 or boost2!=0:
    boost=np.array([-boost1,-boost2])
    coeff[:,:,:,:,1,:,:]=np.roll(coeff[:,:,:,:,1,:,:],boost,axis=(0,1))
    sp_energy[:,:,1,:]=np.roll(sp_energy[:,:,1,:],boost,axis=(0,1))
    U_C2T[:,:,1,:,:]=np.roll(U_C2T[:,:,1,:,:],boost,axis=(0,1))  
    form[:,:,:,:,:,:,1,:,:]=np.roll(form[:,:,:,:,:,:,1,:,:],boost,axis=(0,1))
  
  #process one-body terms and reshape
  #sp_energy [tk1,tk2,tau,a]
  #H_SP [tk1,tk2,s,tau*a,taup*b] 
  H_SP=routines.gen_H_SP(pd,sp_energy)


  #prepare M and tVE
  print("\n---> generating exchange terms...")
  #form [tk1,tk2,tq1,tq2,G1,G2,tau,a,b]
  #M [tk1,tk2,tq1,tq2,G1,G2,tau,a,b]
  #tVE [tk1*tk2*a*d,tq1*tq2*b*c,tau*taup]
  form,M,tVE=routines.gen_M_tVE(pd,form,intFT)
  

  #generate input and reference projectors
  print("\n---> generating input projector...")
  P_in=projectors.gen_projector(pd,pd['in_choice'],sp_energy)
  print("\n---> generating reference projector...")
  P_ref=projectors.gen_projector(pd,pd['ref_choice'],sp_energy)
  
  
  #get runtime estimates for HF routines
  ntimetest=5
  print("\n---> runtime estimate per iteration (averaged over %d iterations)" %ntimetest)
  timelist=np.zeros(4)
  for n in range(ntimetest):
    focktimes,HFham_temp=routines.calc_fock_matrix(pd,P_in,form,M,tVE,timeit=True)
    timelist[0:2]+=focktimes
    timelist[2:4]+=routines.aufbau(pd,HFham_temp,timeit=True) 
  timelist=timelist/ntimetest
  print("    direct term : %.4f seconds" %timelist[0])
  print("    exchange term : %.4f seconds" %timelist[1])
  print("  total calc_fock_matrix() : %.4f seconds" %(timelist[0]+timelist[1]))
  print("    diagonalization : %.4f seconds" %timelist[2])
  print("    projector construction : %.4f seconds" %timelist[3])
  print("  total aufbau() : %.4f seconds" %(timelist[2]+timelist[3]))
   
  
  #measure input projector
  print("\n---> input projector properties:")
  st=time.time()
  HFham_D,HFham_E=routines.calc_fock_matrix(pd,P_in-P_ref,form,M,tVE)
  E=routines.calc_E(pd,P_in,P_ref,sp_energy,HFham_D,HFham_E)  
  HFeigs,fill_index=routines.aufbau(pd,H_SP+HFham_D+HFham_E)[1:3]
  measure.measure_projector(pd,P_in,E,U_C2T,HFeigs,fill_index)
  print("\n  total measurement time : %.4f seconds" %(time.time()-st)) 
  
  
  #start HF iteration
  P_old=P_in.copy()
  P_new=P_in.copy()
  difference=0
  print("\n---> begin HF iteration")
  for HF_iter in range(HF_itermax):
        
    HFham_D_old,HFham_E_old=routines.calc_fock_matrix(pd,P_old-P_ref,form,M,tVE)
    H_old=H_SP+HFham_D_old+HFham_E_old
    P_new=routines.aufbau(pd,H_old)[0]
    difference=1/(N1*N2)*np.linalg.norm(P_old-P_new)

    #generate new HF projector
    if pd['HF_type']=='iteration':
      l=1    
    elif pd['HF_type']=='ODA':
      dP=P_new-P_old
      coef_1=np.real(np.einsum("kKsab,kKsab->",H_SP,dP,optimize=True))
      H_dP=np.sum(routines.calc_fock_matrix(pd,dP,form,M,tVE),axis=0)   
      coef_01=np.real(np.einsum("kKsab,kKsab->",H_dP,P_old-P_ref,optimize=True))
      coef_11=np.real(np.einsum("kKsab,kKsab->",H_dP,dP,optimize=True)) 
      lin=coef_1+coef_01
      quad=0.5*coef_11 
      if lin>0 and np.abs(lin)>1e-12:
        print("!  positive linear ODA coefficient detected : %.4e" %lin)
        l=1
      elif quad<=-lin/2:
        l=1
      elif np.abs(lin)<1e-12:
        l=1
      else:
        l=-lin/2/quad 
      P_new=(1-l)*P_old+l*P_new
    
    #measure projector
    if HF_iter%measure_period==0:
      print("\n  ===== iteration : %d =====" %HF_iter)
        
      # ====== start standard measure block
      HFham_D,HFham_E=routines.calc_fock_matrix(pd,P_new-P_ref,form,M,tVE)
      E=routines.calc_E(pd,P_new,P_ref,sp_energy,HFham_D,HFham_E)
      HFeigs,fill_index=routines.aufbau(pd,H_SP+HFham_D+HFham_E)[1:3]        
      measure.measure_projector(pd,P_new,E,U_C2T,HFeigs,fill_index,difference=difference)
      # ====== end standard measure block
      
    #check tolerance
    if difference<HF_tolerance and HF_iter>pd['HF_itermin']:
      print("\n!  tolerance reached at iteration "+str(HF_iter))
      break

    if HF_iter==HF_itermax-1:
      print("\n!  HF terminated at iteration "+str(HF_iter)) 
    
    #replace projector for next iteration
    P_old=P_new.copy()


  #final measurement of projector
  print("\n---> final projector properties:")  
  HFham_D,HFham_E=routines.calc_fock_matrix(pd,P_new-P_ref,form,M,tVE)
  E=routines.calc_E(pd,P_new,P_ref,sp_energy,HFham_D,HFham_E)
  HFeigs,fill_index, evecs=routines.aufbau(pd,H_SP+HFham_D+HFham_E)[1:4]
  out=measure.measure_projector(pd,P_new,E,U_C2T,HFeigs,fill_index,finalout=True,difference=difference)
  return E[0],out,P_new, HFeigs,fill_index,evecs
  

