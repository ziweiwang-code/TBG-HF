#!/usr/bin/env python3

import numpy as np
      
def time_reversal_P(P, boost1 ,boost2):
  """
  Note that U_T is unity given the way that we construct the K' BM states from those in K
  """
  P_T = np.flip(P, axis = (0 ,1, 3 ,5)).copy()
  P_T = np.roll(P_T, (1, 1), axis = (0, 1))
  P_T = np.roll(P_T, (-boost1, -boost2), axis = (0, 1))
  P_T = np.conj(P_T)

  return P_T

def measure_projector(pd,P,E,U_C2T,HFeigs,fill_index,finalout=False,difference=None):
  """
  Inputs:
    -P [ik1,ik2,s,tau*a,taup*b] 
    -U_C2T [ik1,ik2,tau,a,b]
    -HF_eigs []
    -fill_index []
    
  ! For the order of indices in the projector: 
  we use the convention where P[ik1,ik2,s,alpha,beta] = <c^\dagger_{alpha,s}(ik1,ik2) c_{beta,s}((ik2,ik2))> !
  """
  N1=pd['N1'];N2=pd['N2'];nact=pd['n_active']
  boost1=pd['boost1'];boost2=pd['boost2']
  
  #Pex [tk1,tk2,s,tau,a,taup,b]
  Pex=np.reshape(P,(N1,N2,2,2,2*nact,2,2*nact)).copy()

  #gap
  #note that this defined as 
  #gap=(energy of lowest unoccupied HF state)-(energy of highest occupied HF state)
  mask = np.zeros_like(HFeigs, dtype = bool)
  mask[fill_index] = True
  fill_max=np.max(HFeigs[mask])
  empty_min=np.min(HFeigs[~mask])
  gap=empty_min-fill_max
  
  #occupations
  
  occupations = np.einsum("kKsa -> kKs" , np.reshape(mask, (N1,N2,2,4*nact)).astype(int))
  max_occ_up = int(np.max(occupations[:,:,0]))
  min_occ_up = int(np.min(occupations[:,:,0]))
  max_occ_down = int(np.max(occupations[:,:,1]))
  min_occ_down = int(np.min(occupations[:,:,1]))
  
  
  #flavor polarization
  FlavorP = np.real(np.einsum("kKstata->st",Pex,optimize=True)/(N1*N2))
  SpinP = FlavorP[0 ,0] + FlavorP[0, 1] - FlavorP[1, 0] - FlavorP[1, 1]
  ValleyP = FlavorP[0 ,0] + FlavorP[1, 0] - FlavorP[0, 1] - FlavorP[1, 1]
  
  #IVC order (IVC component only)
  IVC = np.linalg.norm(Pex[:,:,:,0,:,1,:])**2/(N1*N2)
    
  #C2T breaking
  P_C2T=np.einsum("kKtac,kKTbd,kKstcTd->kKstaTb", U_C2T.conj(),U_C2T,np.conj(Pex),optimize=True)
  C2Tbreak=np.square(np.linalg.norm(Pex - P_C2T))/(N1*N2)
  
  #T and Tp breaking
  P_T = time_reversal_P(Pex, boost1, boost2)

  #Tp is spinless T following by a \pi U(1)_v rotation
  P_Tp = P_T.copy()
  P_Tp[:, :, :, 0, :, 1, :] =  - P_T[:, :, :, 0, :, 1, :]
  P_Tp[:, :, :, 1, :, 0, :] =  - P_T[:, :, :, 1, :, 0, :]

  Tbreak = np.square(np.linalg.norm(Pex - P_T))/(N1*N2)
  Tpbreak = np.square(np.linalg.norm(Pex - P_Tp))/(N1*N2)

  out = {}
  out['energy'] = E[0]
  out['gap'] = gap
  out['difference'] = difference
  out['valley polarization'] = ValleyP
  out['IVC'] = IVC
  out['spin polarization'] = SpinP
  out['C2T break'] = C2Tbreak
  out['T break'] = Tbreak
  out['Tp break'] = Tpbreak
  out['Max. Occ. Spin Up'] = max_occ_up
  out['Min. Occ. Spin Up'] = min_occ_up
  out['Max. Occ. Spin Down'] = max_occ_down
  out['Min. Occ. Spin Down'] = min_occ_down

  for key in out:
    val = out[key]
    if val == None:
      continue
    if isinstance(val, int):
      print(f'  {key} : {val:d}')
    else:
      print(f'  {key} : {val:.4e}')

  if finalout==True:
    return out
