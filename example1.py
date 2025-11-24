#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
import json
from singleParticle import gen_moire_hamiltonian

pd = {}

#override from external input
with open("int_input.json") as f:
  pd_input = json.load(f)
pd = {**pd,**pd_input}


# =============================================================================
# Main Functions
# =============================================================================

def bandstruct(pd):
  Ng1=pd['Ng1'];Ng2=pd['Ng2'];
  cbandnum=8*Ng1*Ng2
  energies_K=np.zeros((Nk,2*nbands))
  #We caution that we do not truncate to a circular plane wave cutoff in this example code
  #The difference is not significant for sufficiently large Ng1,Ng2
  for index, k in enumerate(klist):
    energies_K[index,:]=np.linalg.eigh(gen_moire_hamiltonian(pd,(k,0)))[0][cbandnum-nbands:cbandnum+nbands]
  return energies_K



  

  
#this generates Nk evenly spaced points points from -b1/2 to b1/2
Nk=50
klist=np.linspace(-pd["N1"]/2,pd["N1"]/2,Nk) 

nbands=5 #2*nbands bands in valley K will be plotted




fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)  # 1 row, 2 columns

# --- First plot: strain = 0 ---
pd["strain"] = 0
energies_K = bandstruct(pd)
for i in range(2 * nbands):
  axs[0].plot(klist/pd["N1"],1000 * energies_K[:, i])
axs[0].set_ylim((-100, 100))
axs[0].set_ylabel(r'Energy (meV)')
axs[0].set_title("strain = 0%")
axs[0].set_xlabel("$k_1/|b_1|$")

# --- Second plot: strain = 0.3% ---
pd["strain"] = 0.003
energies_K = bandstruct(pd)
for i in range(2 * nbands):
  axs[1].plot(klist/pd["N1"],1000 * energies_K[:, i])
axs[1].set_ylim((-100, 100))
axs[1].set_title("strain = 0.3%")
axs[1].set_xlabel("$k_1/|b_1|$")

plt.tight_layout()
plt.savefig("example1.png",dpi=300)
plt.show()

