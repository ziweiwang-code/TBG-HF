#!/usr/bin/env python3

import numpy as np
import time
import json
import os
import shutil
import matplotlib.pyplot as plt

from mainProgram import mainP

folder=os.getcwd()

TRS=[]
TRSp=[]
IVC=[]
bandgap=[]
boostl=[]
strainl=np.arange(0,0.0031,0.0005)

for strain in strainl:
  pd={}
  pd["strain"]=strain

  pd['intdir']= os.path.join(folder, f"strain_{strain:.4f}") #contains output of singleP
  pd['rootdir']= folder #contains all scripts
  pd['outdir'] = os.path.join(folder, f"results_{strain:.4f}") #HF results output
  
  print(pd['intdir'])
  
  lowest_energy=10**10
  for boost in [0,4]: 
    for iseed in range(10):  
      
      dir_name = pd['outdir'] + f"/b{boost:d}_s{iseed:d}"
      
      os.chdir(dir_name)
      
      with open("output.json") as f:
        output = json.load(f)
      if output["energy"]<lowest_energy:
        lowest_energy_output=output
        lowest_energy=output["energy"]
        lowest_energy_boost=boost
    
  IVC.append(lowest_energy_output["IVC"])
  bandgap.append(lowest_energy_output["gap"]*1000)
  TRS.append(lowest_energy_output["T break"])
  TRSp.append(lowest_energy_output["Tp break"])
  boostl.append(lowest_energy_boost)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid

# Top-left
axs[0, 0].plot(100*strainl, IVC,marker="x")
axs[0, 0].set_xlabel("strain (%)")
axs[0, 0].set_title("IVC")
axs[0, 0].set_ylim(bottom=0)

# Top-right
axs[0, 1].plot(100*strainl, bandgap,marker="x")
axs[0, 1].set_xlabel("strain (%)")
axs[0, 1].set_title("gap (meV)")
axs[0, 1].set_ylim(bottom=0)

# Bottom-left
axs[1, 0].plot(100*strainl, TRS, label="T",marker="x")
axs[1, 0].plot(100*strainl, TRSp, label="T'",marker="x")
axs[1, 0].set_xlabel("strain (%)")
axs[1, 0].legend()
axs[1, 0].set_title("T and T' breaking")

# Bottom-right
axs[1, 1].plot(100*strainl, boostl,marker="x")
axs[1, 1].set_xlabel("strain (%)")
axs[1, 1].set_title("boost")

plt.tight_layout()
plt.savefig(folder+"/example2.png",dpi=300)
plt.show()
