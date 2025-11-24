#!/usr/bin/env python3

import numpy as np
import os
import json
import time
from singleParticle import singleP, get_symmetry

pd = {}

#override from external input
with open("int_input.json") as f:
  pd_input = json.load(f)
pd = {**pd,**pd_input}

folder=os.getcwd()

for pd['strain'] in np.arange(0,0.0031,0.0005):
  strain=pd["strain"]
  pd['intdir']=os.path.join(folder, f"strain_{strain:.4f}")
  pd['N1'] = 12; pd['N2'] = 12
  pd["n_active"] = 1
  
  os.makedirs(pd['intdir'], exist_ok = True)
  os.chdir(pd['intdir'])
  
  #save dictionary
  with open('int_pd.json', 'w') as f:
    json.dump(pd, f, indent=2)
  
  # =============================================================================
  # Code Execution
  # =============================================================================
  
  start_time = time.time() #start timer
  print("\n--- example2_singleParticle.py started ---\n")
  
  singleP(pd)
  
  print("\n--- example2_singleParticle.py done: %.2f seconds ---" % (time.time() - start_time))  
