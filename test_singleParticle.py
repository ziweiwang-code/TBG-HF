#!/usr/bin/env python3
# FOR TESTING PURPOSE

import numpy as np
import os
import json
import time
from singleParticle import singleP

with open("int_input.json") as f:
  pd = json.load(f) #pd is a dictionary containing the parameters

pd['intdir']=os.path.join(os.getcwd(), "bandstructure")
os.makedirs(pd['intdir'], exist_ok = True)
os.chdir(pd['intdir'])

# =============================================================================
# Code Execution
# =============================================================================

start_time = time.time() #start timer
print("\n--- test_singleParticle started ---\n")

singleP(pd)

with open('int_pd.json', 'w') as f:
  json.dump(pd, f, indent=2) #save pd to JSON, to be used in the Hartree-Fock code

print("\n--- test_singleParticle done: %.2f seconds ---" % (time.time() - start_time))  
