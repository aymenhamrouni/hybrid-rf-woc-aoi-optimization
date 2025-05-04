#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:39:55 2024

@author: ahamroun
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:44:02 2024

@author: Aymen Hamrouni
"""

#%cls
#%clear all
from collections import defaultdict
from docplex.mp.relaxer import Relaxer
import docplex.mp.model as cpx
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_generation
import multiprocessing
from joblib import Parallel, delayed
from utils import *
import sys
#from communication import *
from milpModel import *
from matplotlib.pyplot import plot, ion, show
ion()
#%matplotlib auto
import mpld3

from time import sleep
#from tqdm.contrib.telegram import tqdm, trange
from tqdm import tqdm
import paretoMILP
from paretoMILP import *
import itertools
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','no-latex'])




plt.close('all')

plt.show(block = False)

Nmontecarlo = 10
#size=2 #5 km is the size

# define the number of cores (this is how many processes wil run)
num_cores = multiprocessing.cpu_count()
# execute the function in parallel - `return_list` is a list of the results of the function
# in this case it will just be a list of None's



all_pareto_fronts1 = defaultdict(lambda: 0)
all_pareto_fronts2 = defaultdict(lambda: 0)

ZeroDelay = defaultdict(lambda: 0)

n=4
t=15 
    

#for i in trange(Nmontecarlo, token='6972663208:AAGa9eWpdM3sG5hLONyYFolo04Sb4D41gUg', chat_id='6729112338'):
for i in tqdm(range(Nmontecarlo)):

    simulation = data_generation.NetworkCharacteristics(n, t)
    T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp = simulation.generate_messages(messageSize=4,
        mini=1,maxi=4,appSize=1)
    #PriorityLevels = [0.2, 0.5, 1]
    #Pt = simulation.generate_priorities(msgQueues, PriorityLevels)

    V_RF = simulation.generate_visibility(95, 100)
    V_L = simulation.generate_visibility(95, 100)

    
    # introduce freshness, data needs to be sent asap as it has ending and begin date
    # the ending represent where the data is no longer fresh and we would want to maxmimize freshness
    # later on this MessageBegin and MessageEnd we can get it from a preprocessing model

    # multi-objective function
    # generate energy for coin cell batteries such as wearables and sensors
    EnergyTotal_L = simulation.generate_energy(200, 900)
    EnergyTotal_RF= simulation.generate_energy(200, 900)
    
    #NodesLoc=initialize_nodes(n,size)
    #x, y = zip(*NodesLoc)
    #plt.figure()
    #plt.scatter(x, y)
    #plt.show()
    #Distances=calculate_distances(NodesLoc)
    

    MILP=paretomodel(2,n,t,T,V_RF,V_L,EnergyTotal_L,EnergyTotal_RF,msgQueues,BiggestMsg)

    all_pareto_fronts1,all_pareto_fronts2,ZeroDelay=MILP.buildMILP(all_pareto_fronts1,all_pareto_fronts2,ZeroDelay)
    
s=[r for r in ZeroDelay.values()]
    # Extract values for plotting
#print(all_pareto_fronts1)
#print(all_pareto_fronts2)

f1_values = [p*100 for p in all_pareto_fronts1.values()]
f2_values = [p*100 for p in all_pareto_fronts2.values()]

# Plotting





plt.figure(figsize=(10, 6))
degree=np.linspace(0, max(max(all_pareto_fronts1)[0],max(all_pareto_fronts2)[0]), len(f1_values))
plt.scatter(f2_values,f1_values,s=s, c=degree, cmap='viridis')



plt.plot(100,0,'ro')

cbar = plt.colorbar()
cbar.set_label(label=r'Regularization $\alpha_1$ for Energy Consumption (RF)', size='large', weight='bold')
cbar.ax.tick_params(labelsize=12)


plt.annotate('*', (f2_values[-2], f1_values[-2]),fontsize=15)

plt.annotate('*', (f2_values[-1], f1_values[-1]),fontsize=15)



plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel('Avg. Delay (%)',fontsize=15)
#plt.xlim((0.25,0.6))
plt.ylabel('Avg. Consumed Energy (%)',fontsize=15)
plt.grid(True)
plt.subplots_adjust(hspace=0)

degree=np.linspace(0, max(max(all_pareto_fronts1)[0],max(all_pareto_fronts2)[0]), len(f1_values))
plt.scatter(f3_values,f4_values,s=[50,50,60,90,100,100,160,200] ,c=np.linspace(0, 0.15, len(f3_values)), cmap='plasma')
plt.annotate('*', (f3_values[-1], f4_values[-1]),fontsize=15)

cbar = plt.colorbar(orientation="vertical")
cbar.set_label(label=r'Regularization $\alpha_1$ for Energy Consumption (OC)', size='large', weight='bold')
cbar.ax.tick_params(labelsize=12)

#print(all_pareto_fronts)
plt.show()



 











   

   
   
    
   

