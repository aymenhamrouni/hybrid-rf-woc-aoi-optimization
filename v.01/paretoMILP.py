#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:41:08 2024

@author: ahamroun
"""



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
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict

# millijoules per second, we suppose that one message last 1 second
SendEnergy = [15, 30] # battery consumption in mW for RF and Light
RecieveEnergy = [8 ,2] # battery consumption in mW for RF and Light

thresholdRF = 0.92
thresholdL = 0.92


class paretomodel:
    def __init__(self, M,n,t,T,V_RF,V_L,EnergyTotal_L,EnergyTotal_RF,msgQueues,BiggestMsg):
       self.M=M
       self.n=n
       self.t=t
       self.T=T
       self.V_RF=V_RF
       self.V_L=V_L
       self.EnergyTotal_L=EnergyTotal_L
       self.EnergyTotal_RF=EnergyTotal_RF
       self.counter=0
       self.msgQueues=msgQueues
       self.BiggestMsg=BiggestMsg 


    
    def buildMILP(self, all_pareto_fronts1,all_pareto_fronts2,ZeroDelay):
       
    
        
        M=self.M
        n=self.n
        t=self.t
        T=self.T
        V_RF=self.V_RF
        V_L=self.V_L
        EnergyTotal_L=self.EnergyTotal_L
        EnergyTotal_RF=self.EnergyTotal_RF

        msgQueues=self.msgQueues
        BiggestMsg =self.BiggestMsg

        
        opt_model = cpx.Model(name="MIP Model")
        x = {(i, j, m, k): opt_model.binary_var(name="x_{}_{}_{}_{}".format(i, j, m, k))
             for i in range(n) for j in range(n) for m in range(M) for k in range(t)}
    
        # determine which message was sent from i to j
        x_m = {}
        for i in range(n):
            for j in range(n):
                for f in range(0, len(msgQueues[i][j])):
                    x_m.update({(i, j, f): opt_model.binary_var(
                        name="xm_{}_{}_{}".format(i, j, f))})
    
        # represent which mode the node i used to communicate at time step k
        delta = {(i, j, k): opt_model.integer_var(lb=0, ub=BiggestMsg+1, name="delta_{}_{}_{}".format(i, j, k))
                 for i in range(n) for j in range(n) for k in range(t)}
        status = {(i, k): opt_model.binary_var(
            name="status_{}_{}".format(i, k)) for i in range(n) for k in range(t)}
        temp0 = {(i, j, k): opt_model.binary_var(name="temp0_{}_{}_{}".format(
            i, j, k)) for i in range(n) for j in range(n) for k in range(t)}
        temp1 = {(i, j, k): opt_model.binary_var(name="temp1_{}_{}_{}".format(
            i, j, k)) for i in range(n) for j in range(n) for k in range(t)}
    
        #switch = {(i): opt_model.integer_var(
            #lb=0, ub=t, name="switch_{}".format(i)) for i in range(n)}
        
        #AoI = {(i,j, k): opt_model.integer_var(
           # name="AoI_{}_{}_{}".format(i, j,k)) for i in range(n) for j in range(n) for k in range(t)}
        
        
    
        # node can choose only one mode and one node to communicate with at time step k
        constraints1 = {(i, k): opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for j in range(
            n) for m in range(M)) <= 1, ctname="constraint1_{}_{}".format(i, k)) for i in range(n) for k in range(t)}
    
        # node can can only recieve from one node with one mode at time step k
        constraints2 = {(j, k): opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for i in range(
            n) for m in range(M)) <= 1, ctname="constraint2_{}_{}".format(j, k)) for j in range(n) for k in range(t)}
    
        constraints22 = {(i, j, m, k): opt_model.add_constraint(ct=T[k][i][j] >= x[i, j, m, k], ctname="constraint22_{}_{}_{}_{}".format(
            i, j, m, k)) for i in range(n) for j in range(n) for m in range(M) for k in range(t)}
    
        # if node choose to communicate in radio, it should be feasible
        # constraints3 = { (i,j,k) : opt_model.add_constraint(ct=opt_model.if_then( (x[i,j,0,k] == 1) , opt_model.logical_or(temp0[i,j,k] *V[k][i][j]==2,temp0[i,j,k] *V[k][i][j]==3)==1) ,ctname="constraint3_{}_{}_{}".format(i,j,k)) for i in range(n) for j in range(n) for k in range(t)}
        constraints3 = {(i, j, k): opt_model.add_constraint(ct=temp0[i, j, k] * V_RF[k][i][j] >= thresholdRF*x[i, j, 0, k],
                                                            ctname="constraint3_{}_{}_{}".format(i, j, k)) for i in range(n) for j in range(n) for k in range(t)}
    
        # if node choose to communicate in light, it should be feasible
        # constraints4 = { (i,j,k) : opt_model.add_constraint(ct=opt_model.if_then( (x[i,j,1,k] == 1) , opt_model.logical_or(temp1[i,j,k] *V[k][i][j]==1,temp1[i,j,k] *V[k][i][j]==3)==1) ,ctname="constraint4_{}_{}_{}".format(i,j,k)) for i in range(n) for j in range(n) for k in range(t)}
        
        if M>1:
            constraints4 = {(i, j, k): opt_model.add_constraint(
                ct=temp1[i, j, k] * V_L[k][i][j] >= thresholdL*x[i, j, 1, k], ctname="constraint4_{}_{}_{}".format(i, j, k)) for i in range(n) for j in range(n) for k in range(t)}
        
        # nodes cannot talk to itself, this should be obvious with the matrix T but still we add it
        constraints5 = {(i, m, k): opt_model.add_constraint(ct=x[i, i, m, k] == 0, ctname="constraint5_{}_{}_{}".format(
            i, m, k)) for i in range(n) for m in range(M) for k in range(t)}
    
        # if it's recieving it cannot send  in same timestamp versa
        constraints6 = {(i, j, m, m2, f, k): opt_model.add_constraint(ct=x[i, j, m, k]+x[j, f, m2, k] <= 1, ctname="constraint6_{}_{}_{}_{}_{}_{}".format(
            i, j, m, m2, f, k)) for i in range(n) for j in range(n) for m in range(M) for m2 in range(M) for f in range(n) for k in range(t)}
    
        # each node does not surpass it's light energy and radio energy
        constraints7 = {(i): opt_model.add_constraint(ct=opt_model.sum(x[i, j, 0, k]*SendEnergy[0]+ x[j, i, 0, k]*RecieveEnergy[0] for j in range(
            n) for k in range(t)) <= EnergyTotal_RF[i], ctname="constraint6_{}".format(i)) for i in range(n)}
    
        if M>1:
            constraints7_1 = {(i): opt_model.add_constraint(ct=opt_model.sum(x[i, j, 1, k]*SendEnergy[1]+x[j, i, 1, k]*RecieveEnergy[1] for j in range(
            n) for k in range(t)) <= EnergyTotal_L[i], ctname="constraint6_{}".format(i)) for i in range(n)}
    
    
        # each device cannot send the same message more than one time
        Max3=0
        NombreMsg=0
        for i in range(n):
            for j in range(n):
                for nmessage in range(0, len(msgQueues[i][j])):
                    NombreMsg+=1
                    subrange = list(msgQueues[i][j][nmessage])
                    
                    opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(
                        M) for k in subrange) <= 1, ctname="constraint81_{}_{}".format(i, j))
                    
                    opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(
                        M) for k in subrange) >= x_m[i, j, nmessage], ctname="constraint82_{}_{}".format(i, j))
                    
                    opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(
                        M) for k in subrange) == 1, x_m[i, j, nmessage] == 1), ctname="constraint877_{}_{}".format(i, j))
                    
                    opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(
                        M) for k in subrange) == 0, x_m[    i, j, nmessage] == 0), ctname="constraint84_{}_{}".format(i, j))
                    
                    opt_model.add_constraint(ct=opt_model.if_then(x_m[i, j, nmessage] == 1,  opt_model.sum(
                        x[i, j, m, k] for m in range(M) for k in subrange) == 1), ctname="constraint85_{}_{}".format(i, j))
                    
                    opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(M) for k in set([item for sublist in msgQueues[i][j] for item in sublist])) == opt_model.sum(
                        x_m[i, j, nmessage] for nmessage in range(0, len(msgQueues[i][j]))), ctname="constraint83_{}_{}".format(i, j))
    
                    for sub in subrange:
                        opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, sub] for m in range(
                            M)) == 1, delta[i, j, sub] == (sub-subrange[0])+1), ctname="constraint89_{}_{}".format(i, j))
                        opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, sub] for m in range(
                            M)) == 0, delta[i, j, sub] == BiggestMsg+1), ctname="constraint90_{}_{}".format(i, j))
                        Max3=Max3+BiggestMsg+1
    
    
        constraints13 = {(j,m, k): opt_model.add_constraint(ct=opt_model.if_then((opt_model.sum(x[i, j, m, k] for i in range(n)) + opt_model.sum(
            x[j, i, m, k] for i in range(n))) == 1, status[j, k] == m), ctname="constraint13_{}_{}".format(j, m,k)) for j in range(n) for m in range(M) for k in range(t)}
        
  
        #constraints4581 = {(j,m, k): opt_model.add_constraint(ct=opt_model.if_then(status[j, k] == m,(opt_model.sum(x[i, j, m, k] for i in range(n)) + opt_model.sum(
            #x[j, i, m, k] for i in range(n))) == 1), ctname="constraints4581{}_{}".format(j, m,k)) for j in range(n) for m in range(M) for k in range(t)}
        
  
        constraints4582 = {(j,k): opt_model.add_constraint(ct=status[j, k] <= (M-1), ctname="constraints4582_{}_{}".format(j,k)) for j in range(n) for k in range(t)}
        
  
  
        #constraints14 = {(j, k): opt_model.add_constraint(ct=opt_model.if_then((opt_model.sum(x[i, j, 1, k] for i in range(n)) + opt_model.sum(
        #x[j, i, 1, k] for i in range(n))) == 1, status[j, k] == 1), ctname="constraint14_{}_{}".format(j, k)) for j in range(n) for k in range(t)}
        
        constraints15 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(n)) for m in range(M))) == 0, status[j, k-1] == 0) <= (status[j, k] == 0), ctname="constraint15_{}_{}".format(j, k)) for j in range(n) for k in range(1, t)}
        
        constraints55 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(n)) for m in range(M))) == 0, status[j, k-1] == 1) <= (status[j, k] == 1), ctname="constraint99_{}_{}".format(j, k)) for j in range(n) for k in range(1, t)}

        #opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(n)) for m in range(M)) 
        
        
        constraints16 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(n)) for m in range(M))) == 0, status[j, k] == 1) <= (status[j, k-1] == 1), ctname="constraint16_{}_{}".format(j, k)) for j in range(n) for k in range(1, t)}
        constraints17 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(n)) for m in range(M))) == 0, status[j, k] == 0) <= (status[j, k-1] == 0), ctname="constraint17_{}_{}".format(j, k)) for j in range(n) for k in range(1, t)}

    
        # constraints17 = {(i,j, k): opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(M)) == 1, AoI[i,j,k] == 1), ctname="constraint17_{}_{}_{}".format(i,j, k)) for i in range(n) for j in range(n) for k in range(t)}
        # constraints18 = {(i,j, k): opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(M)) == 0, (AoI[i,j,k] - AoI[i,j,k-1])==1), ctname="constraint18_{}_{}_{}".format(i,j, k)) for i in range(n) for j in range(n) for k in range(1,t)}
    
        # constraints19 = {(i,j): opt_model.add_constraint(ct= AoI[i,j,0] == 0, ctname="constraint19_{}".format(i)) for i in range(n) for j in range(n)}
    
    
        # ----------------------
        # constraints11 = { (i,k) : opt_model.add_constraint(ct=opt_model.if_then( opt_model.sum(x[i,j,0,k] for j in range(n)) == 1,delta[i,0,k]==-1 ),ctname="constraint125_{}_{}".format(i,k)) for i in range(n) for k in range(t)}
    
        # constraints12 = { (i,k) : opt_model.add_constraint(ct=opt_model.if_then( opt_model.sum(x[i,j,1,k] for j in range(n)) == 1,delta[i,1,k]==1 ),ctname="constraint154_{}_{}".format(i,k)) for i in range(n) for k in range(t)}
    
        # constraints110 = { (i,k) : opt_model.add_constraint(ct=opt_model.if_then( opt_model.sum(x[i,j,0,k] for j in range(n)) == 0,delta[i,0,k]==0 ),ctname="constraint1250_{}_{}".format(i,k)) for i in range(n) for k in range(t)}
    
        # constraints120 = { (i,k) : opt_model.add_constraint(ct=opt_model.if_then( opt_model.sum(x[i,j,1,k] for j in range(n)) == 0,delta[i,1,k]==0 ),ctname="constraint1540_{}_{}".format(i,k)) for i in range(n) for k in range(t)}
    
        # ----------------------
    
        # if it's recieving in light cannot send in light in same timestamp versa
        # constraints6 = { (i,k) : opt_model.add_constraint(ct= opt_model.sum(x[i,j,1,k]+x[j,f,1,k] for j in range(n) for f in range(n)) <=1 ,ctname="constraint6_{}_{}".format(j,k)) for j in range(n) for k in range(t) }
    
        # we can remove these two to make sending and recieving both ways but different bands
        # we can introduce a scheduling to make reciever only send in case of recieving or have already a message
        # we can introduce a queue that gets filled going from time stamp to time  stamp
    
        # if it's recieving in light cannot send in radio in same timestamp versa
        # constraints7 = { (j,k) : opt_model.add_constraint(ct= opt_model.sum(x[i,j,1,k]+x[j,f,0,k] for i in range(n) for f in range(n)) <=1 ,ctname="constraint7_{}_{}".format(j,k)) for i in range(n) for k in range(t) }
    
        # if it's recieving in radio cannot send in light in same timestamp versa
        # constraints8 = { (j,k) : opt_model.add_constraint(ct= opt_model.sum(x[i,j,0,k]+x[j,f,1,k] for i in range(n) for f in range(n)) <=1 ,ctname="constraint8_{}_{}".format(j,k)) for i in range(n) for k in range(t) }
    
        # recieved 0 and sent 1
        # constraints9 = { (i,j,j1,k,k1) : opt_model.add_constraint(ct= x[i,j,0,k]+ x[j,j1,1,k1] <= switch[j]  ,ctname="constraint9_{}_{}_{}_{}_{}".format(i,j,j1,k,k1))  for i in range(n)  for k in range(0,t) for j1 in range(0,n) for k1 in range(0,t) for j in range(0,n)}
        # recieved 1 and sent 0
        # constraints10 = { (i,j,j1,k,k1) : opt_model.add_constraint(ct= x[i,j,1,k]+ x[j,j1,0,k1] <= switch[j]  ,ctname="constraint10_{}_{}_{}_{}_{}".format(i,j,j1,k,k1))  for i in range(n)  for k in range(0,t) for j1 in range(0,n) for k1 in range(0,t)  for j in range(0,n)}
    
        # recieved 0 and recieved 1
        # constraints11 = { (i,j,i1,k,k1) : opt_model.add_constraint(ct= x[i,j,0,k]+ x[i1,j,1,k1] <= switch[j]  ,ctname="constraint11_{}_{}_{}_{}_{}".format(i,j,i1,k,k1))  for i in range(n)  for k in range(0,t) for i1 in range(0,n) for k1 in range(0,t)  for j in range(0,n)}
        # sent 0 and sent 1
        # constraints12 = { (i,j,j1,k,k1) : opt_model.add_constraint(ct= x[i,j,0,k]+ x[i,j1,1,k1] <= switch[i]  ,ctname="constraint12_{}_{}_{}_{}_{}".format(i,j,j1,k,k1))  for i in range(n)  for k in range(0,t) for j1 in range(0,n) for k1 in range(0,t)  for i in range(0,n) }
    
        # constraints13 = { (j) : opt_model.add_constraint(ct= opt_model.sum(  x[i,j,0,k]* x[j,j1,1,k1] + x[i,j,1,k]*x[j,j1,0,k1] +x[i,j,0,k]* x[j1,j,1,k1]+x[i,j,0,k]* x[i,j1,1,k1]+ x[j,j1,1,k1]* x[i,j,0,k] +x[j,j1,0,k1]* x[i,j,1,k]  for k in range(0,t) for k1 in range(k,t) for j1 in range(0,n) for i in range(0,n) )<= switch[j]  ,ctname="constraint13_{}".format(j))  for j in range(n)  }
    
        # between -1 and 1
        Max1=0
        if M==1:
            for i in range(n):
                for j in range(n): 
                    for k in range(t):
                        if V_RF[k][i][j] >= thresholdRF and T[k][i][j]>=1:
                            Max1=Max1+SendEnergy[0]
                    
        else:
            for i in range(n):
                for j in range(n): 
                    for k in range(t):
                        if V_L[k][i][j] >= thresholdL and T[k][i][j]>=1:
                            Max1=Max1+SendEnergy[1]
                        elif V_RF[k][i][j] >= thresholdRF and T[k][i][j]>=1:
                            Max1=Max1+SendEnergy[0]
                         
        Max2 = sum(1 for j in range(0, n) for k in range(1, t))
        Max3 = sum(BiggestMsg+1 for j in range(0, n) for i in range(0, n) for k in range(0, t))
    
        N_possiblecom = getPossibleCom(V_L, V_RF, msgQueues,thresholdRF,thresholdL,M)
        #Max1=N_possiblecom
    
        if Max1==0 or Max2==0 or Max3==0:
            print('No solution could possibly found!')
        
        
        #Obj1=(opt_model.sum((x[i, j, m, k]*T[k][i][j])/SendEnergy[m] for m in range(M) for i in range(n) for j in range(n) for k in range(t)))/Max1
        Obj1=(opt_model.sum((x[i, j, m, k])*SendEnergy[m] for m in range(M) for i in range(n) for j in range(n) for k in range(t)))/(Max1)

        Obj2=opt_model.sum(opt_model.abs(status[j, k]-status[j, k-1]) for j in range(0, n) for k in range(1, t))/Max2
        Obj3=opt_model.sum(delta[i, j, k] for i in range(n) for j in range(n) for k in range(0, t))/Max3
        
        
    
        # w1=[i/10 for  i in range(0,11,1) ]
        # w2=[i/10 for  i in range(0,11,1) ]
        # print('Starting Building dataset')
        # OMEGA=[(i,j) for i in w1 for j in w2 if i+j<=1]

        
        # Prepare to collect results for different weights
        
        # Define weights
        w1=[i/100 for  i in range(0,110,1) ]
        w2=[i/100 for  i in range(0,110,1) ]
        OMEGA=[(i,j) for i in w1 for j in w2 if i+j==0.9]
        W1=[i[0] for i in OMEGA]
        for weight1 in W1:
            weight2=1-weight1
            # Clear previous objective and add the new weighted objective
            opt_model.remove_objective()
            objective = 0.1*Obj1 + weight1*Obj2+weight2*Obj3

            opt_model.minimize(objective)
            #opt_model.set_multi_objective('min', [Obj1,Obj3]
                   #, weights=[weight1,weight2])
            # opt_model.parameters.mip.tolerances.mipgap.set(0.02)
            # opt_model.parameters.timelimit=20
            # opt_model.set_time_limit(20)
            # Solve the model
            solution = opt_model.solve(log_output=False)
            
        
            MaxDelay = 0
            Delay = 0
            Nswitch = 0
            Nsatisfied = 0
            
            
            
            #-----------------------------------------------------
            Binaryx = getSolution(x)

                  
            indices = []
            Scheduling = []
            for i in range(n):
                for j in range(n):
                    for m in range(M):
                        for k in range(t):
                            indices.append([i, j, m, k])
            
            for index, value in enumerate(indices):
                if Binaryx[index] > 0.5:
                    Nsatisfied = Nsatisfied+T[value[3]][value[0]][value[1]]
                    if M==1 and value[2]>0:
                        raise('ERROR, SWITCHED WITH RF ONLY, DEBUG...')
                    Scheduling.append(value)
            
            if len(Scheduling) ==0:              
                print('No device has been selected--',weight1)
                # if all_pareto_fronts1[weight1,weight2]==0:
                #     all_pareto_fronts1[weight1,weight2]=0
                # if all_pareto_fronts2[weight1,weight2]==0:
                #     all_pareto_fronts2[weight1,weight2]=1
                   
                break 
            
            
            consumedE_RF=0
            consumedE_Hybrid=0
            if M>1:
                for index, value in enumerate(indices):
                    if Binaryx[index] > 0.5:
                        if value[2]>0.5:
                            consumedE_Hybrid=consumedE_Hybrid+SendEnergy[1]
                        else:
                            consumedE_Hybrid=consumedE_Hybrid+SendEnergy[0]
            
            else:
                for index, value in enumerate(indices):
                    if Binaryx[index] > 0.5:
                        consumedE_RF=consumedE_RF+SendEnergy[0]
            
            
            ObjectiveValueG = opt_model.solution.get_objective_value()
         
                
                
                
            Nsatisfied = Nsatisfied/N_possiblecom
            #-----------------------------------------------------
            
            Binaryx_m = getSolution(x_m)
            indices = []
            choices = []
            for i in range(n):
                for j in range(n):
                    for nmessage in range(0, len(msgQueues[i][j])):
                        indices.append([i, j, nmessage])
            
            for index, value in enumerate(indices):
                if Binaryx_m[index] > 0.5:
                    choices.append(value)
            
            Scheduling.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
            
            if len(Scheduling) != len(choices):
                # continue
                print(Scheduling)
                print(choices)
                print(msgQueues)
                raise Exception("Dimension not matching for some reason.")
            
            rk=0
            for st in range(0, len(Scheduling)):
                i, j, m, k = Scheduling[st]
                i, j, f = choices[st]
                try:
                    Delay = Delay +  np.argwhere(np.array(msgQueues[i][j][f]) == k)[0][0]
                    MaxDelay = MaxDelay+len(np.array(msgQueues[i][j][f]))-1
                    rk=rk+1
                except IndexError:
                    print('error')
                    
            print(Delay,NombreMsg,rk,weight1)
            if Delay==0:
                ZeroDelay[weight1]+=20
            else:
                ZeroDelay[weight1]+=1
            Delay=Delay+(NombreMsg-rk)*(BiggestMsg+1)
            Delay = Delay/((NombreMsg)*(BiggestMsg+1))
            
            
            statusSol = getSolution(status)
            
            indices = []
            switching = defaultdict(list)
            nodeswitch = defaultdict(lambda: 0)
            
            for i in range(n):
                for k in range(t):
                    indices.append([i, k])
            
            for index, value in enumerate(indices):
                if statusSol[index]>0.9:
                    switching[value[0], value[1]] = 1
                elif statusSol[index]<0.1:
                    switching[value[0], value[1]] = 0
                    
            for i in range(0, n):
                for k in range(1, t):
                    if switching[i, k] != switching[i, k-1]:
                        #print(i,k,switching[i, k],switching[i, k-1])
                        Nswitch = Nswitch+1
                        if i in nodeswitch:
                            nodeswitch[i] += 1
                        else:
                            nodeswitch[i] = 1
                if nodeswitch[i] == 0:
                    continue
            
            Nswitch = Nswitch/((t-1)*n)
            
            if M==1:
                ConsumE=consumedE_RF/(Max1) 
            else:
                ConsumE=consumedE_Hybrid/(Max1) 
            if M==1 and Nswitch>0:
                print(Nswitch)
                print(Scheduling)
                raise('ERROR, SWITCHED WITH RF ONLY, DEBUG...')
            if M>1 and Nswitch>0:
                print('-----WE HAVE A SWITCH')
            
            
            
            
            
            if all_pareto_fronts1[weight1,weight2]!=0:
                all_pareto_fronts1[weight1,weight2]=(all_pareto_fronts1[weight1,weight2]+ConsumE)/2
            else:
                all_pareto_fronts1[weight1,weight2]=ConsumE
            if all_pareto_fronts2[weight1,weight2]!=0:
                all_pareto_fronts2[weight1,weight2]=(all_pareto_fronts2[weight1,weight2]+Delay)/2
            else:
                all_pareto_fronts2[weight1,weight2]=Delay

        






# =============================================================================
#         for (omega1,omega2) in [(0,0),(0,1)]:
#             objective = omega1*Obj1 + omega2* Obj2 + (1-omega1-omega2)*Obj3
#             
#             opt_model.maximize(objective)
#             #opt_model.set_multi_objective('max', [Obj1,Obj2,Obj3], [1,1,1], [omega1,omega2,(1-omega1-omega2)], abstols=None, reltols=None, names=None)
#       
#             opt_model.parameters.mip.tolerances.mipgap.set(0.02)
#             start = time.time()
#             opt_model.parameters.timelimit=20
#             opt_model.set_time_limit(20)
#             MIP_solution = opt_model.solve()
#             end = time.time()
#         
#             MaxDelay = 0
#             Delay = 0
#             Nswitch = 0
#             Nsatisfied = 0
#             
#         
#         
#             #-----------------------------------------------------
#             try:
#                 Binaryx = getSolution(x)
#             except:
#                  continue
#                   
#             indices = []
#             Scheduling = []
#             for i in range(n):
#                 for j in range(n):
#                     for m in range(M):
#                         for k in range(t):
#                             indices.append([i, j, m, k])
#         
#             for index, value in enumerate(indices):
#                 if Binaryx[index] > 0.5:
#                     Nsatisfied = Nsatisfied+T[value[3]][value[0]][value[1]]
#                     Scheduling.append(value)
#             
#             
#             consumedE_RF=0
#             consumedE_Hybrid=0
#             if M>1:
#                 for index, value in enumerate(indices):
#                     if Binaryx[index] > 0.5:
#                         if value[2]>0.5:
#                             consumedE_Hybrid=consumedE_Hybrid+SendEnergy[1]+RecieveEnergy[1]
#                         else:
#                             consumedE_Hybrid=consumedE_Hybrid+SendEnergy[0]+RecieveEnergy[0]
#     
#             else:
#                 for index, value in enumerate(indices):
#                     if Binaryx[index] > 0.5:
#                         consumedE_RF=consumedE_RF+SendEnergy[0]+RecieveEnergy[0]
#     
# 
#             ObjectiveValueG = opt_model.solution.get_objective_value()
#             if len(Scheduling) ==0:              
#                 print('No device has been selected')
#                 continue 
#                 
#                 
#                 
#             Nsatisfied = Nsatisfied/N_possiblecom
#             #-----------------------------------------------------
#             
#             Binaryx_m = getSolution(x_m)
#             indices = []
#             choices = []
#             for i in range(n):
#                 for j in range(n):
#                     for nmessage in range(0, len(msgQueues[i][j])):
#                         indices.append([i, j, nmessage])
#         
#             for index, value in enumerate(indices):
#                 if Binaryx_m[index] > 0.5:
#                     choices.append(value)
#         
#             Scheduling.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
#         
#             if len(Scheduling) != len(choices):
#                 # continue
#                 print(Scheduling)
#                 print(choices)
#                 print(msgQueues)
#                 raise Exception("Dimension not matching for some reason.")
#         
#             for st in range(0, len(Scheduling)):
#                 i, j, m, k = Scheduling[st]
#                 i, j, f = choices[st]
#                 try:
#                     Delay = Delay +  np.argwhere(np.array(msgQueues[i][j][f]) == k)[0][0]
#                     MaxDelay = MaxDelay+len(np.array(msgQueues[i][j][f]))-1
#                 except IndexError:
#                     print('error')
#             Delay = Delay/MaxDelay
#         
#         
#             statusSol = getSolution(status)
#         
#             indices = []
#             switching = defaultdict(list)
#             nodeswitch = defaultdict(lambda: 0)
#         
#             for i in range(n):
#                 for k in range(t):
#                     indices.append([i, k])
#         
#             for index, value in enumerate(indices):
#                 if statusSol[index]>0.9:
#                     switching[value[0], value[1]] = 1
#                 elif statusSol[index]<0.1:
#                     switching[value[0], value[1]] = 0
#                     
#             for i in range(0, n):
#                 for k in range(1, t):
#                     if switching[i, k] != switching[i, k-1]:
#                         #print(i,k,switching[i, k],switching[i, k-1])
#                         Nswitch = Nswitch+1
#                         if i in nodeswitch:
#                             nodeswitch[i] += 1
#                         else:
#                             nodeswitch[i] = 1
#                 if nodeswitch[i] == 0:
#                     continue
#         
#             Nswitch = Nswitch/((t-1)*n)
#             
#            
#             if M==1 and Nswitch>0:
#                 print(Nswitch)
#                 print(Scheduling)
#                 raise('ERROR, SWITCHED WITH RF ONLY, DEBUG...')
#             
# 
#             if omega1==0 and omega2==1:
#                 MinSwitch=Nswitch
#             elif omega1==0 and omega2==0:
#                 MinDelay=Delay
#         
#         MinEnergy=0
#         print('we got:',MinSwitch,MinDelay,MinEnergy )
#             #-----------------------------------------------------
#             
#         print('Starting testing')
# ########################Re-run AgAIN#################
# 
#         threshold=0.08
#         for (omega1,omega2) in OMEGA:
# 
#             objective = omega1*Obj1 + omega2* Obj2 + (1-omega1-omega2)*Obj3
#             opt_model.maximize(objective)
#             #opt_model.set_multi_objective('max', [Obj1,Obj2,Obj3], [1,1,1], [omega1,omega2,(1-omega1-omega2)], abstols=None, reltols=None, names=None)
#       
#             opt_model.parameters.mip.tolerances.mipgap.set(0.02)
#             opt_model.parameters.timelimit=20
#             opt_model.set_time_limit(20)
#             opt_model.solve()
#         
#             MaxDelay = 0
#             Delay = 0
#             Nswitch = 0
#             Nsatisfied = 0
#             
#         
#         
#             #-----------------------------------------------------
#             try:
#                 Binaryx = getSolution(x)
#             except:
#                  continue
#                   
#             indices = []
#             Scheduling = []
#             for i in range(n):
#                 for j in range(n):
#                     for m in range(M):
#                         for k in range(t):
#                             indices.append([i, j, m, k])
#         
#             for index, value in enumerate(indices):
#                 if Binaryx[index] > 0.5:
#                     Nsatisfied = Nsatisfied+T[value[3]][value[0]][value[1]]
#                     if M==1 and value[2]>0:
#                         raise('ERROR, SWITCHED WITH RF ONLY, DEBUG...')
#                     Scheduling.append(value)
#             
#             
#             consumedE_RF=0
#             consumedE_Hybrid=0
#             if M>1:
#                 for index, value in enumerate(indices):
#                     if Binaryx[index] > 0.5:
#                         if value[2]>0.5:
#                             consumedE_Hybrid=consumedE_Hybrid+SendEnergy[1]+RecieveEnergy[1]
#                         else:
#                             consumedE_Hybrid=consumedE_Hybrid+SendEnergy[0]+RecieveEnergy[0]
#     
#             else:
#                 for index, value in enumerate(indices):
#                     if Binaryx[index] > 0.5:
#                         consumedE_RF=consumedE_RF+SendEnergy[0]+RecieveEnergy[0]
#     
# 
#             ObjectiveValueG = opt_model.solution.get_objective_value()
#             if len(Scheduling) ==0:              
#                 print('No device has been selected')
#                 continue 
#                 
#                 
#                 
#             Nsatisfied = Nsatisfied/N_possiblecom
#             #-----------------------------------------------------
#             
#             Binaryx_m = getSolution(x_m)
#             indices = []
#             choices = []
#             for i in range(n):
#                 for j in range(n):
#                     for nmessage in range(0, len(msgQueues[i][j])):
#                         indices.append([i, j, nmessage])
#         
#             for index, value in enumerate(indices):
#                 if Binaryx_m[index] > 0.5:
#                     choices.append(value)
#         
#             Scheduling.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
#         
#             if len(Scheduling) != len(choices):
#                 # continue
#                 print(Scheduling)
#                 print(choices)
#                 print(msgQueues)
#                 raise Exception("Dimension not matching for some reason.")
#         
#             for st in range(0, len(Scheduling)):
#                 i, j, m, k = Scheduling[st]
#                 i, j, f = choices[st]
#                 try:
#                     Delay = Delay +  np.argwhere(np.array(msgQueues[i][j][f]) == k)[0][0]
#                     MaxDelay = MaxDelay+len(np.array(msgQueues[i][j][f]))-1
#                 except IndexError:
#                     print('error')
#             Delay = Delay/MaxDelay
#         
#         
#             statusSol = getSolution(status)
#         
#             indices = []
#             switching = defaultdict(list)
#             nodeswitch = defaultdict(lambda: 0)
#         
#             for i in range(n):
#                 for k in range(t):
#                     indices.append([i, k])
#         
#             for index, value in enumerate(indices):
#                 if statusSol[index]>0.9:
#                     switching[value[0], value[1]] = 1
#                 elif statusSol[index]<0.1:
#                     switching[value[0], value[1]] = 0
#                     
#             for i in range(0, n):
#                 for k in range(1, t):
#                     if switching[i, k] != switching[i, k-1]:
#                         #print(i,k,switching[i, k],switching[i, k-1])
#                         Nswitch = Nswitch+1
#                         if i in nodeswitch:
#                             nodeswitch[i] += 1
#                         else:
#                             nodeswitch[i] = 1
#                 if nodeswitch[i] == 0:
#                     continue
#         
#             Nswitch = Nswitch/((t-1)*n)
#             
#             if M==1:
#                 ConsumE=consumedE_RF/(Max1*Max1) 
#             else:
#                 ConsumE=consumedE_Hybrid/(Max1*Max1) 
#             if M==1 and Nswitch>0:
#                 print(Nswitch)
#                 print(Scheduling)
#                 raise('ERROR, SWITCHED WITH RF ONLY, DEBUG...')
#             if M>1 and Nswitch>0:
#                 print('-----WE HAVE A SWITCH')
#             #if omega1!=1 and omega2!=1:  
#                    
#             alpha1=1 if MinSwitch==0 else MinSwitch
#             alpha3=1 if MinEnergy==0 else MinEnergy
#             
#             print(abs(MinSwitch-Nswitch),alpha1*threshold)
#             print(abs(MinDelay-Delay),MinDelay*threshold)
#             print(abs(MinEnergy-ConsumE),alpha3*threshold)
# 
#             
#             
#          
# 
# 
#             #print(abs(MaxSatisfied-Nsatisfied),MaxSatisfied*threshold)
#             if abs(MinSwitch-Nswitch) <=alpha1*threshold:
#                 weight_scores[(omega1, omega2, 1-omega1-omega2)] += 1
#             if abs(MinDelay-Delay) <=MinDelay*threshold:
#                 weight_scores[(omega1, omega2, 1-omega1-omega2)] += 1
#             if abs(MinEnergy-ConsumE) <=alpha3*threshold:
#                 weight_scores[(omega1, omega2, 1-omega1-omega2)] += 1
#                 
# 
#                         
#                     
#                 
# 
#         #print(weight_scores)
#         # Find the weight combination with the highest score
#         print(weight_scores)
# 
#         
#         best_weights = max(weight_scores, key=weight_scores.get)
#         best_score = weight_scores[best_weights]
# 
#         print(best_weights)
#         
# =============================================================================
        
        
        # # Pareto front function: More accurate method
        # def pareto_front(points):
        #     # Initialize a mask to identify Pareto optimal points
        #     is_pareto = np.ones(points.shape[0], dtype=bool)
        #     for i, point in enumerate(points):
        #         # Keep points that are not dominated by any other points
        #         is_pareto &= np.any(points<point, axis=1) | np.all(points<=point, axis=1)
        #     return is_pareto

        # # Define weight combinations

        
        # # Plotting setup
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Compute and plot Pareto fronts for each weight combination

        # points = np.vstack((f1, f2, f3)).T
        # pareto_indices = pareto_front(points)
        # ax.scatter(points[pareto_indices, 0], points[pareto_indices, 1], points[pareto_indices, 2])

        # ax.set_xlabel('$f_1(x, y)$')
        # ax.set_ylabel('$f_2(x, y)$')
        # ax.set_zlabel('$f_3(x, y)$')
        # ax.set_title('Pareto Fronts for Different Weight Combinations')
        # ax.legend(title="Weights", bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()




        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        
        # # Plotting each function with a different color
        # ax.scatter(w1, w2, f1, color='r', label='f1')
        # ax.scatter(w1, w2, f2, color='g', label='f2')
        # ax.scatter(w1, w2, f3, color='b', label='f3')
        
        # # Adding labels and title
        # ax.set_xlabel('Weight w1')
        # ax.set_ylabel('Weight w2')
        # ax.set_zlabel('Function Values')
        # ax.set_title('3D Scatter Plot of Function Values Against Weights')
        # ax.legend()
        # plt.show()
            
            # Show the plot
        
        

        
        
        
        return all_pareto_fronts1,all_pareto_fronts2,ZeroDelay