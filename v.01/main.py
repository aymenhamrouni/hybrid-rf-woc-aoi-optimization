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
from tqdm.contrib.telegram import tqdm, trange
#from tqdm import tqdm

import itertools

marker = itertools.cycle((',', '+', '.', 'o', '*')) 




plt.close('all')

plt.show(block = False)

Nmontecarlo = 1
#size=2 #5 km is the size

# define the number of cores (this is how many processes wil run)
num_cores = multiprocessing.cpu_count()
# execute the function in parallel - `return_list` is a list of the results of the function
# in this case it will just be a list of None's

def run_MIP_simulation(OMEGA, n, t,nAPs):

    plotswitch_RF = []
    plotsatisfied_RF = []
    plotDelay_RF = []
    plotRunningtime_RF = []
    ObjectiveFunction_RF = []
    
    plotswitch_Hybrid = []
    plotsatisfied_Hybrid = []
    plotDelay_Hybrid = []
    plotRunningtime_Hybrid = []
    ObjectiveFunction_Hybrid = []
    
    PAoI_RF=[]
    MAoI_RF=[]
    PAoI_Hybrid=[]
    MAoI_Hybrid=[]
    MGAP_RF=[]
    MGAP_Hybrid=[]
    global size
    global SendEnergy
    global RecieveEnergy
    W1,W2=np.meshgrid(OMEGA,OMEGA)
    WeightSwitch=np.full((W1.shape[0],W1.shape[1]),-1)
    WeightDelay=np.full((W1.shape[0],W1.shape[1]),-1)
    WeightSatisfied=np.full((W1.shape[0],W1.shape[1]),-1)
    ObjF=np.full((W1.shape[0],W1.shape[1]),-1)
    PAoIRF=np.full((W1.shape[0],W1.shape[1]),-1)
    MAoIRF=np.full((W1.shape[0],W1.shape[1]),-1)
    # WeightSwitch=np.zeros((int(max(W1[0])*100),int(max(W1[1])*100)))
    # WeightDelay=np.zeros((int(max(W1[0])*100),int(max(W1[1])*100)))
    # WeightSatisfied=np.zeros((int(max(W1[0])*100),int(max(W1[1])*100)))
    # WeightEnergy=np.zeros((int(max(W1[0])*100),int(max(W1[1])*100)))



    
    #coupleOmega=[(i,j) for i in OMEGA for j in OMEGA if i+j<=1]
    coupleOmega=[(OMEGA[0],OMEGA[1])]
    for omega1,omega2 in coupleOmega:
        Totalsatisfied_Hybrid = []
        Totalswitch_Hybrid = []
        TotalDelay_Hybrid = []
        Runningtime_Hybrid = []
        ObjectiveF_Hybrid = []
        
        Totalsatisfied_RF = []
        Totalswitch_RF = []
        TotalDelay_RF = []
        Runningtime_RF = []
        ObjectiveF_RF = []
        PeakAoI_RF=[]
        MeanAoI_RF=[]
        PeakAoI_Hybrid=[]
        MeanAoI_Hybrid=[]
        GAP_RF=[]
        GAP_Hybrid=[]
        
        MeanE_RF=[]
        MeanE_Hybrid=[]
        E_RF=[]
        E_Hybrid=[]
        
        
        for i in trange(Nmontecarlo, token='6972663208:AAGa9eWpdM3sG5hLONyYFolo04Sb4D41gUg', chat_id='6729112338'):
        #for i in tqdm(range(Nmontecarlo)):

            simulation = data_generation.NetworkCharacteristics(n, t,nAPs)
            T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp = simulation.generate_messages(messageSize=t-10,
                mini=1,maxi=4,appSize=2)
            #PriorityLevels = [0.2, 0.5, 1]
            #Pt = simulation.generate_priorities(msgQueues, PriorityLevels)

            V_RF = simulation.generate_visibility_RF(85, 100)
            V_L = simulation.generate_visibility_VLC(90, 100)

            
            # introduce freshness, data needs to be sent asap as it has ending and begin date
            # the ending represent where the data is no longer fresh and we would want to maxmimize freshness
            # later on this MessageBegin and MessageEnd we can get it from a preprocessing model

            # multi-objective function
            # generate energy for coin cell batteries such as wearables and sensors
            EnergyTotal_L = simulation.generate_energy(500, 900)
            EnergyTotal_RF= simulation.generate_energy(400, 600)
            
            #NodesLoc=initialize_nodes(n,size)
            #x, y = zip(*NodesLoc)
            #plt.figure()
            #plt.scatter(x, y)
            #plt.show()
            #Distances=calculate_distances(NodesLoc)
            


        
            MILP=model(1,n,t,T,V_RF,V_L,EnergyTotal_L,EnergyTotal_RF,msgQueues,BiggestMsg, omega1, omega2)

            opt_model ,x,x_m,delta,status,N_possiblecom= MILP.buildMILP(relaxer=False)




        
            Nsatisfied,Nswitch,Delay,end,start,ObjectiveValueG,peak_aoi, mean_aoi,gap,consumedE_RF,consumedE_Hybrid=MILP.solve_MILP(opt_model,msgApp,N_possiblecom,
                                                                          x,x_m,delta,status)
            
            if Nsatisfied!=0:

                Totalsatisfied_RF.append(Nsatisfied)
                Totalswitch_RF.append(Nswitch)
                TotalDelay_RF.append(Delay)
                Runningtime_RF.append(end-start)
                ObjectiveF_RF.append(np.mean(ObjectiveValueG))
                PeakAoI_RF.append(peak_aoi)
                MeanAoI_RF.append(mean_aoi)
                GAP_RF.append(gap)
                MeanE_RF.append(consumedE_RF)
                MeanE_Hybrid.append(consumedE_Hybrid)

            else:
                print('No device was assigned')

                
            MILP=model(2,n,t,T,V_RF,V_L,EnergyTotal_L,EnergyTotal_RF,msgQueues,BiggestMsg, omega1, omega2)
            opt_model ,x,x_m,delta,status,N_possiblecom= MILP.buildMILP(relaxer=False)
            

            Nsatisfied,Nswitch,Delay,end,start,ObjectiveValueG,peak_aoi, mean_aoi,gap,consumedE_RF,consumedE_Hybrid=MILP.solve_MILP(opt_model,msgApp,N_possiblecom,
                                                                            x,x_m,delta,status)
            if Nsatisfied!=0:
                Totalsatisfied_Hybrid.append(Nsatisfied)
    
                Totalswitch_Hybrid.append(Nswitch)
                TotalDelay_Hybrid.append(Delay)
                Runningtime_Hybrid.append(end-start)
                ObjectiveF_Hybrid.append(np.mean(ObjectiveValueG))
                PeakAoI_Hybrid.append(peak_aoi)
                MeanAoI_Hybrid.append(mean_aoi)
                GAP_Hybrid.append(gap)
                MeanE_RF.append(consumedE_RF)
                MeanE_Hybrid.append(consumedE_Hybrid)
            else:
                print('No device was assigned')


            
          
            

        #print('We have run the following number of simulations: ', montecarlo)
        print('Current weights is: Transmission ',omega1,' Switching ',omega2,' and delay ', 1-omega1-omega2)
        print('Switched in Hybrid Mode %', np.mean(Totalswitch_Hybrid))
        print('Satisfied in Hybrid Mode %', np.mean(Totalsatisfied_Hybrid))
        print('Delay avg in Hybrid Mode %', np.mean(TotalDelay_Hybrid))
        print('Running time s in Hybrid Mode', np.mean(Runningtime_Hybrid))

        print('-------------------')
        print('Switched in RF Mode %', np.mean(Totalswitch_RF))
        print('Satisfied in RF Mode %', np.mean(Totalsatisfied_RF))
        print('Delay avg in RF Mode %', np.mean(TotalDelay_RF))
        print('Running time s in RF Mode', np.mean(Runningtime_RF))
        
        plotswitch_Hybrid.append(np.mean(Totalswitch_Hybrid))
        plotsatisfied_Hybrid.append(np.mean(Totalsatisfied_Hybrid))
        plotDelay_Hybrid.append(np.mean(TotalDelay_Hybrid))
        plotRunningtime_Hybrid.append(np.mean(Runningtime_Hybrid))
        ObjectiveFunction_Hybrid.append(np.mean(ObjectiveF_Hybrid))
        PAoI_Hybrid.append(np.mean(PeakAoI_Hybrid))
        MAoI_Hybrid.append(np.mean(MeanAoI_Hybrid))
        MGAP_RF.append(np.mean(GAP_RF))
                
        plotswitch_RF.append(np.mean(Totalswitch_RF))
        plotsatisfied_RF.append(np.mean(Totalsatisfied_RF))
        plotDelay_RF.append(np.mean(TotalDelay_RF))
        plotRunningtime_RF.append(np.mean(Runningtime_RF))
        ObjectiveFunction_RF.append(np.mean(ObjectiveF_RF))
        PAoI_RF.append(np.mean(PeakAoI_RF))
        MAoI_RF.append(np.mean(MeanAoI_RF))
        MGAP_Hybrid.append(np.mean(GAP_Hybrid))
        
        E_RF.append(np.mean(MeanE_RF))
        E_Hybrid.append(np.mean(MeanE_Hybrid))

        
        
        # WeightDelay[int(omega1*10)-1][int(omega2*10)-1]=np.mean(TotalDelay_RF)
        # WeightSatisfied[int(omega1*10)-1][int(omega2*10)-1]=np.mean(Totalsatisfied_RF)
        # WeightSwitch[int(omega1*10)-1][int(omega2*10)-1]=np.mean(Totalswitch_RF)
        # ObjF[int(omega1*10)-1][int(omega2*10)-1]=np.mean(ObjectiveF_RF)
        # PAoIRF[int(omega1*10)-1][int(omega2*10)-1]=np.mean(PeakAoI_RF)
        # MAoIRF[int(omega1*10)-1][int(omega2*10)-1]=np.mean(MeanAoI_RF)


    # plotWeightsFig(W1,W2,WeightDelay,'delay')
    # plotWeightsFig(W1,W2,WeightSatisfied,'satisfied')
    # plotWeightsFig(W1,W2,WeightSwitch,'switch')
    # plotWeightsFig(W1,W2,ObjF,'ObjectiveFunction')
    # plotWeightsFig(W1,W2,PAoIRF,'Peak AoI')
    # plotWeightsFig(W1,W2,MAoIRF,'Mean AoI')


    return plotswitch_RF, plotsatisfied_RF, plotDelay_RF, plotRunningtime_RF, ObjectiveFunction_RF,plotswitch_Hybrid, plotsatisfied_Hybrid, plotDelay_Hybrid, plotRunningtime_Hybrid, ObjectiveFunction_Hybrid,PAoI_RF,MAoI_RF,PAoI_Hybrid,MAoI_Hybrid,MGAP_RF,MGAP_Hybrid,E_RF,E_Hybrid


def plotWeightsFig(W1,W2,F,title):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    # Use the plot_surface function to create the 3D plot
    surf = ax.plot_surface(W1, W2, F, cmap='viridis', edgecolor='none')
    
    # Labels and title
    ax.set_xlabel('Weight 1 (w1)')
    ax.set_ylabel('Weight 2 (w2)')
    ax.set_title(str(title))

    # Colorbar to show the objective function value
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()




def plotAoI(d,name):
    
    #plt.bar(range(len(d)), list(d.values()), align='center')
    #plt.xticks(range(len(d)), list(d.keys()))
    lists = sorted(d.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists)
    plt.xlabel("Number of IoT devices")
    
    ax = plt.subplot()
    ax.plot(x, y,marker = next(marker),markersize=9,label=name)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))











#def main():

OMEGA =[0.1,0.1] # heuresticly decided by figure
TotalN = range(5, 10)
AoIRFP=dict()
AoIRFM=dict()
AoIHybridP=dict()
AoIHybridM=dict()
SRF=dict()
SHybrid=dict()
RRF=dict()
RHybrid=dict()
GAPR=dict()
GAPH=dict()
ERF=dict()
EHYBRID=dict()

switchRF=dict()
switchHybrid=dict()


for i in TotalN:
    print(i)
    plotswitch_RF, plotsatisfied_RF, plotDelay_RF, plotRunningtime_RF, ObjectiveFunction_RF,       plotswitch_Hybrid, plotsatisfied_Hybrid, plotDelay_Hybrid, plotRunningtime_Hybrid, ObjectiveFunction_Hybrid,PAoI_RF,MAoI_RF,PAoI_Hybrid,MAoI_Hybrid ,MGAP_RF,MGAP_Hybrid,E_RF,E_Hybrid= run_MIP_simulation(
    OMEGA, i, t=50,nAPs=3)
    AoIRFP[i]=PAoI_RF[0]
    AoIRFM[i]=MAoI_RF[0]
    AoIHybridP[i]=PAoI_Hybrid[0]
    AoIHybridM[i]=MAoI_Hybrid[0]
    SRF[i]=plotsatisfied_RF
    SHybrid[i]=plotsatisfied_Hybrid
    RRF[i]=plotRunningtime_RF
    RHybrid[i]=plotRunningtime_Hybrid
    GAPR[i]=MGAP_RF
    GAPH[i]=MGAP_Hybrid
    ERF[i]=E_RF
    EHYBRID[i]=E_Hybrid
    switchRF[i]=plotswitch_RF
    switchHybrid[i]=plotswitch_Hybrid
    
    
    

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(7, 7))
plotAoI(AoIRFP,'P-AoI for RF')
plotAoI(AoIHybridP  ,'P-AoI for RF-OC')
plotAoI(AoIRFM,'M-AoI for RF')
plotAoI(AoIHybridM,'M-AoI for RF-OC')
plt.ylabel('Age of Information')
plt.legend(loc="upper left",prop={'size':18})
plt.grid()
plt.savefig('AoI_Analysis.png', bbox_inches='tight')

plt.figure(figsize=(7, 7))
plotAoI(SRF,'RF')
plotAoI(SHybrid,'RF-OC')
plt.ylabel('Transmissions Rate')
plt.legend(loc="upper right")
plt.grid()
plt.savefig('Satisfied_Analysis.png', bbox_inches='tight')


plt.figure(figsize=(7, 7))
plotAoI(RRF,'RFC')
plotAoI(RHybrid,'VLC-RFC')
plt.ylabel('Running Time (s)')
plt.legend(loc="upper left")
plt.grid()
plt.savefig('Complexity_Analysis.png', bbox_inches='tight')

plt.figure(figsize=(7, 7))
plotAoI(GAPR,'RFC')
plotAoI(GAPH,'VLC-RFC')
plt.ylabel('Duality GAP (%)')
plt.legend(loc="upper left")
plt.grid()
plt.savefig('GAP_Analysis.png', bbox_inches='tight')



plt.figure(figsize=(7, 7))
plotAoI(ERF,'RF')
plotAoI(EHYBRID,'RF-OC')
plt.ylabel('Energy Consumption Rate')
plt.legend(loc="upper left",prop={'size':20})
plt.grid()
plt.savefig('Energy_Analysis.png', bbox_inches='tight')




plt.figure(figsize=(7, 7))

plotAoI(switchRF,'RF-only')
plotAoI(switchHybrid,'Hybrid')
plt.ylabel('Technology Switching Rate')
plt.legend(loc="upper left")
plt.grid()
plt.savefig('Switching_Analysis.png', bbox_inches='tight')


    

    #plotswitch, plotsatisfied, plotDelay, plotRunningtime, ObjectiveFunction = run_MIP_simulation(
     #   OMEGA, i, t=40)
    
    # plt.plot(OMEGA, np.array(plotsatisfied)-np.array(plotswitch))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Diff Satsfied - Switch for nodes {}".format(i))
    # plt.show()

    # plt.plot(OMEGA, np.array(plotswitch))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Number Switch for nodes {}".format(i))

    # plt.show()
    # plt.plot(OMEGA, np.array(plotsatisfied))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Number Satisfied for nodes {}".format(i))
    # plt.show()

    # plt.plot(OMEGA, np.array(plotDelay))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Total Delay for nodes {}".format(i))
    # plt.show()

    # plt.plot(OMEGA, np.array(plotRunningtime))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Total Running Time for nodes {}".format(i))
    # plt.show()

    # plt.plot(OMEGA, np.array(ObjectiveFunction))
    # plt.xlabel("Omega Values")
    # plt.ylabel("Total Objective function for nodes {}".format(i))
    # plt.show()

    # Tplotswitch = []
    # Tplotsatisfied = []
    # TplotDelay = []
    # TplotRunningtime = []
    # TObjectiveFunction = []
    # # OMEGA=[0.7] #best omega is 0.1
    # TotalN = range(3, 10)
    # # totalT=range(5,20)

    # for i in TotalN:
    #     plotswitch, plotsatisfied, plotDelay, plotRunningtime, ObjectiveFunction = run_MIP_simulation([
    #                                                                                                   0.7,0.7], i, t=20, M=2)

    #     Tplotswitch.append(plotswitch)
    #     Tplotsatisfied.append(plotsatisfied)
    #     TplotDelay.append(plotDelay)
    #     TplotRunningtime.append(plotRunningtime)
    #     TObjectiveFunction.append(ObjectiveFunction)

    # plt.plot(TotalN, np.array(Tplotsatisfied)-np.array(Tplotswitch))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Diff Satsfied - Switch")
    # plt.show()

    # plt.plot(TotalN, np.array(Tplotswitch))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Number Switch")

    # plt.show()
    # plt.plot(TotalN, np.array(Tplotsatisfied))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Number Satisfied")
    # plt.show()

    # plt.plot(TotalN, np.array(TplotDelay))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Total Delay")
    # plt.show()

    # plt.plot(TotalN, np.array(TplotRunningtime))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Total Running Time")
    # plt.show()

    # plt.plot(TotalN, np.array(TObjectiveFunction))
    # plt.xlabel("Nodes number")
    # plt.ylabel("Total Objective function")
    # plt.show()


# if __name__ == "__main__":
#     main()
