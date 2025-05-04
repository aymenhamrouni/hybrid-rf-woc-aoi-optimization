# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:02:18 2024

@author: Aymen Hamrouni
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
import itertools

marker = itertools.cycle((',', '+', '.', 'o', '*')) 


def AoI(t,selection,intervals,typ='exp'):
    
    # Parameters
    # Determine the simulation parameters
    input_data=[]
    for i in range(0,len(selection)):
        input_data.append([selection[i][0],selection[i][1],selection[i][2],selection[i][2]+intervals[i]])
        
    
    
    # Identify unique sender-receiver pairs
    pairs = set((sender, receiver) for sender, receiver, _, _ in input_data)
    
    # Initialize simulation parameters
    total_time = max(item[3] for item in input_data) + 1
    total_time = t

    time_step = 0.01
    time_vector = np.arange(0, total_time, time_step)
    
    # Initialize AoI matrix for each pair
    aoi_matrices = {pair: np.full(len(time_vector), np.inf) for pair in pairs}
    
    # Process packet receptions to update AoI for each pair
    for sender, receiver, generated_time, received_time in sorted(input_data, key=lambda x: x[3]):
        pair = (sender, receiver)
        reception_index = np.searchsorted(time_vector, received_time)
        if reception_index < len(time_vector):
            # Calculate AoI based on the current packet
            current_aoi = received_time - generated_time
            aoi_matrices[pair][reception_index] = min(aoi_matrices[pair][reception_index], current_aoi)
            # Increment AoI linearly from this point
            if typ=='exp':
                for idx in range(0,reception_index):
                    if aoi_matrices[pair][idx]==np.inf:
                        aoi_matrices[pair][idx] = np.exp(idx* time_step)
                        
                        
                for idx in range(reception_index + 1, len(time_vector)):
                    aoi_matrices[pair][idx] = np.exp(aoi_matrices[pair][idx - 1]+ time_step)
            else:
                for idx in range(0,reception_index):
                    if aoi_matrices[pair][idx]==np.inf:
                        aoi_matrices[pair][idx] = idx* time_step
                
                for idx in range(reception_index + 1, len(time_vector)):
                    aoi_matrices[pair][idx] = aoi_matrices[pair][idx - 1] + time_step
        # print(aoi_matrices[pair])
    # Plot AoI for each sender-receiver pair
    plt.figure(figsize=(14, 8))
    for pair, aoi_matrix in aoi_matrices.items():
        sender, receiver = pair
        plt.plot(time_vector, aoi_matrix, marker = next(marker),markersize=3, label=f'Sender {sender} to Receiver {receiver}')
    
    plt.xlabel('Time (s)')
    plt.title('Age of Information Over Time for Each Sender-Receiver Pair')
    plt.hlines(y =1, xmin = 0, xmax = t,color = 'r', linestyle = '-', label='Minimum AOI') 

    plt.legend()
    plt.grid(True)
    plt.show()
    
    peak_aoi_per_pair = {}
    mean_aoi_per_pair = {}
    
    # Calculate peak and mean AoI for each sender-receiver pair
    for pair, aoi_matrix in aoi_matrices.items():
        # Assuming AoI starts counting after the first packet is received,
        # we exclude the initial np.inf values for accurate calculations
        valid_aoi_values = aoi_matrix[np.isfinite(aoi_matrix)]
        
        # Calculate and store peak AoI
        try:
            peak_aoi_per_pair[pair] = np.max(valid_aoi_values)
            mean_aoi_per_pair[pair] = np.mean(valid_aoi_values)

        except:
            print(selection)
            print(pair)
            print(aoi_matrix)
            peak_aoi_per_pair[pair]=t
            mean_aoi_per_pair[pair]=t

        
        # Calculate and store mean AoI
    
    # Print the calculated metrics for each sender-receiver pair
    # for pair in pairs:
    #     print(f"Sender {pair[0]} to Receiver {pair[1]} - Peak AoI: {peak_aoi_per_pair[pair]}, Mean AoI: {mean_aoi_per_pair[pair]}")
    #np.array(list(peak_aoi_per_pair.values())).mean()

    return np.array(list(peak_aoi_per_pair.values())).mean(),  np.array(list(mean_aoi_per_pair.values())).mean()
    
def AoIperApp(t,selection,intervals,typ='exp'):
    

   howManyapp=set([selection[i][3] for i in range(0,len(selection))])
    # Parameters
    # Determine the simulation parameters
   for app in list(howManyapp):
        input_data=[]
        for i in range(0,len(selection)):
            if selection[i][3]==app:
                input_data.append([selection[i][0],selection[i][1],selection[i][2],selection[i][2]+intervals[i]])
            
        
        
        # Identify unique sender-receiver pairs
        pairs = set((sender, receiver) for sender, receiver, _, _ in input_data)
        
        # Initialize simulation parameters
        total_time = max(item[3] for item in input_data) + 1
        total_time=t
        time_step = 0.01
        time_vector = np.arange(0, total_time, time_step)
        
        # Initialize AoI matrix for each pair
        aoi_matrices = {pair: np.full(len(time_vector), np.inf) for pair in pairs}
        
        # Process packet receptions to update AoI for each pair
        for sender, receiver, generated_time, received_time in sorted(input_data, key=lambda x: x[3]):
            pair = (sender, receiver)
            reception_index = np.searchsorted(time_vector, received_time)
            if reception_index < len(time_vector):
                # Calculate AoI based on the current packet
                current_aoi = received_time - generated_time
                aoi_matrices[pair][reception_index] = min(aoi_matrices[pair][reception_index], current_aoi)
                # Increment AoI linearly from this point
                if typ=='exp':
                    for idx in range(0,reception_index):
                        if aoi_matrices[pair][idx]==np.inf:
                            aoi_matrices[pair][idx] = np.exp(idx* time_step)
                            
                            
                    for idx in range(reception_index + 1, len(time_vector)):
                        aoi_matrices[pair][idx] = np.exp(aoi_matrices[pair][idx - 1]+ time_step)
                else:
                    for idx in range(0,reception_index):
                        if aoi_matrices[pair][idx]==np.inf:
                            aoi_matrices[pair][idx] = idx* time_step
                    
                    for idx in range(reception_index + 1, len(time_vector)):
                        aoi_matrices[pair][idx] = aoi_matrices[pair][idx - 1] + time_step
        
        # Plot AoI for each sender-receiver pair
        plt.figure(figsize=(9, 7))
        for pair, aoi_matrix in aoi_matrices.items():
            sender, receiver = pair
            plt.plot(time_vector, aoi_matrix, marker = next(marker),markersize=3, label=f'Device {sender} to Device {receiver} Data {app}')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('AoI')
        plt.hlines(y =1, xmin = 0, xmax = t,color = 'r', linestyle = '-', label='Min AoI') 
        plt.rcParams.update({'font.size': 20})

        plt.legend(loc="upper left",prop={'size':16})
        plt.grid(True)
        plt.show()
        
        peak_aoi_per_pair = {}
        mean_aoi_per_pair = {}
        
        # Calculate peak and mean AoI for each sender-receiver pair
        for pair, aoi_matrix in aoi_matrices.items():
            # Assuming AoI starts counting after the first packet is received,
            # we exclude the initial np.inf values for accurate calculations
            valid_aoi_values = aoi_matrix[np.isfinite(aoi_matrix)]
            
            # Calculate and store peak AoI
            try:
                peak_aoi_per_pair[pair] = np.max(valid_aoi_values)
                mean_aoi_per_pair[pair] = np.mean(valid_aoi_values)

            except:
                print('No device has sent a msg fo application: ',app)
                print(selection)
                
            # Calculate and store mean AoI
        
        # Print the calculated metrics for each sender-receiver pair
        # for pair in pairs:
        #     print(f"Sender {pair[0]} to Receiver {pair[1]} for Application {app} - Peak AoI: {peak_aoi_per_pair[pair]}, Mean AoI: {mean_aoi_per_pair[pair]}")
                   
    

def getPossibleCom(V_L,V_RF,msgQueues,thresholdRF,thresholdL,M):
    N_possiblecom=0
    if M>1:
        i=0
        for tmp in msgQueues:
            j=0
            for f in tmp:
                for mm in f:
                    for ss in mm:
                        if V_RF[ss][i][j]>=thresholdRF or V_L[ss][i][j]>=thresholdL:
                            N_possiblecom=N_possiblecom+1
                            break    
                j=j+1  
            i=i+1
    else:
        i=0
        for tmp in msgQueues:
            j=0
            for f in tmp:
                for mm in f:
                    for ss in mm:
                        if V_RF[ss][i][j]>=thresholdRF:
                            N_possiblecom=N_possiblecom+1
                            break    
                j=j+1  
            i=i+1
    return N_possiblecom

def getSolution(x):
    solution = pd.DataFrame.from_dict(x, orient="index", columns = ["variable_object"])
    solution.reset_index(inplace=True)
    solution["solution_value"] = solution["variable_object"].apply(lambda item: item.solution_value)
    return solution["solution_value"]


    
    
    