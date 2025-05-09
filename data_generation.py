# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:02:33 2024

@author: Aymen Hamrouni
"""

"""
Network and Message Data Generation Module

This module provides functionality for generating network characteristics and message data
for the hybrid RF-WOC network optimization. It includes:

1. NetworkCharacteristics Class:
   - Generates network topology
   - Calculates channel parameters
   - Computes energy consumption
   - Generates message queues

2. Key Features:
   - Random network topology generation
   - SNR and capacity matrix calculation
   - Energy consumption modeling
   - Message generation with different priorities
   - Support for multiple applications

Author: Aymen Hamrouni
Date: 2024
"""

# data_generation.py
import numpy as np
import random
from collections import defaultdict
from utils import rician_fading
from constants import (
    q, k_B, Temperature, R_L, IlluminationCoefficient, f_WOC, B_WOC,
    P_tx_WOC, T_optical, R, R_photo, lamda, u, theta, phi, A_rec,
    P_N, V_bias, f_RF, lambda_RF, B_RF, P_tx_RF, attenuation_coefficient_RF,
    d0, PL0, sigma_shadowing, G_tx_RF, G_rx_RF
)
import matplotlib.pyplot as plt

class NetworkCharacteristics:
    """
    A class to generate and manage network characteristics for hybrid RF-WOC networks.
    
    This class handles the generation of:
    - Network topology
    - Channel parameters (SNR, capacity)
    - Energy consumption models
    - Message queues and priorities
    
    Attributes:
        numAPs (int): Number of Access Points
        numDevices (int): Number of IoT devices
        num_time_slots (int): Number of time slots in simulation
    """
    
    def __init__(self, numAPs, numDevices, num_time_slots):
        """
        Initialize network characteristics.
        
        Args:
            numAPs (int): Number of Access Points
            numDevices (int): Number of IoT devices
            num_time_slots (int): Number of time slots
        """
        self.numAPs = numAPs
        self.numDevices = numDevices
        self.num_time_slots = num_time_slots
        self.N = numAPs + numDevices  # Total number of nodes

    def generate_network_parameters(self, minD, maxD, minE, maxE):
        """
        Generate network parameters including distances, SNR, capacity, and energy.
        
        Args:
            minD (float): Minimum distance between nodes
            maxD (float): Maximum distance between nodes
            minE (float): Minimum energy consumption
            maxE (float): Maximum energy consumption
            
        Returns:
            tuple: (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
                   snr_db_matrix_RF, capacity_matrix_RF, snr_matrix,
                   capacity_matrix, SendEnergy, RecieveEnergy, EnergyTotal)
        """
        # Initialize matrices
        distance_matrix = np.random.uniform(minD, maxD, size=(self.N, self.N))
        snr_db_matrix_WOC = np.zeros((self.N, self.N, self.num_time_slots))
        capacity_matrix_WOC = np.zeros((self.N, self.N, self.num_time_slots))
        snr_db_matrix_RF = np.zeros((self.N, self.N, self.num_time_slots))
        capacity_matrix_RF = np.zeros((self.N, self.N, self.num_time_slots))
        
        snr_matrix = np.zeros((2, self.N, self.N, self.num_time_slots))
        capacity_matrix = np.zeros((2, self.N, self.N, self.num_time_slots))
        
        SendEnergy = np.zeros((2, self.N, self.N, self.num_time_slots))
        RecieveEnergy = np.zeros((2, self.N, self.N, self.num_time_slots))
        
        # Generate energy for devices (APs have infinite energy)
        EnergyTotal = np.random.uniform(minE, maxE, size=(self.N - self.numAPs))
        EnergyTotal = np.concatenate(([np.inf] * self.numAPs, EnergyTotal))
        
        # Calculate SNR and capacity for each node pair and time slot
        for i in range(self.N):
            for j in range(self.N):
                d = distance_matrix[i, j]
                
                # WOC calculations
                P_signal = P_tx_WOC * T_optical * R * IlluminationCoefficient * ((u+1) * A_rec * (np.cos(theta)**u)*np.cos(phi)) / ((2 * np.pi * (d**2)))
                I = R_photo * P_signal
                In = R_photo * P_N
                N_shot = np.sqrt(2 * q * (I+In) * B_WOC)
                N_thermal = np.sqrt((4 * k_B * Temperature) / R_L * B_WOC)
                N_total = np.sqrt(N_shot**2 + N_thermal**2)
                snr_WOC = I**2 / N_total**2
                snr_WOC_db = 10 * np.log10(snr_WOC)
                channel_capacity_WOC = B_WOC * np.log2(1 + snr_WOC)
                
                # RF calculations with Rician fading
                rician = rician_fading(k_factor=10)  # K-factor = 10 dB for LoS dominance
                fading = np.sqrt(rician**2)
                
                path_loss_RF = PL0 + 10 * attenuation_coefficient_RF * np.log10(d/d0) + np.random.normal(0, sigma_shadowing)
                G_tx_linear = 10**(G_tx_RF / 10)
                G_rx_linear = 10**(G_rx_RF / 10)
                P_rx_RF = P_tx_RF * fading**2 * 10**(-path_loss_RF/10) * G_tx_linear * G_rx_linear
                noise_power = k_B * Temperature * B_RF
                snr_linear_RF = P_rx_RF / noise_power
                snr_RF_db = 10 * np.log10(snr_linear_RF)
                channel_capacity_RF = B_RF * np.log2(1 + snr_linear_RF)
                
                for t in range(self.num_time_slots):
                    # WOC
                    blockage_probability = 0
                    blockage_event = np.random.rand() < blockage_probability
                    
                    if (blockage_event) or (i==j) or (i<self.numAPs and j<self.numAPs) or (i>=self.numAPs and j>=self.numAPs):
                        snr_db_matrix_WOC[i][j][t] = -20
                        capacity_matrix_WOC[i][j][t] = 0
                        SendEnergy[1][i][j][t] = 1e50
                        RecieveEnergy[1][i][j][t] = 1e50
                    else:
                        snr_db_matrix_WOC[i][j][t] = snr_WOC_db
                        capacity_matrix_WOC[i][j][t] = channel_capacity_WOC
                        SendEnergy[1][i][j][t] = (P_tx_WOC/R_photo)/capacity_matrix_WOC[i][j][t]
                        RecieveEnergy[1][i][j][t] = (I*V_bias)/capacity_matrix_WOC[i][j][t]
                    blockage_probability = 0
                    blockage_event = np.random.rand() < blockage_probability
                   
                    # RF
                    if (blockage_event) or i==j or (i<self.numAPs and j<self.numAPs):
                        snr_db_matrix_RF[i][j][t] = -20
                        capacity_matrix_RF[i][j][t] = 0
                        SendEnergy[0][i][j][t] = 1e50
                        RecieveEnergy[0][i][j][t] = 1e50
                    else:
                        snr_db_matrix_RF[i][j][t] = snr_RF_db
                        capacity_matrix_RF[i][j][t] = channel_capacity_RF
                        SendEnergy[0][i][j][t] = P_tx_RF/capacity_matrix_RF[i][j][t]
                        RecieveEnergy[0][i][j][t] = P_rx_RF/capacity_matrix_RF[i][j][t]
                    
                    # Update combined matrices
                    snr_matrix[0][i][j][t] = snr_db_matrix_RF[i][j][t]
                    snr_matrix[1][i][j][t] = snr_db_matrix_WOC[i][j][t]
                    capacity_matrix[0][i][j][t] = capacity_matrix_RF[i][j][t]
                    capacity_matrix[1][i][j][t] = capacity_matrix_WOC[i][j][t]
        
        return (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
                snr_db_matrix_RF, capacity_matrix_RF, snr_matrix, capacity_matrix,
                SendEnergy, RecieveEnergy, EnergyTotal)
    
    def generate_messages(self, minSize, maxSize, appSize, data_gen_prob):
        """
        Generate message queues with different priorities and sizes.
        
        Args:
            minSize (int): Minimum message size
            maxSize (int): Maximum message size
            appSize (int): Application-specific size
            data_gen_prob (float): Probability of message generation
            
        Returns:
            tuple: (T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp)
        """
        T = np.zeros((self.num_time_slots, self.N, self.N))
        MessageBegin = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        MessageEnd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        msgQueues = [[[] for _ in range(self.N)] for _ in range(self.N)]
        msgApp = [[[] for _ in range(self.N)] for _ in range(self.N)]
        BiggestMsg = 0
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j or (i < self.numAPs and j < self.numAPs):  # no messages between APs or self
                    continue
                    
                # Bernoulli process for each time slot
                for t in range(self.num_time_slots):
                    if np.random.random() < data_gen_prob:  # 70% probability of message
                        # Generate message length
                        msg_length = np.random.randint(minSize, maxSize + 1)
                        if t + msg_length > self.num_time_slots:
                            continue
                            
                        # Create message
                        msg_id = len(msgQueues[i][j])
                        msgQueues[i][j].append(list(range(t, t + msg_length)))
                        MessageBegin[i][j][msg_id] = t
                        MessageEnd[i][j][msg_id] = t + msg_length
                        
                        # Update T matrix
                        for k in range(t, t + msg_length):
                            T[k][i][j] = 1
                            
                        # Update BiggestMsg
                        if msg_length > BiggestMsg:
                            BiggestMsg = msg_length
                            
                        # Assign application type
                        app_type = np.random.randint(1, appSize + 1)
                        msgApp[i][j].append([app_type] * msg_length)
        
        return T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp
                                
    def generate_size(self, msgQueues, sizemin, sizemax, S_p):
        """
        Generate message sizes for the queues.
        
        Args:
            msgQueues (list): List of message queues
            sizemin (int): Minimum size
            sizemax (int): Maximum size
            S_p (int): Packet size
            
        Returns:
            tuple: (Pt, Pt_number, maxN)
        """
        #low, meduim, high
        Pt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        Pt_number = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        maxN=0
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    for nmessage in range(0,len(msgQueues[i][j])):
                        Pt[i][j][nmessage]=random.randint(sizemin, sizemax) # generate between 1 and 20 Megabites
                        Pt_number[i][j][nmessage]=int(Pt[i][j][nmessage]/S_p) # generate between 1 and 20 Megabites
                        maxN= int(Pt[i][j][nmessage]/S_p) if int(Pt[i][j][nmessage]/S_p)>maxN else maxN

                        
                            
                        
        return Pt,Pt_number,maxN
    

    