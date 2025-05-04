# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:02:33 2024

@author: Aymen Hamrouni
"""

# data_generation.py
import numpy as np
import random
from collections import defaultdict

class NetworkCharacteristics:
    def __init__(self, num_nodes, num_time_slots,nAPs):
        self.num_nodes = num_nodes
        self.num_time_slots = num_time_slots
        self.nAPs=nAPs
        #we need to introduce data types
        #
    def generate_messages(self,messageSize=2,mini=1,maxi=3,appSize=2):
        #add a pipeline for many messages
        MessageBegin=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        MessageEnd=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i==j or ((i< self.nAPs) and (j< self.nAPs)):
                    continue
                #np.random.seed(0)

                nmessages=random.randint(0, messageSize)
                tbeg=0
                for nmessage in range(nmessages):
                    #np.random.seed(0)

                    tbeg=random.randint(tbeg,tbeg+2)
                    if tbeg is None:
                        continue
                    if tbeg>= self.num_time_slots:
                        continue
                    MessageBegin[i][j][nmessage]=tbeg
                    #np.random.seed(0)

                    MessageEnd[i][j][nmessage]=tbeg+random.randint(mini,maxi)
                    while MessageEnd[i][j][nmessage]> self.num_time_slots:
                        #np.random.seed(0)
                        MessageEnd[i][j][nmessage]=tbeg+random.randint(mini,maxi)
                    tbeg=MessageEnd[i][j][nmessage]
                    #doesnt make sense to send two messages to one signle reciever at the same time
                    #random.randint(1,4) freshness of the data
        T = np.zeros((self.num_time_slots, self.num_nodes, self.num_nodes))
        msgQueues=[[[[] for f in range(len(MessageBegin[i][j]))] for j in range (self.num_nodes)] for i in range(self.num_nodes)]
        BiggestMsg=0
        msgApp=[[[[] for f in range(len(MessageBegin[i][j]))] for j in range (self.num_nodes)] for i in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for nmessage in range(len(MessageBegin[i][j])):
                    if MessageBegin[i][j][nmessage]!=[]:
                        s=MessageBegin[i][j][nmessage]
                        appNumber=random.choice(range(1,appSize+1))
                        while s < MessageEnd[i][j][nmessage]:
                            msgQueues[i][j][nmessage].append(s)
                            T[s][i][j]=1
                            s=s+1
                            msgApp[i][j][nmessage].append(appNumber)
                        if BiggestMsg< (MessageEnd[i][j][nmessage]-MessageBegin[i][j][nmessage]):
                            BiggestMsg=MessageEnd[i][j][nmessage]-MessageBegin[i][j][nmessage]
        return T,MessageBegin,MessageEnd,msgQueues,BiggestMsg,msgApp
                                


    def generate_priorities(self,msgQueues,PriorityLevels=[0.2,0.5,1]):
        #low, meduim, high
        Pt = np.zeros((self.num_time_slots, self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for nmessage in range(0,len(msgQueues[i][j])):
                    #np.random.seed(0)

                    Pt[nmessage][i][j]=np.random.choice(PriorityLevels)
                        
                        
        return Pt
    
    
    def generate_energy(self,mini=400,maxi=600):
        #np.random.seed(0)

        EnergyTotal=np.random.randint(mini, maxi, size=(self.num_nodes))

        return EnergyTotal
    
    
    

    def generate_visibility_RF(self,mini=0,maxi=100):
        #np.random.seed(0)
        nAPs=self.nAPs
        visibility_matrix = np.random.randint(mini, maxi, size=(self.num_time_slots, self.num_nodes, self.num_nodes))
            # the visibility matrix can also be dynamic 
        # Make the matrix symmetric
        for k in range(self.num_time_slots):
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    visibility_matrix[k][i][j]= visibility_matrix[k][j][i]
        
        for k in range(self.num_time_slots):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if (i<nAPs and j<nAPs):
                        visibility_matrix[k][i][j]=0
        
        visibility_matrix=visibility_matrix/100
        return visibility_matrix
    
    def generate_visibility_VLC(self,mini=0,maxi=100):
        #np.random.seed(0)
        nAPs=self.nAPs

        visibility_matrix = np.random.randint(mini, maxi, size=(self.num_time_slots, self.num_nodes, self.num_nodes))
            # the visibility matrix can also be dynamic 
        # Make the matrix symmetric
        for k in range(self.num_time_slots):
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    visibility_matrix[k][i][j]= visibility_matrix[k][j][i]
                    if ((i>=nAPs and j>=nAPs) or (i<nAPs and j<nAPs)):
                        visibility_matrix[k][i][j]=0
                        
                
        for k in range(self.num_time_slots):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if ((i>=nAPs and j>=nAPs) or (i<nAPs and j<nAPs)):
                        visibility_matrix[k][i][j]=0
                                                
        visibility_matrix=visibility_matrix/100
        return visibility_matrix
    
    
    