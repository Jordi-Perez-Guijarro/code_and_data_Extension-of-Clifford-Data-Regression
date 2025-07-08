# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:30:40 2023

@author: Jordi
"""
import matplotlib.pyplot as plt
import math
import pennylane as qml
from pennylane import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'






from numpy.linalg import inv 
from IPython import get_ipython;   
get_ipython().magic('reset -sf')




import library_QEM as QEM





#--------------------------------------------------------------------------

#Data exported from library_QEM.py
##########################################################
n=QEM.n; #Number of qubits of the testing circuits       # 
type_of_noise=QEM.type_of_noise; #Noise model used       #
S=QEM.S; #Size of the training set used                  #
F=QEM.F; #Number of non-substituted gates                #
p=QEM.p_CNOT; #Noise level used                          #
##########################################################

N_rep=3; #Denotes the number of times that the estimate is repeated
J=3; #J used (J_1) for ZNE-insertion
reg=10**(-5); #Regularization parameter
theta=np.pi/8; #Angle used for the insertion feature map.



print('Noise level p='+str(p))
estimate=np.zeros(shape=(N_rep,6))    




    ##################################
    #Generate QFT circuit of n qubits#
    ##################################    
    
gates_list,wires_list ,angle_list=QEM.generate_circuit_QFT(n)
    
    
    
        
# Evaluate both ideal and non-corrected values.                        
ideal=QEM.circuit_ideal(gates_list,wires_list,angle_list)
non_corr=QEM.circuit_noisy(gates_list,wires_list,angle_list,type_of_noise)
    
# Evaluate CDR methods N_rep times
for i in range(N_rep):
        
        
    #Initialization of the training set
    y_training_ideal=np.zeros(shape=(S,1))
    training_set_gates=[]
    
    vector_excluded=QEM.generate_mask(gates_list,F)
    
    
        
    for z in range (S):
            
        #Generation of the training set from the unitary
        gates_clifford,a,b =QEM.generate_clifford_distribution_U(gates_list,angle_list,vector_excluded)
        training_set_gates.append(gates_clifford)
            
        #Compute ideal value        
        y_training_ideal[z]=QEM.circuit_ideal(gates_clifford,wires_list,angle_list)
        
        
    #VANILLA
    weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,1,0,training_set_gates,y_training_ideal,1,0,reg)
    estimate[i,0]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,0,1,1,0)
        
            
    #ZNE
    estimator=1;
    weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J,estimator,training_set_gates,y_training_ideal,1,0,reg)
    estimate[i,1]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J,1,0)
    
    
    
    #INSETION METHOD
    estimator=5; J_2=7;J_1=J;
    weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,reg)
    estimate[i,2]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J_2,J_1,theta)
            
            
    estimator=5; J_2=1;J_1=J;
    weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,reg)
    estimate[i,4]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J_2,J_1,theta)
            
       
    #NON-CORRECTED
    estimate[i,5]=QEM.circuit_noisy(gates_list,wires_list,angle_list,type_of_noise)

    
    print('N_test='+str(i+1))


    
    
    
#%%
    
    
mean_estimate=np.mean(estimate[:,0]) 
mean_estimate2=np.mean(estimate[:,1])
mean_estimate3=np.mean(estimate[:,2])
mean_estimate5=np.mean(estimate[:,4])
mean_estimate4=np.mean(estimate[:,5])



print(' Estimate vanilla:      '+str(mean_estimate))
print(' Estimate ZNE:          '+str(mean_estimate2))
print(' Estimate insertion:    '+str(mean_estimate5))
print(' Estimate insertion+ZNE:'+str(mean_estimate3))
print(' Estimate Non-corrected:'+str(mean_estimate4))





    