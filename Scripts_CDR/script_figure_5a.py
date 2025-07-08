# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:30:40 2023

@author: Jordi
"""
import matplotlib
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
##########################################################
# IMPORTANT: The number of qubits of the devices used (n) 
# must be greater than or equal to n_max.



N_rep=1; #Denotes the number of times that the estimate is repeated
n_max=3; #We measure the performance for QFT circuits from size 1 to n_max.
J=4; #J used (J_1) for ZNE-insertion
reg=0.001; #Regularization parameter
theta=np.pi/8; #Angle used for the insertion feature map.









vector_ideal=np.zeros(n_max)
vector_non_corr=np.zeros(n_max)

std_deviation_error=np.zeros(n_max)
estimate=np.zeros(shape=(N_rep,n_max,6))    



for q in range(n_max):
    
    ##################################
    #Generate QFT circuit of n qubits#
    ##################################
    n=q+1
    gates_list,wires_list ,angle_list=QEM.generate_circuit_QFT(n)
    
    
    
        
    # Evaluate both ideal and non-corrected values.        
    vector_ideal[q]=QEM.circuit_ideal(gates_list,wires_list,angle_list)
    vector_non_corr[q]=QEM.circuit_noisy(gates_list,wires_list,angle_list,type_of_noise)

    
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
        estimate[i,q,0]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,0,1,1,0)
        
            
        #ZNE
        estimator=1;
        weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J,estimator,training_set_gates,y_training_ideal,1,0,reg)
        estimate[i,q,1]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J,1,0)
    
    
    
        #INSETION METHOD
        estimator=5; J_2=3;J_1=J;
        weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,reg)
        estimate[i,q,2]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J_2,J_1,theta)
                        
        estimator=5; J_2=1;J_1=J;
        weights=QEM.compute_weigths_shared_training_with_custom_regu(gates_list,wires_list,angle_list,type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,reg)
        estimate[i,q,4]=QEM.estimator(gates_list,wires_list,angle_list,weights,type_of_noise,estimator,J_2,J_1,theta)
            
    
            
        print('Actual n='+str(n)+' N_test='+str(i+1))

    
    
    
#%%



error_final_f1=np.zeros(n_max)
error_final_f2=np.zeros(n_max)
error_final_f3=np.zeros(n_max)
error_final_f5=np.zeros(n_max)


for n in range(n_max):
        
    mean_loss=np.mean(estimate[:,n,0])
    error_final_f1[n]=mean_loss
    
    mean_loss2=np.mean(estimate[:,n,1])
    error_final_f2[n]=mean_loss2

    mean_loss3=np.mean(estimate[:,n,2])
    error_final_f3[n]=mean_loss3
    
    mean_loss5=np.mean(estimate[:,n,4])
    error_final_f5[n]=mean_loss5
    






#%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "lmodern"
})

matplotlib.rcParams['figure.dpi'] = 300




n_vector=np.arange(1,n_max+1)


plt.plot(n_vector,vector_ideal,color='green')
plt.plot(n_vector,vector_non_corr,color='orange',marker='s')
plt.plot(n_vector,error_final_f1,'k',marker='v')
plt.plot(n_vector,error_final_f2,'r',marker='o')
plt.plot(n_vector,error_final_f5,'c',marker='^')
plt.plot(n_vector,error_final_f3,'b',marker='+')


## The part of the code with variables ending in _000 is used to 
## plot the N → ∞ scenario, but it requires uploading the saved file.

#plt.plot(n_vector,vector_non_corr_000,color='orange',linestyle='dashed',marker='s')
#plt.plot(n_vector,error_final_f1_000,'k--',marker='v')
#plt.plot(n_vector,error_final_f2_000,'r--',marker='o')
#plt.plot(n_vector,error_final_f5_000,'c--',marker='^')
#plt.plot(n_vector,error_final_f3_000,'b--',marker='+')


plt.grid(visible=True, which='both', axis='both',color='gray',alpha=0.5)

plt.legend(['Ideal Value','Non-corrected','Classical CDR','CDR based on ZNE',"Insertion method","Generalized Insertion method ($J_2=7$)"],bbox_to_anchor=(-0.07, -0.55), loc="lower left", ncol=3)

plt.xticks(range(1,6))
plt.xlim(1,5.)
plt.xlabel("$n$")
plt.ylabel("Estimate of $f(U)$")
   