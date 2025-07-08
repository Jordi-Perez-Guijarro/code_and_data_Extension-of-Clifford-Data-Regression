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



#%%%


#Data exported from library_QEM.py
##########################################################
n=QEM.n; #Number of qubits of the testing circuits       # 
type_of_noise=QEM.type_of_noise; #Noise model used       #
S=QEM.S; #Size of the training set used                  #
F=QEM.F; #Number of non-substituted gates                #
##########################################################


J_max=5; #Maximum J used (J_1) for ZNE-insertion
N_test=1; #Number of testing circuits used
l=30; #Number of gates on the testing circuits
N_CNOT=7; #Minimum number of CNOT gates in the testing circuits
regu_vector=[10**(-6),10**(-3),10**(-2)] #Regularizations used
theta=np.pi/8; #Angle used for the insertion feature map.

np.random.seed(0)











Number_of_regularizations=len(regu_vector)
loss_circuits=np.zeros(shape=(N_test,J_max,6,Number_of_regularizations))    




##GENERATE TESTING SET TO EVALUATE ERROR OF THE ESTIMATOR
#%%----------------------------------------------------------------
gates_list=[]
for i in range(N_test):
    gates_aux=(np.around(3*np.random.rand(1,l))+1).astype(int)
    gates=[gates_aux[0,j] for j in range(l)]
    gates=QEM.circuit_add_CNOT(N_CNOT,gates)
    gates_list.append(gates)
        
        
        
wires_list=[]
for i in range(N_test):
    wires_indexes_aux=np.around((n-1)*np.random.rand(2,l))
    wires_indexes=wires_indexes_aux.astype(int).tolist()
    wires_indexes=QEM.correct_indexes_wiresCNOT(wires_indexes)
    wires_list.append(wires_indexes)
    
angle_list=[]
for i in range(N_test):
    angles=2*np.pi*np.random.rand(l)
    angle_list.append(angles)

    
            
    #---------------------------------------
    

#Evaluate the test error
for i in range(N_test):

    #Initialization of the training set
    y_training_ideal=np.zeros(shape=(S,1))
    training_set_gates=[]
    vector_excluded=QEM.generate_mask(gates_list[i],F)

    
    
    for z in range (S):
        
        #Generation of the training set from the unitary
        gates_clifford,a,b =QEM.generate_clifford_distribution_U(gates_list[i],angle_list[i],vector_excluded)
        training_set_gates.append(gates_clifford)
        
        #Compute ideal value        
        y_training_ideal[z]=QEM.circuit_ideal(gates_clifford,wires_list[i],angle_list[i])
    
    
    #Estimate and measure the error for the different feature maps.
    
    
    #Vanilla [1, f(U)]^T
    estimator=0
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,1,estimator,training_set_gates,y_training_ideal,1,0,regu_vector)
    for p in range(len(regu_vector)):
        loss_circuits[i,0,0,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,0,1,1,0)
    
    
    for j in range(J_max):
        
        J=j+1;
        
        #ZNE
        estimator=1;
        weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J,estimator,training_set_gates,y_training_ideal,1,0,regu_vector)
        for p in range(len(regu_vector)):
            loss_circuits[i,j,1,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,estimator,J,1,0)
        
        
        
        #INSETION METHOD
        estimator=5; J_2=6;J_1=J; 
        weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
        for p in range(len(regu_vector)):
            loss_circuits[i,j,2,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,estimator,J_2,J_1,theta)
            
        estimator=5; J_2=3;J_1=J;
        weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
        for p in range(len(regu_vector)):
            loss_circuits[i,j,3,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,estimator,J_2,J_1,theta)
            
        estimator=5; J_2=1;J_1=J;
        weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
        for p in range(len(regu_vector)):
            loss_circuits[i,j,4,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,estimator,J_2,J_1,theta)
            
            
        #GEOMETRIC METHOD
        estimator=4; 
        weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J,estimator,training_set_gates,y_training_ideal,0,0,regu_vector)
        for p in range(len(regu_vector)):
            loss_circuits[i,j,5,p]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[p],type_of_noise,estimator,J,J,0)
        
                
        
        
        
        print('J used='+str(J)+' N_test='+str(i+1))
       


#NON-CORRECTED ERROR#
loss_non_corr=np.zeros(N_test)
for i in range(N_test):
    loss_non_corr[i]=QEM.loss_circuit_non_correction(gates_list[i],wires_list[i],angle_list[i],type_of_noise)

mean_non_corr=np.mean(loss_non_corr)
vector_non_corr=mean_non_corr*np.ones(J_max)


#%%

#In this section, the results stored in loss_circuits are plotted.
    
p=2; #denotes the regularization parameter for which the 
     #figure is shown
     
     
error_final_f1=np.zeros(J_max)
error_final_f2=np.zeros(J_max)
error_final_f3=np.zeros(J_max)
error_final_f4=np.zeros(J_max)
error_final_f5=np.zeros(J_max)
error_final_f6=np.zeros(J_max)     
     

for r in range(J_max):
    
    mean_loss=np.mean(loss_circuits[:,r,0,p])
    error_final_f1[r]=mean_loss
    
    mean_loss2=np.mean(loss_circuits[:,r,1,p])
    error_final_f2[r]=mean_loss2
    
    mean_loss3=np.mean(loss_circuits[:,r,2,p])
    error_final_f3[r]=mean_loss3
    
    mean_loss4=np.mean(loss_circuits[:,r,3,p])
    error_final_f4[r]=mean_loss4
    
    mean_loss5=np.mean(loss_circuits[:,r,4,p])
    error_final_f5[r]=mean_loss5
    
    mean_loss6=np.mean(loss_circuits[:,r,5,p])
    error_final_f6[r]=mean_loss6
    


error_final_f1=error_final_f1[0]*np.ones(J_max)






#%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

matplotlib.rcParams['figure.dpi'] = 300

J_vector=np.arange(1,J_max+1)


line,=plt.plot(J_vector,np.sqrt(vector_non_corr))
plt.plot(J_vector,np.sqrt(error_final_f1),'k')
plt.plot(J_vector,np.sqrt(error_final_f6),'g')
plt.plot(J_vector,np.sqrt(error_final_f2),'r')
plt.plot(J_vector,np.sqrt(error_final_f5),'c')
plt.plot(J_vector,np.sqrt(error_final_f4),'y')
plt.plot(J_vector,np.sqrt(error_final_f3),'b')

#This part of the code is used to print the N=infty part
#of figure 1, before running this part, must be stored and opened.    

#plt.plot(J_vector,np.sqrt(vector_non_corr_000),color=line.get_color(),linestyle ='dashed')
#plt.plot(J_vector,np.sqrt(error_final_f1_000),'k--')
#plt.plot(J_vector,np.sqrt(error_final_f6_000),'g--')
#plt.plot(J_vector,np.sqrt(error_final_f2_000),'r--')
#plt.plot(J_vector,np.sqrt(error_final_f5_000),'c--')
#plt.plot(J_vector,np.sqrt(error_final_f4_000),'y--')
#plt.plot(J_vector,np.sqrt(error_final_f3_000),'b--')

plt.grid(visible=True, which='both', axis='both',color='gray',alpha=0.5)

plt.xlabel("$J$ ($J_1$ for the Insertion-ZNE method)")
plt.ylabel("Root Mean Square Error")

plt.legend(['Non-corrected','Classical CDR','Geometric method','CDR based on ZNE',"Insertion method","Insertion-ZNE method ($J_2=3$)","Insertion-ZNE method ($J_2=6$)"],bbox_to_anchor=(-0.07, -0.55), loc="lower left", ncol=3)
   
