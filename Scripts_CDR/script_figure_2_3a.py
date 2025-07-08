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

#Data exported from library_QEM.py
##########################################################
n=QEM.n; #Number of qubits of the testing circuits       # 
type_of_noise=QEM.type_of_noise; #Noise model used       #
S=QEM.S; #Size of the training set used                  #
F=QEM.F; #Number of non-substituted gates                #
##########################################################

L_reg=9; #Number of different regularizations used in the figure
N_test=5; #Number of testing circuits used
l=30; #Number of gates on the testing circuits
N_CNOT=7; #Minimum number of CNOT gates in the testing circuits
J=7; #J used (J_1) for ZNE-insertion
theta=np.pi/8; #Angle used for the insertion feature map.


regu_vector=np.logspace(-8,-1,num=L_reg,endpoint=True)
np.random.seed(50)
loss_circuits=np.zeros(shape=(N_test,L_reg,6))    




##GENERATE TESTING SET TO EVALUATE ERROR OF THE ESTIMATOR
#----------------------------------------------------------------
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

    
            
#%%    



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

        
        
    #VANILLA
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,1,0,training_set_gates,y_training_ideal,1,0,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,0]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,0,1,1,0)
            
       
        
    #ZNE
    estimator=1;
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J,estimator,training_set_gates,y_training_ideal,1,0,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,1]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,estimator,J,1,0)



    #INSETION METHOD
    estimator=5; J_2=6;J_1=J;
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,2]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,estimator,J_2,J_1,theta)
        
    estimator=5; J_2=3;J_1=J; 
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,3]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,estimator,J_2,J_1,theta)
        
    estimator=5; J_2=1;J_1=J; 
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J_2,estimator,training_set_gates,y_training_ideal,J_1,theta,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,4]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,estimator,J_2,J_1,theta)
        
        
    #GEOMETRIC METHOD
    estimator=4; 
    weights_list=QEM.compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_list[i],wires_list[i],angle_list[i],type_of_noise,J,estimator,training_set_gates,y_training_ideal,0,0,regu_vector)
    for t in range(len(regu_vector)):
        loss_circuits[i,t,5]=QEM.loss_circuit(gates_list[i],wires_list[i],angle_list[i],weights_list[t],type_of_noise,estimator,J,J,0)
        


    print('N_test='+str(i+1))


#NON-CORRECTED
loss_non_corr=np.zeros(N_test)
for i in range(N_test):
    loss_non_corr[i]=QEM.loss_circuit_non_correction(gates_list[i],wires_list[i],angle_list[i],type_of_noise)

mean_non_corr=np.mean(loss_non_corr)
vector_non_corr=mean_non_corr*np.ones(L_reg)


#%%


#In this section, the results stored in loss_circuits are plotted.



error_final_f1=np.zeros(L_reg)
error_final_f2=np.zeros(L_reg)
error_final_f3=np.zeros(L_reg)
error_final_f4=np.zeros(L_reg)
error_final_f5=np.zeros(L_reg)
error_final_f6=np.zeros(L_reg)

#loss_circuits=loss_circuits_aux_;
    
for r in range(L_reg):
    
    mean_loss=np.mean(loss_circuits[:,r,0])
    error_final_f1[r]=mean_loss
    
    mean_loss2=np.mean(loss_circuits[:,r,1])
    error_final_f2[r]=mean_loss2
    
    mean_loss3=np.mean(loss_circuits[:,r,2])
    error_final_f3[r]=mean_loss3
    
    mean_loss4=np.mean(loss_circuits[:,r,3])
    error_final_f4[r]=mean_loss4
    
    mean_loss5=np.mean(loss_circuits[:,r,4])
    error_final_f5[r]=mean_loss5
    
    mean_loss6=np.mean(loss_circuits[:,r,5])
    error_final_f6[r]=mean_loss6
    

#%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

matplotlib.rcParams['figure.dpi'] = 300



line,=plt.plot(regu_vector,np.sqrt(vector_non_corr))
plt.plot(regu_vector,np.sqrt(error_final_f1),'k')
plt.plot(regu_vector,np.sqrt(error_final_f6),'g')
plt.plot(regu_vector,np.sqrt(error_final_f2),'r')
plt.plot(regu_vector,np.sqrt(error_final_f5),'c')
plt.plot(regu_vector,np.sqrt(error_final_f4),'y')
plt.plot(regu_vector,np.sqrt(error_final_f3),'b')

#plt.plot(regu_1_vector,np.sqrt(vector_non_corr_000),color=line.get_color(),linestyle ='dashed')
#plt.plot(regu_1_vector,np.sqrt(error_final_f1_000),'k--')
#plt.plot(regu_1_vector,np.sqrt(error_final_f6_000),'g--')
#plt.plot(regu_1_vector,np.sqrt(error_final_f2_000),'r--')
#plt.plot(regu_1_vector,np.sqrt(error_final_f5_000),'c--')
#plt.plot(regu_1_vector,np.sqrt(error_final_f4_000),'y--')
#plt.plot(regu_1_vector,np.sqrt(error_final_f3_000),'b--')

plt.grid(visible=True, which='both', axis='both',color='gray',alpha=0.5)



legend=plt.legend(['Non-corrected','Classical CDR','Geometric method','CDR based on ZNE',"Insertion method","Insertion-ZNE method ($J_2=3$)","Insertion-ZNE method ($J_2=6$)"],bbox_to_anchor=(1.6,0.75),loc='upper right')
plt.xscale('log',base=10)
plt.xlabel("$\mu$")
plt.ylabel("Root Mean Square Error")


 

#%%


pos=4;#Selection of a near-optimal regularization parameter
aux=np.sqrt(loss_circuits)  # The square root is included to cancel
                            # out the square in the loss function.


# These vectors of errors are used to generate a histogram of the errors.
vector_ZNE=np.zeros(N_test)
vector_ZNE=aux[:,pos,1]
vector_classic=aux[:,pos,0]
vector_Insertion=aux[:,pos,4]
vector_Insertion_J2_6=aux[:,pos,3]
vector_geometric=aux[:,pos,5]

## The part of the code with variables ending in _000 is used to 
## plot the N → ∞ scenario, but it requires uploading the saved file.
#aux_000=np.sqrt(loss_circuits_000)
#vector_ZNE_000=aux_000[:,pos,1]
#vector_classic_000=aux_000[:,pos,0]
#vector_Insertion_000=aux_000[:,pos,4]
#vector_Insertion_R_6_000=aux_000[:,pos,3]
#vector_geometric_000=aux_000[:,pos,5]


#%%

#The parameters in this section are fine-tuned for the case N_test = 1000,
#with the remaining parameters matching those given in the paper.



f,(ax,ax_1,ax_2,ax_3,ax_4)=plt.subplots(5, 1,sharex=True,sharey=True)

y_max=160;y_max_000=160
   
ax.hist(vector_classic,bins=60,alpha=0.4,color='k',edgecolor = 'Black')
ax.axvline(vector_classic.mean(), color='k',alpha=0.4)
ax.text(vector_classic.mean()+0.01, y_max*1.7, 'Mean \n {:.2f}'.format(vector_classic.mean()),fontsize='small')
ax.set_xlim(0,0.6); ax.set_ylim(0,700)


#ax.hist(vector_classic_000,bins=60,alpha=0.6,color='black',)
#ax.axvline(vector_classic_000.mean(), color='k',alpha=1)
#ax.text(vector_classic_000.mean()+0.01, y_max_000*1.7, 'Mean\n{:.2f}'.format(vector_classic_000.mean()),fontsize='small',color='k')


ax_1.hist(vector_geometric,bins=60,alpha=0.4,color='green',edgecolor = 'Black')
ax_1.axvline(vector_geometric.mean(), color='green',alpha=0.4)
ax_1.text(vector_geometric.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_geometric.mean()),fontsize='small',color='green',)

#ax_1.hist(vector_geometric_000,bins=60,alpha=0.6,color='green')
#ax_1.axvline(vector_geometric_000.mean(), color='green',alpha=1)
#ax_1.text(vector_geometric_000.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_geometric_000.mean()),fontsize='small',color='green')



ax_2.hist(vector_ZNE,bins=60,alpha=0.3,color='red',edgecolor = 'Black')
ax_2.axvline(vector_ZNE.mean(), color='red',alpha=0.4)
ax_2.text(vector_ZNE.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_ZNE.mean()),fontsize='small',color='red')

#ax_2.hist(vector_ZNE_000,bins=60,alpha=0.6,color='red')
#ax_2.axvline(vector_ZNE_000.mean(), color='red',alpha=1)
#ax_2.text(vector_ZNE_000.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_ZNE_000.mean()),fontsize='small',color='red')

ax_3.hist(vector_Insertion,bins=60,alpha=0.4,color='c',edgecolor = 'Black')
ax_3.axvline(vector_Insertion.mean(), color='c',alpha=0.4)
ax_3.text(vector_Insertion.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_Insertion.mean()),fontsize='small',color='c')

#ax_3.hist(vector_Insertion_000,bins=60,alpha=0.6,color='c')
#ax_3.axvline(vector_Insertion_000.mean(), color='c',alpha=1)
#ax_3.text(vector_Insertion_000.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_Insertion_000.mean()),fontsize='small',color='c')


ax_4.hist(vector_Insertion_J2_6,bins=60,alpha=0.4,color='y',edgecolor = 'Black')
ax_4.axvline(vector_Insertion_J2_6.mean(), color='y',alpha=0.4)
ax_4.text(vector_Insertion_J2_6.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_Insertion_J2_6.mean()),fontsize='small',color='y')

#ax_4.hist(vector_Insertion_R_6_000,bins=60,alpha=0.6,color='y')
#ax_4.axvline(vector_Insertion_R_6_000.mean(), color='y',alpha=1)
#ax_4.text(vector_Insertion_R_6_000.mean()+0.01, y_max*1.7, 'Mean\n{:.2f}'.format(vector_Insertion_R_6_000.mean()),fontsize='small',color='y')



plt.xlabel('Estimation Error')
f.subplots_adjust(hspace=0)   
