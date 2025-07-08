#IMPORT LIBRARIES
import matplotlib.pyplot as plt
import math
import pennylane as qml
from pennylane import numpy as np
#import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



from qiskit_ibm_runtime import QiskitRuntimeService





from numpy.linalg import inv 
from IPython import get_ipython;   
get_ipython().magic('reset -sf')

    #GATE CODES


# Hadamard --> 1      Identity-->2
# PauliX --> 3        PauliY --> 4   
# PauliZ --> 5        S --> 6
# CNOT --> 7          T--> 8
# 




    #PROPERTIES OF THE CIRCUITS: IDEAL CIRCUIT and NOISY CIRCUIT
#------------------------------------------------------------------
n=3 #Number of qubits of the circuit
p_Single=0 #Noise level for 1-qubit gates
p_CNOT=0.1 #Noise level for 2-qubit gates



S=120 #Number of samples of the training set 
F=7 #Number of non-clifford gates in the training set



N=1000; #Number of shots used to compute the empirical means.
type_of_noise=1 #Type of noise 


######INDEXES TYPE OF NOISE ###########
##  1 one Qubit Depolarazing         ##
##  2 one Qubit Amplitude Damping    ##
##  3 one Qubit Phase Damping        ##
##  4 n Qubit Depolarazing           ##
#######################################

type_of_loss=1

#  1  Mean Absolute Error
#  2  Squared Error


dev_noisy=qml.device("default.mixed",wires=n,shots=N)#Noisy device
#dev_noisy=qml.device("default.mixed",wires=n) # Noisy device using
                                               # N-->infty shots.
dev_ideal=qml.device("default.mixed",wires=n) #Ideal device






#-----------------------------------------------------------------------------#
#----------------DEFINITION OF A n-qubit DEPOLARAZING CHANNEL-----------------#
#-----------------------------------------------------------------------------#


def next_sequence(seq_binary):
    
    n=len(seq_binary)
    seq_copy=seq_binary.copy()
    
    extra=0
    for j in range(n):
        
        if j==0:
            seq_copy[j]=seq_copy[j]+1
            if seq_copy[j]>4:
                seq_copy[j]=1
                extra=1
        else:
            seq_copy[j]=seq_copy[j]+extra
            if seq_copy[j]>4:
                seq_copy[j]=1
                extra=1
            else:
                extra=0
    return seq_copy

def generate_Pauli_matrix(seq_binary):
    
    n=len(seq_binary)
    
    I=[[1,0],[0,1]]
    X=[[0,1],[1,0]]
    Y=[[0,-1j],[1j,0]]
    Z=[[1,0],[0,-1]]
    
    
    Pauli_matrix=[1]
    
    for i in range(n):
        if seq_binary[i]==1:
            Pauli_matrix=np.kron(Pauli_matrix,I)
        elif seq_binary[i]==2:
            Pauli_matrix=np.kron(Pauli_matrix,X)
        elif seq_binary[i]==3:
            Pauli_matrix=np.kron(Pauli_matrix,Y)
        elif seq_binary[i]==4:
            Pauli_matrix=np.kron(Pauli_matrix,Z)

        
    return Pauli_matrix
                
            


def kraus_operators_Depo(n,p):
    #n denotes the number of qubits.
    # Codes for this small section
    # 1--> I   2-->X   3-->Y   4-->Z
    
    seq_binary=np.ones(n)
    
    kraus_operations=[]
    
    I=np.identity(2**n)
    kraus_operations.append(np.sqrt(1-p+p/4**n)*I)
    
    
    cont_next=1    
    while cont_next==1:
        
        seq_binary=next_sequence(seq_binary)
        Pauli_matrix=generate_Pauli_matrix(seq_binary)
        
        kraus_operations.append(np.sqrt(p/(4**n))*Pauli_matrix)
        
        if np.mean(seq_binary)==4:
            cont_next=0
    
    return kraus_operations




def depolarazing_general(n,p):
    
    Kraus_list_=kraus_operators_Depo(n,p) #We compute the kraus operators
    
    wirelist = [i for i in range(n)]
    qml.QubitChannel(K_list=Kraus_list_,wires=wirelist)


#-----------------------------------------------------------------------------#
#---------------- IDEAL CIRCUIT and Auxiliary Functions-----------------------#
#-----------------------------------------------------------------------------#

def code_2_gate(code_gate,wire_gate,phi):
    
    if code_gate==1:
        qml.RX(phi,wires=wire_gate[0])
    elif code_gate==2:
        qml.RY(phi,wires=wire_gate[0])
    elif code_gate==3:
        qml.RZ(phi,wires=wire_gate[0])
    elif code_gate==4:
        qml.CNOT(wires=[wire_gate[0],wire_gate[1]])
    elif code_gate==5:
        qml.PauliX(wires=wire_gate[0])
    elif code_gate==6:
        qml.PauliY(wires=wire_gate[0])  
    elif code_gate==7:
        qml.PauliZ(wires=wire_gate[0])
    elif code_gate==8:
        qml.Identity(wires=wire_gate[0])
    elif code_gate==9:
        qml.Hadamard(wires=wire_gate[0])
    elif code_gate==10:
        qml.S(wires=wire_gate[0])
    elif code_gate==11:
        #sqrt(X)
        qml.SX(wires=wire_gate[0])
    elif code_gate==12:
        #sqrt(Y)
        qml.S(wires=wire_gate[0])
        qml.SX(wires=wire_gate[0])
        qml.adjoint(qml.S(wires=wire_gate[0]))
    elif code_gate==13:
        qml.PhaseShift(phi,wires=wire_gate[0])


          
def correct_indexes_wiresCNOT(seq_wires):
    
    
    for i in range(len(seq_wires[0])):
        while seq_wires[0][i]==seq_wires[1][i]:
            aux=np.around((n-1)*np.random.rand()).astype(int)
            seq_wires[1][i]=aux
                
    return seq_wires
    
    

def ideal_circuit(sequence_gates,sequence_wires,sequence_angles):
    for i in range(len(sequence_gates)):
        code_2_gate(sequence_gates[i],[sequence_wires[0][i],sequence_wires[1][i]],sequence_angles[i])


@qml.qnode(dev_ideal)
def circuit_ideal(sequence_gates,sequence_wires,sequence_angles):#IDEAL CIRCUIT THAT WE WANT TO GENERATE
    
    ideal_circuit(sequence_gates,sequence_wires,sequence_angles)
        
    #Observable Z^{\otimes n} 
    wirelist = [i for i in range(n)]
    PauliZ_matrix=[[1,0],[0,-1]]
    Obs=[1]
    for i in range(n):
        Obs=np.kron(Obs,PauliZ_matrix)

        
    return qml.expval(qml.Hermitian(Obs,wirelist))
       

#-----------------------------------------------------------------------------#
#--------------------NOISY CIRCUIT and Auxiliary Functions -------------------#
#-----------------------------------------------------------------------------#
  
    
def noisy_Hadamard(wires_index,type_noise):
    qml.Hadamard(wires=wires_index[0])
    
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)    
    

def noisy_sqrtX(wires_index,type_noise):
    qml.SX(wires=wires_index[0])
    
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)  

def noisy_sqrtY(wires_index,type_noise):
    
    qml.S(wires=wires_index[0])
    qml.SX(wires=wires_index[0])
    qml.adjoint(qml.S(wires=wires_index[0]))
    
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)  


def noisy_rotationX(wires_index,phi,type_noise):
    qml.RX(phi,wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)      

def noisy_rotationY(wires_index,phi,type_noise):
    qml.RY(phi,wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)
            
def noisy_rotationZ(wires_index,phi,type_noise):
    qml.RZ(phi,wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)            

    
def noisy_S(wires_index,type_noise):
    qml.S(wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)  
        
def noisy_CNOT(wires_index,type_noise):
    qml.CNOT(wires=wires_index)
    if type_noise==1:
        qml.DepolarizingChannel(p_CNOT,wires=wires_index[0])
        qml.DepolarizingChannel(p_CNOT,wires=wires_index[1])
    elif type_noise==2:
        qml.AmplitudeDamping(p_CNOT,wires=wires_index[0])
        qml.AmplitudeDamping(p_CNOT,wires=wires_index[1])
    elif type_noise==3:
        qml.PhaseDamping(p_CNOT,wires=wires_index[0])
        qml.PhaseDamping(p_CNOT,wires=wires_index[1])
    elif type_noise==4:
        depolarazing_general(n,p_CNOT)        

def noisy_T(wires_index,type_noise):
    qml.T(wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)   
         
def noisy_PauliX(wires_index,type_noise):
    qml.PauliX(wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)    
        
def noisy_PauliY(wires_index,type_noise):
    qml.PauliY(wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)    
        
def noisy_PauliZ(wires_index,type_noise):
    qml.PauliZ(wires=wires_index[0])
    if p_Single>0:
        if type_noise==1:
            qml.DepolarizingChannel(p_Single,wires=wires_index[0])
        elif type_noise==2:
            qml.AmplitudeDamping(p_Single,wires=wires_index[0])
        elif type_noise==3:
            qml.PhaseDamping(p_Single,wires=wires_index[0])
        elif type_noise==4:
            depolarazing_general(n,p_Single)   
         
def code_2_gate_noisy(code_gate,wire_gate,phi,type_noise):
    
    if code_gate==1:
        noisy_rotationX(wire_gate,phi,type_noise)
    elif code_gate==2:
        noisy_rotationY(wire_gate,phi,type_noise)  
    elif code_gate==3:
        noisy_rotationZ(wire_gate,phi,type_noise)
    elif code_gate==4:
        noisy_CNOT(wire_gate,type_noise)
    elif code_gate==5:
        noisy_PauliX(wire_gate,type_noise)
    elif code_gate==6:
        noisy_PauliY(wire_gate,type_noise)  
    elif code_gate==7:
        noisy_PauliZ(wire_gate,type_noise)
    elif code_gate==8:
        qml.Identity(wires=wire_gate[0])
    elif code_gate==9:
        noisy_Hadamard(wire_gate,type_noise)
    elif code_gate==10:
        noisy_S(wire_gate,type_noise)
    elif code_gate==11:
        noisy_sqrtX(wire_gate,type_noise)
    elif code_gate==12:
        noisy_sqrtY(wire_gate,type_noise)
        

def noisy_circuit(sequence_gates,sequence_wires,sequence_angles,type_noise):
    
    for i in range(len(sequence_gates)):
        code_2_gate_noisy(sequence_gates[i],[sequence_wires[0][i],sequence_wires[1][i]],sequence_angles[i],type_noise)
        


@qml.qnode(dev_noisy)#Definition of the noisy circuit
def circuit_noisy(sequence_gates,sequence_wires,sequence_angles,type_noise):
    
    noisy_circuit(sequence_gates,sequence_wires,sequence_angles,type_noise)
        
        
    wirelist = [i for i in range(n)]
    PauliZ_matrix=[[1,0],[0,-1]]
    Obs=[1]
    for i in range(n):
        Obs=np.kron(Obs,PauliZ_matrix)

        
    return qml.expval(qml.Hermitian(Obs,wirelist))



@qml.qnode(dev_noisy)#Definition of the noisy circuit
def circuit_noisy_rep(Repetitions,sequence_gates,sequence_wires,sequence_angles,type_noise):
    
    # This function is used for the numerical experiments related to the
    # geometric feature map. 
    # It evaluates the noisy value of \bra{0}U^{rep,\dagger} O U^{rep}\ket{0}
    
    for i in range(Repetitions):
        noisy_circuit(sequence_gates,sequence_wires,sequence_angles,type_noise)
        
        
    wirelist = [i for i in range(n)]
    PauliZ_matrix=[[1,0],[0,-1]]
    Obs=[1]
    for i in range(n):
        Obs=np.kron(Obs,PauliZ_matrix)

        
    return qml.expval(qml.Hermitian(Obs,wirelist))

#-----------------------------------------------------------#
#------------------Quantum Fourier Transform Circuit--------#
#-----------------------------------------------------------#

def generate_circuit_QFT(size_of_circuit):
    
    #size_of_circuit denotes the number of qubits.
    
    #Output: a circuit of the QFT up to an overall phase + layer of
    # Hadarmard at the beggining
    
    number_of_gates=5*(size_of_circuit-1)*size_of_circuit/2+2*size_of_circuit
    number_of_gates=int(number_of_gates)
    
    sequence_of_gates=np.zeros(shape=(number_of_gates))
    angles_of_gates=np.zeros(shape=(number_of_gates))
    wires_of_gates=np.zeros(shape=(2,number_of_gates))
    
    
    
    #Hadamard Layer
    
    index=0
    for i in range(size_of_circuit):
        
        sequence_of_gates[index]=9
        wires_of_gates[0,index]=i
        index=index+1
    
    
    cont_aux=size_of_circuit
    for i in range(size_of_circuit):
        
        target_wire=size_of_circuit-cont_aux
        control_wire=size_of_circuit-cont_aux+1
        
        for j in range(cont_aux):
        

            #Next wire from the target
                                                   
            if j==0:
                #introduce a Hadamard gate
                sequence_of_gates[index]=9
                wires_of_gates[0,index]=target_wire
                index=index+1
            else:
                
                
                distance_wires=control_wire-target_wire
                fundamental_angle=2*np.pi/(2**(distance_wires+1))
                
                
                #introduce the decomposition of the control rotation
                sequence_of_gates[index]=3
                wires_of_gates[0,index]=control_wire
                angles_of_gates[index]=fundamental_angle/2
                
                sequence_of_gates[index+1]=3
                wires_of_gates[0,index+1]=target_wire
                angles_of_gates[index+1]=fundamental_angle/2

                sequence_of_gates[index+2]=4 #CNOT gate
                wires_of_gates[0,index+2]=control_wire
                wires_of_gates[1,index+2]=target_wire

                sequence_of_gates[index+3]=3
                wires_of_gates[0,index+3]=target_wire
                angles_of_gates[index+3]=-fundamental_angle/2
                
                sequence_of_gates[index+4]=4 #CNOT gate
                wires_of_gates[0,index+4]=control_wire
                wires_of_gates[1,index+4]=target_wire

                #UPDATE the control_wire and index
                control_wire=control_wire+1 
                index=index+5

        cont_aux=cont_aux-1

    return sequence_of_gates, wires_of_gates.astype(int).tolist(), angles_of_gates 


#-----------------------------------------------------------------------------#
#---------------------------Different FEATURE VECTORS-------------------------#
#-----------------------------------------------------------------------------#

def feature_vector_VANILLA(sequence_gates,sequence_wires,sequence_angles,type_noise):
    feature=np.zeros(2)
    feature[0]=1
    feature[1]=circuit_noisy(sequence_gates,sequence_wires,sequence_angles,type_noise)
    
    return feature



def feature_vector_ZNE_CNOT(sequence_gates,sequence_wires,sequence_angles,J,type_noise):
    feature=np.zeros(J+1)
    feature[0]=1
    
    
    
    for i in range(J): 
        new_gates,new_wires,new_angles=circuit_3_CNOT_individual(sequence_gates,sequence_wires,sequence_angles,i)
        feature[i+1]=circuit_noisy(new_gates,new_wires,new_angles,type_noise)
    
    
    return feature


def feature_vector_ZNE_CNOT_(sequence_gates,sequence_wires,sequence_angles,J,type_noise):
    
    #This implementation of the ZNE-based approach differs from the previous one
    #only on the way the noise is increased. Check function circuit_3_CNOT
    
    feature=np.zeros(J+1)
    feature[0]=1
    
    for i in range(J): 
        new_gates,new_wires,new_angles=circuit_3_CNOT(sequence_gates,sequence_wires,sequence_angles,i)
        feature[i+1]=circuit_noisy(new_gates,new_wires,new_angles,type_noise)
    
    return feature




def feature_vector_ZNE_insertion(sequence_gates,sequence_wires,sequence_angles,J_2,type_noise,J_1,angle):
    
    
    l=len(sequence_gates)
    midle_index=round(l/2)
    
    copy_sequence_gates=sequence_gates.copy()
    copy_sequence_wires=sequence_wires.copy()
    copy_sequence_angles=sequence_angles.copy()
    
    new_sequence_gates=np.zeros(l+1)
    new_sequence_wires=np.zeros(shape=(2,l+1))
    new_sequence_angles=np.zeros(l+1)
    
    new_sequence_gates[0:midle_index]=copy_sequence_gates[0:midle_index]
    new_sequence_wires[0][0:midle_index]=copy_sequence_wires[0][0:midle_index]
    new_sequence_wires[1][0:midle_index]=copy_sequence_wires[1][0:midle_index]
    new_sequence_angles[0:midle_index]=copy_sequence_angles[0:midle_index]
    
    new_sequence_gates[midle_index]=1
    new_sequence_wires[:,midle_index]=[0,1]
    new_sequence_angles[midle_index]=angle
    
    new_sequence_gates[midle_index+1:]=copy_sequence_gates[midle_index:]
    new_sequence_wires[0][midle_index+1:]=copy_sequence_wires[0][midle_index:]
    new_sequence_wires[1][midle_index+1:]=copy_sequence_wires[1][midle_index:]
    new_sequence_angles[midle_index+1:]=copy_sequence_angles[midle_index:]
    
    
    new_sequence_wires=new_sequence_wires.astype(int).tolist()
    
    
    feature_matrix=np.zeros(J_2*J_1+1)
    feature_matrix[0]=1
    
    for i in range(J_1): #Sampling of the rotation
    
        new_sequence_angles[midle_index]=i*angle
        slice_feature=feature_vector_ZNE_CNOT(new_sequence_gates,new_sequence_wires,new_sequence_angles,J_2,type_noise)
        feature_matrix[i*J_2+1:(i+1)*J_2+1]=slice_feature[1:]
        
    return feature_matrix



def feature_vector_geometric(sequence_gates,sequence_wires,sequence_angles,J,type_noise):
    
    feature=np.zeros(J+1)
    feature[0]=1
    
    for i in range(J):
        feature[i+1]=circuit_noisy_rep(i+1,sequence_gates,sequence_wires,sequence_angles,type_noise)
    
    return feature


    
    

#-----------------------------------------------------------------------------#
#----------------------Auxiliary Functions for Features ----------------------#
#-----------------------------------------------------------------------------#





def circuit_3_CNOT_individual(sequence_gates,sequence_wires,sequence_angles,rep):
    
    #Adds 2*rep CNOT in the circuit
    #A repetition is a folding of a CNOT.
    
    l=len(sequence_gates)
    number_cnot=0
    
    
    for i in range(l):
        if sequence_gates[i]==4:
            number_cnot=number_cnot+1
    
    #GENERATE THE ARRAY WITH THE NUMBER OF INSERTIONS
    
    array_rep=np.zeros(number_cnot)
    
    if number_cnot>0:
        for i in range(rep):
            j=i % number_cnot
            array_rep[j]=array_rep[j]+1
       
    array_rep.astype(int)    
    
    final_gates=np.zeros(l+2*rep)
    final_wires=np.zeros(shape=(2,l+2*rep))
    final_angles=np.zeros(l+2*rep)
    
    
    index=0;
    cont_CNOT=0;
    for i in range(l):
        if sequence_gates[i]==4:
            
            for j in range(int(2*array_rep[cont_CNOT]+1)):
                final_gates[index+j]=4
                
                final_wires[0][index+j]=sequence_wires[0][i]
                final_wires[1][index+j]=sequence_wires[1][i]
                
                final_angles[index+j]=sequence_angles[i]
            
            #update index
            index=int(index+2*array_rep[cont_CNOT]+1)
            
            #Update cont_CONT
            cont_CNOT=cont_CNOT+1
            
        else:
            final_gates[index]=sequence_gates[i]
            final_wires[0][index]=sequence_wires[0][i]
            final_wires[1][index]=sequence_wires[1][i]

            final_angles[index]=sequence_angles[i]
            
            index=index+1

        
    final_wires=final_wires.astype(int).tolist()
                
            
    return final_gates, final_wires, final_angles  





def circuit_3_CNOT(sequence_gates,sequence_wires,sequence_angles,rep):
    
    #Adds 2*rep CNOTs after each original CNOT in the circuit

    
    l=len(sequence_gates)
    number_cnot=0
    
    for i in range(l):
        if sequence_gates[i]==4:
            number_cnot=number_cnot+1
    
    final_gates=np.zeros(l+2*rep*number_cnot)
    final_wires=np.zeros(shape=(2,l+2*rep*number_cnot))
    final_angles=np.zeros(l+2*rep*number_cnot)
    
    index=0
    for i in range(l):
        if sequence_gates[i]==4:
            
            for j in range(2*rep+1):
                final_gates[index+j]=4
                
                final_wires[0][index+j]=sequence_wires[0][i]
                final_wires[1][index+j]=sequence_wires[1][i]
                
                final_angles[index+j]=sequence_angles[i]
            
            #update index
            index=index+2*rep+1
        else:
            final_gates[index]=sequence_gates[i]
            final_wires[0][index]=sequence_wires[0][i]
            final_wires[1][index]=sequence_wires[1][i]

            final_angles[index]=sequence_angles[i]
            
            index=index+1

        
    final_wires=final_wires.astype(int).tolist()
                
            
    return final_gates, final_wires, final_angles   











#-----------------------------------------------------------------------------#
#-------------------------ESTIMATOR and Error---------------------------------#
#-----------------------------------------------------------------------------#


def estimator(sequence_gates,sequence_wires,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle):
    
    
    #feature_vec=np.zeros(R_value+1)
    if type_of_feature==0:
        feature_vec=feature_vector_VANILLA(sequence_gates,sequence_wires,sequence_angles,type_noise)
    elif type_of_feature==1:
        feature_vec=feature_vector_ZNE_CNOT(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==2: 
        feature_vec=feature_vector_ZNE_CNOT_(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==4:
        feature_vec=feature_vector_geometric(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==5:
        feature_vec=feature_vector_ZNE_insertion(sequence_gates,sequence_wires,sequence_angles,J,type_noise,J_1,angle)
        #For this last feature vector the argument J is in reality J_2
    
    l=len(feature_vec)
    estimator=np.dot(np.reshape(weights,l),feature_vec)
    
    if estimator>1:
        estimator=1
    elif estimator<-1:
        estimator=-1
    
    return estimator


def estimator_without_B(sequence_gates,sequence_wires,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle):
    
    
    #feature_vec=np.zeros(R_value+1)
    if type_of_feature==0:
        feature_vec=feature_vector_VANILLA(sequence_gates,sequence_wires,sequence_angles,type_noise)
    elif type_of_feature==1:
        feature_vec=feature_vector_ZNE_CNOT(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==2: 
        feature_vec=feature_vector_ZNE_CNOT_(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==4:
        feature_vec=feature_vector_geometric(sequence_gates,sequence_wires,sequence_angles,J,type_noise)
    elif type_of_feature==5:
        feature_vec=feature_vector_ZNE_insertion(sequence_gates,sequence_wires,sequence_angles,J,type_noise,J_1,angle)

    
    l=len(feature_vec)
    estimator=np.dot(np.reshape(weights,l),feature_vec)

    
    return estimator
   

def loss_circuit(gate_sequence,wires_sequence,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle):
    
    #For the ZNE-insertion method J denotes J_2.
    y_ideal=np.mean(circuit_ideal(gate_sequence,wires_sequence,sequence_angles))
    y_estimator=estimator(gate_sequence,wires_sequence,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle)

    
    if type_of_loss==2:    
        loss=abs(y_ideal-y_estimator)**2
    elif type_of_loss==1:
        loss=abs(y_ideal-y_estimator)
                
    return loss




def loss_circuit_no_abs(gate_sequence,wires_sequence,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle):
            
    y_ideal=np.mean(circuit_ideal(gate_sequence,wires_sequence,sequence_angles))
    y_estimator=estimator_without_B(gate_sequence,wires_sequence,sequence_angles,weights,type_noise,type_of_feature,J,J_1,angle)

    
    if type_of_loss==2:    
        loss=(y_ideal-y_estimator)**2
    elif type_of_loss==1:
        loss=(y_ideal-y_estimator)
                
    return loss

def loss_circuit_non_correction(gate_sequence,wires_sequence,angle_sequence,type_noise):
            
    y_training_ideal=np.mean(circuit_ideal(gate_sequence,wires_sequence,angle_sequence))
    y_training_estimator=circuit_noisy(gate_sequence,wires_sequence,angle_sequence,type_noise)

    if type_of_loss==2:     
        loss=abs(y_training_ideal-y_training_estimator)**2
    elif type_of_loss==1:
        loss=abs(y_training_ideal-y_training_estimator)
        
    return loss


#-----------------------------------------------------------------------------#
#-------------------------Learning and Training set (Aux)---------------------#
#-----------------------------------------------------------------------------#




def compute_weigths_shared_training_with_custom_regu(gates_sequence,wires_sequence,angle_sequence,type_noise,J,type_of_feature,training_set,y_training_ideal,J_1,angle,regularization):

    
    if type_of_feature==0:
        y_training_estimator=np.zeros(shape=(S,2))
    elif type_of_feature==1:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==2:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==4:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==5:
        y_training_estimator=np.zeros(shape=(S,J*J_1+1))



    for i in range (S):
        
        gates_clifford=training_set[i]
        
        if type_of_feature==0:
            y_training_estimator[i]=feature_vector_VANILLA(gates_clifford,wires_sequence,angle_sequence,type_noise)
        elif type_of_feature==1:
            y_training_estimator[i]=feature_vector_ZNE_CNOT(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==2: 
            y_training_estimator[i]=feature_vector_ZNE_CNOT_(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==4:
            y_training_estimator[i]=feature_vector_geometric(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==5:
            y_training_estimator[i]=feature_vector_ZNE_insertion(gates_clifford,wires_sequence,angle_sequence,J,type_noise,J_1,angle)
    
    dim=y_training_estimator.shape
    


    A=inv(np.matmul(np.transpose(y_training_estimator),y_training_estimator)+regularization*np.identity(dim[1]))
    B=np.matmul(A,np.transpose(y_training_estimator))
    weights=np.dot(B,y_training_ideal)
    
    return weights


def compute_weigths_shared_training_with_custom_regu_multiple_regu(gates_sequence,wires_sequence,angle_sequence,type_noise,J,type_of_feature,training_set,y_training_ideal,J_1,angle,regu_list):
    
    #regu_1 is a vector in this case and the function outputs 
    #a list of the weights for each value of the regularization
    #This is done for computational efficiency.
    
    if type_of_feature==0:
        y_training_estimator=np.zeros(shape=(S,2))
    elif type_of_feature==1:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==2:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==4:
        y_training_estimator=np.zeros(shape=(S,J+1))
    elif type_of_feature==5:
        y_training_estimator=np.zeros(shape=(S,J*J_1+1))


    for i in range (S):
        
        gates_clifford=training_set[i]
        
        if type_of_feature==0:
            y_training_estimator[i]=feature_vector_VANILLA(gates_clifford,wires_sequence,angle_sequence,type_noise)
        elif type_of_feature==1:
            y_training_estimator[i]=feature_vector_ZNE_CNOT(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==2: 
            y_training_estimator[i]=feature_vector_ZNE_CNOT_(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==4:
            y_training_estimator[i]=feature_vector_geometric(gates_clifford,wires_sequence,angle_sequence,J,type_noise)
        elif type_of_feature==5:
            y_training_estimator[i]=feature_vector_ZNE_insertion(gates_clifford,wires_sequence,angle_sequence,J,type_noise,J_1,angle)
    
    dim=y_training_estimator.shape    
    
    weights_list=[]
    
    for i in range(len(regu_list)):
        
        regu=regu_list[i]
        
        A=inv(np.matmul(np.transpose(y_training_estimator),y_training_estimator)+regu*np.identity(dim[1]))
        B=np.matmul(A,np.transpose(y_training_estimator))
        weights=np.dot(B,y_training_ideal)
        
        weights_list.append(weights)
    
    return weights_list




def generate_mask(gates_sequence,t): 
    
    # t denotes the number of unchanged gates.
    # Output: a vector with the indexes of the unchanged gates
        
    vector_excluded=np.zeros(t)
    
    
    
    list_of_indexes_with_rotation=[]
    for i in range(len(gates_sequence)):
        if gates_sequence[i]<=3:
            list_of_indexes_with_rotation.append(i)
            
    if t>len(list_of_indexes_with_rotation):
        t=len(list_of_indexes_with_rotation)
        
        
    for j in range(t):
        
        proposal=np.around((len(list_of_indexes_with_rotation)-1)*np.random.rand()).astype(int)
        vector_excluded[j]=list_of_indexes_with_rotation[proposal]
        
        list_of_indexes_with_rotation.remove(vector_excluded[j])
        
        
        
        
    
    return vector_excluded
                


def generate_clifford_distribution_U(gates_sequence,angle_sequence,vector_excluded):
    
    #Generate circuits distributed as D(U)
    
    length_circuit=len(gates_sequence)
    t=len(vector_excluded)
    
    sign_of_circuit=1
    normalization_constant=1
    

    
    new_gates=gates_sequence.copy()
    for i in range(length_circuit):
        exclude=False
        for r in range(t):
            if vector_excluded[r]==i:
                exclude=True
        
        if exclude==False and (gates_sequence[i]==3 or gates_sequence[i]==1 or gates_sequence[i]==2):
            
            
            angle=angle_sequence[i]
            
            a=(1+math.cos(angle)-math.sin(angle))/2
            b=(1-math.cos(angle)-math.sin(angle))/2
            c=1-a-b
            
            
            normalization_constant=normalization_constant*(np.abs(a)+np.abs(b)+np.abs(c))
            
            p_a=np.abs(a)/(np.abs(a)+np.abs(b)+np.abs(c))
            p_b=np.abs(b)/(np.abs(a)+np.abs(b)+np.abs(c))
            p_c=np.abs(c)/(np.abs(a)+np.abs(b)+np.abs(c))
            
            random_gate=np.around(np.random.choice(3,p=[p_a,p_b,p_c])).astype(int)
            


            
            if gates_sequence[i]==1: #RX
                if random_gate==0:
                    #indentity
                    gate=8
                    sign_of_circuit=sign_of_circuit*np.sign(a)
                elif random_gate==1:
                    #X
                    gate=5
                    sign_of_circuit=sign_of_circuit*np.sign(b)
                elif random_gate==2:
                    #sqrt(X)
                    gate=11
                    sign_of_circuit=sign_of_circuit*np.sign(c)

                    
                    
            if gates_sequence[i]==2: #RY
                if random_gate==0:
                    #indentity
                    gate=8
                    sign_of_circuit=sign_of_circuit*np.sign(a)

                elif random_gate==1:
                    #Y
                    gate=6
                    sign_of_circuit=sign_of_circuit*np.sign(b)

                elif random_gate==2:
                    #sqrt(Y)
                    gate=12
                    sign_of_circuit=sign_of_circuit*np.sign(c)


            if gates_sequence[i]==3: #RZ
                if random_gate==0:
                    #indentity
                    gate=8
                    sign_of_circuit=sign_of_circuit*np.sign(a)

                elif random_gate==1:
                    #Z
                    gate=7
                    sign_of_circuit=sign_of_circuit*np.sign(b)

                elif random_gate==2:
                    #S
                    gate=10
                    sign_of_circuit=sign_of_circuit*np.sign(c)

                
                
            new_gates[i]=gate
    
    return new_gates, sign_of_circuit,normalization_constant


def generate_clifford_mask(gates_sequence,vector_excluded):
    
    
    #Generate random clifford circuits with the same structure than U
    
    length_circuit=len(gates_sequence)
    t=len(vector_excluded)
    
    
    new_gates=gates_sequence.copy()
    for i in range(length_circuit):
        exclude=False
        for r in range(t):
            if vector_excluded[r]==i:
                exclude=True
        
        if exclude==False and (gates_sequence[i]==3 or gates_sequence[i]==1 or gates_sequence[i]==2):
            random_gate=np.around(5*np.random.rand()).astype(int)
            
            if random_gate==0:
                gate=5 #X gate
            elif random_gate==1:
                gate=6 #Y gate
            elif random_gate==2:
                gate=7 #Z gate
            elif random_gate==3:
                gate=8 #I gate
            elif random_gate==4:
                gate=9 #H gate
            elif random_gate==5:
                gate=10 #S gate
                
                
            new_gates[i]=gate
    
    return new_gates



def generate_clifford(gates_sequence,t):
    
    
    length_circuit=len(gates_sequence)
    
    vector_excluded=np.zeros(t)
    for j in range(t):
        vector_excluded[j]=np.around((length_circuit-1)*np.random.rand())
    
    new_gates=gates_sequence.copy()
    for i in range(length_circuit):
        exclude=False
        for r in range(t):
            if vector_excluded[r]==i:
                exclude=True
        
        if exclude==False and (gates_sequence[i]==3 or gates_sequence[i]==1 or gates_sequence[i]==2):
            random_gate=np.around(5*np.random.rand()).astype(int)
            
            if random_gate==0:
                gate=5 #X gate
            elif random_gate==1:
                gate=6 #Y gate
            elif random_gate==2:
                gate=7 #Z gate
            elif random_gate==3:
                gate=8 #I gate
            elif random_gate==4:
                gate=9 #H gate
            elif random_gate==5:
                gate=10 #S gate
                
                
            new_gates[i]=gate
    
    return new_gates
                
            
def circuit_add_CNOT_(t,gates):
    
    l_circuit=len(gates)
    new_circuit=gates.copy()
    
    cont=0
    for i in range(l_circuit):
        if gates[i]==4:
            cont=cont+1
            
    t=t-cont
    
    
    while t>0:
        pos=np.around((l_circuit-1)*np.random.rand()).astype(int)
        if gates[pos]!=4:
            new_circuit[pos]=4
            t=t-1
            
    return new_circuit
            
def circuit_add_CNOT(N_CNOT,clifford_gates):
    
    l_circuit=len(clifford_gates)
    new_circuit=clifford_gates.copy()
    
    while N_CNOT>0:
        pos=np.around((l_circuit-1)*np.random.rand()).astype(int)
        new_circuit[pos]=4
        N_CNOT=N_CNOT-1
            
    return new_circuit
