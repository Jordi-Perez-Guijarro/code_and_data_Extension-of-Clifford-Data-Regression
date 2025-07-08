# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:15:06 2024

@author: Jordi
"""

import matplotlib
import matplotlib.pyplot as plt
import math
import pennylane as qml
from pennylane import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# The data for this figure is generated using script_figure_1.py for several
# values of the number of shots N. To do this, we need to modify the value 
# of N in the library_QEM module and run script_figure_1.py with the 
# appropriate parameters.(Some lines of script_figure_1.py are not necessary
# for this simulation)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

matplotlib.rcParams['figure.dpi'] = 300


N_values=[1000,2000,5000,10000,20000,50000,100000,200000]

insertion_J2_6=[0.162341,0.1348789592394973 ,0.109327,0.0906627,0.0816502,0.0614146,0.0541657,0.0468254]
insertion_J2_6_infty=0.026799


classical=[0.204255,0.17634,0.137108,0.113243,0.0968815,0.0841376,0.0808141,0.0782276]
classical_infty=0.0747721


ZNE_based=[0.190274,0.151193,0.122299,0.100494,0.089001,0.0727154,0.0614089,0.0517544]
ZNE_based_infty=0.0287161

insertion_J2_3=[0.151036,0.120673,0.102933,0.0880477,0.0757702,0.0656199,0.0573196,0.0562788]
insertion_J2_3_infty=0.0424784

insertion_J2_1=[0.159944,0.133569,0.108751,0.0880783,0.0792191,0.0738661,0.0686638,0.0657307]
insertion_J2_1_infty=0.06461



plt.plot(N_values,classical,'k')
plt.plot(N_values,ZNE_based,'r')
plt.plot(N_values,insertion_J2_1,'c')
plt.plot(N_values,insertion_J2_3,'y')
plt.plot(N_values,insertion_J2_6,'b')


plt.plot(N_values,classical_infty*np.ones(8),'k--')
plt.plot(N_values,ZNE_based_infty*np.ones(8),'r--')
plt.plot(N_values,insertion_J2_1_infty*np.ones(8),'c--')
plt.plot(N_values,insertion_J2_3_infty*np.ones(8),'y--')
plt.plot(N_values,insertion_J2_6_infty*np.ones(8),'b--')

plt.xscale('log',base=10)
plt.grid(visible=True, which='both', axis='both',color='gray',alpha=0.5)
plt.xlabel("$N$")
plt.ylabel("Root Mean Square Error")

legend=plt.legend(['Classical CDR','CDR based on ZNE',"Insertion method","Insertion-ZNE method ($J_2=3$)","Insertion-ZNE method ($J_2=6$)"],bbox_to_anchor=(1.55,0.75),loc='upper right')




