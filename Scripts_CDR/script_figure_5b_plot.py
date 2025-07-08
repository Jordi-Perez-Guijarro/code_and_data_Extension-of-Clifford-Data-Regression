# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:53:22 2024

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



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

matplotlib.rcParams['figure.dpi'] = 300

import library_QEM as QEM


#Data using number of shoots N=1000
estimate_classic_CDR_1k=[0.81587,0.465476,0.0776279,-0.00757489,0.00878259,-0.000270593,0.0339278,0.00519294]
estimate_ZNE_1k=[0.859069,0.692806,0.237099,-0.0253641,0.014633,-0.00703412,0.0352686,0.0126619]
estimate_Insertion_ZNE_1k=[0.893386,0.736775,0.390575,0.0678537,-0.0516474,-0.00539698,0.0537306,0.00466858]
estimate_Insertion_1k=[0.85447,0.677484,0.329224,0.0570367,0.00566426,0.00663254,0.0533951,7.75373e-06]
estimate_non_corr_1k=[0.3494,0.1136,0.047,0.0144,-0.0012,-0.0022]

#Data using number of shoots N --> infty
estimate_classic_CDR_inf=[0.907584,0.802976,0.699676,0.57191,0.420773,0.305436,0.0339278,0.00504806]
estimate_ZNE_inf=[0.933144,0.842979,0.757232,0.628441,0.485896,0.317182,0.0352686,0.0043851]
estimate_Insertion_ZNE_inf=[0.933551,0.843597,0.760811,0.63367,0.496396,0.343478,0.0537306,0.00244976]
estimate_Insertion_inf=[0.907586,0.802985,0.699733,0.572224,0.422285,0.313355,0.0533951,0.00421424]
estimate_non_corr_inf=[0.346783,0.117425,0.0386534,0.0122873,0.00373216,0.00106304]

p_values_aux=[0.01,0.02,0.03,0.04,0.05,0.06]
p_values=[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1]


plt.plot(p_values_aux,np.ones(shape=(6,1)),color='green')
plt.plot(p_values,estimate_classic_CDR_1k,'k',marker='v')
plt.plot(p_values_aux,estimate_non_corr_1k,color='orange',marker='s')
plt.plot(p_values,estimate_ZNE_1k,'r',marker='o')
plt.plot(p_values,estimate_Insertion_1k,'c',marker='^')
plt.plot(p_values,estimate_Insertion_ZNE_1k,'b',marker='+')


plt.plot(p_values,estimate_classic_CDR_inf,'k--',marker='v')
plt.plot(p_values,estimate_ZNE_inf,'r--',marker='o')
plt.plot(p_values,estimate_Insertion_ZNE_inf,'b--',marker='+')
plt.plot(p_values,estimate_Insertion_inf,'c--',marker='^')
plt.plot(p_values_aux,estimate_non_corr_inf,color='orange',linestyle='dashed',marker='s')



plt.legend(['Ideal Value','Classical CDR','Non-corrected','CDR based on ZNE',"Insertion method","Insertion-ZNE method ($J_2=7$)"],bbox_to_anchor=(-0.07, -0.55), loc="lower left", ncol=3)





plt.xlim(0.01,0.06);plt.xlabel('$p$');plt.ylabel('Estimate of $f(U)$')

plt.grid(visible=True, which='both', axis='both',color='gray',alpha=0.5)














