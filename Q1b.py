#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def generate_brownian_paths(NoOfPaths, NoOfSteps, T):    
    np.random.seed(123)
    dt = T / float(NoOfSteps)
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
             Z[:, i] = (Z[:, i] - np.mean(Z[:, i]))/np.std(Z[:, i])
        W[:, i+1] = W[:, i] + np.sqrt(dt)*Z[:, i]
        time[i+1] = time[i] + dt
    
    return {"time": time, "W": W}

# comparing two integrals
def validate_integrals(T=5, NoOfPaths=10000, NoOfSteps=1000):
    Paths = generate_brownian_paths(NoOfPaths, NoOfSteps, T)
    timeGrid = Paths["time"]
    W = Paths["W"]
    dt = T/NoOfSteps
    
    #  Integral of W(s) ds
    integral_Ws = np.sum(W[:, :-1]*dt, axis=1)
    
    # Integral of  (t-s) dW(s)
    weights = (T - timeGrid[:-1]).reshape(1, -1)  
    dW = np.diff(W, axis=1)  
    integral_T_minus_s_dWs = np.sum(weights*dW, axis=1)
    
    # compute  difference between the two integrals
    differences = integral_Ws - integral_T_minus_s_dWs
    mean_difference = np.mean(differences)
    
    print(differences)
    print(f"Mean absolute error: {np.mean(np.abs(differences)):.6f}")
    print(f"Mean difference: {mean_difference:.6f}")
    


validate_integrals()


# In[ ]:




