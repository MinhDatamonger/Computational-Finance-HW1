#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

def generate_W_X(NoOfPaths, NoOfSteps, T):    
    np.random.seed(123)
    dt = T / float(NoOfSteps)
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
             Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # Standardize
        
        # Simulate Brownian motion paths
        W[:, i+1] = W[:,i] + np.sqrt(dt)*Z[:, i] 
        time[i+1] = time[i] +dt 
        
        #Compute X paths
        for j in range(NoOfPaths):
            t = time[i+1]
            X[j, i+1] =  W[j, i+1] - (t/T)*W[j, np.argmin(np.abs(time - T-t)) ] # argmin index to get the integer index

    return {"time": time, "W": W, "X": X}

# Theoretical variance function
def variance_X(t, T):
    return t -2*(t/T)*min(t, T-t)+(t**2/T**2)*(T-t)

def mainCalculation():
    NoOfPaths = 1000
    NoOfSteps = 1000
    T = 10  
    
    Paths = generate_W_X(NoOfPaths, NoOfSteps, T)
    timeGrid = Paths["time"]
    X = Paths["X"]

    sample_variance_X = np.var(X, axis=0)

    #  calculate the theoretical variance for X(t)
    theoretical_variance_X = np.array([variance_X(t, T) for t in timeGrid])

    # Plot sample variance vs theoretical variance
    plt.figure(1)
    plt.plot(timeGrid, sample_variance_X, label='Sample Variance')
    plt.plot(timeGrid, theoretical_variance_X, label='Theoretical Variance', linestyle='--')
    plt.xlabel("t")
    plt.ylabel("Variance of X(t)")
    plt.legend()
    plt.grid(True)

    #calculate mean absolute error
    mae = np.mean(np.abs(sample_variance_X - theoretical_variance_X))
    print(f"Mean Absolute Error between simulated variance and theoretical Variance: {mae:.6f}")

mainCalculation()


# In[ ]:




