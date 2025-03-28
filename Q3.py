#!/usr/bin/env python
# coding: utf-8

# In[6]:


from scipy.stats import norm
import numpy as np

#  Theoretical option price from Black-Scholes formula 
def black_scholes_theoretical_price(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

#  Black-Scholes option price via simulation
def black_scholes_option_price(S0 = 90, K=100,  T=1, r=0.05, sigma=0.4, NoOfPaths=10000, NoOfSteps=500, standardize=True):
    np.random.seed(123)
    dt = T / NoOfSteps
    Z = np.random.normal( 0.0, 1.0, [NoOfPaths, NoOfSteps])
    
    if standardize:
        # standardize 
        for i in range(NoOfSteps):
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i]))/np.std(Z[:, i])
    
    #  simulate paths 
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    S[:, 0] = S0
    
    for i in range(NoOfSteps):
        S[:, i+ 1] = S[:, i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, i])
    
    #  calculate call option payoff
    payoff = np.maximum(S[:, -1] - K , 0)
    option_price = np.exp(-r*T)*np.mean(payoff)
    
    return option_price


def compare_option_pricing():
    S0 = 90
    K = 100
    T = 1
    r = 0.02
    sigma = 0.4
    NoOfPaths = 10000
    NoOfSteps = 500
    
    #  theoretical price
    theoretical_price = black_scholes_theoretical_price(S0, K, T, r, sigma)
    
    #  Simulated prices 
    option_price_with_standardization = black_scholes_option_price(S0, K, T, r, sigma, NoOfPaths, NoOfSteps, standardize=True)
    option_price_without_standardization = black_scholes_option_price(S0, K, T, r, sigma, NoOfPaths, NoOfSteps, standardize=False)
    
    #  Absolute errors
    error_with_standardization = abs(option_price_with_standardization - theoretical_price)
    error_without_standardization = abs(option_price_without_standardization - theoretical_price)
    
    print(f"Theoretical price: {theoretical_price}")
    print(f"Price with standardization: {option_price_with_standardization}")
    print(f"Price without standardization: {option_price_without_standardization}")
    
    print(f"Absolute error with standardization: {error_with_standardization}")
    print(f"Absolute error without standardization: {error_without_standardization}")
    
    print(f"Ratio compared to theoretical with standardization: {error_with_standardization/theoretical_price}")
    print(f"Ratio compared to theoretical without standardization: {error_without_standardization/theoretical_price}")


compare_option_pricing()


# In[ ]:




