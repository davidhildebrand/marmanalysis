#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Useful guidance provided by:
# https://machinelearningmastery.com/bfgs-optimization-in-python/
# https://datascience.stackexchange.com/questions/69092/how-to-minimize-mean-square-error-using-python
# https://medium.com/analytics-vidhya/solving-our-python-basketball-model-d89edfb83f79


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def Rf(params, T):
    # based on Pattadkal etal Priebe 2022 bioRxiv 
    #   https://doi.org/10.1101/2022.06.23.497220
    Tpref = params[0] # rad, preferred direction
    beta = params[1] # 'tuning width factor'
    c = params[2] # baseline
    a1 = params[3] # peak 1 maxiumum amplitude
    a2 = params[4] # peak 2 maxiumum amplitude
    R = (a1 * np.exp(beta * np.cos(T - Tpref))) + \
        (a2 * np.exp(beta * np.cos(np.pi + T - Tpref))) + \
        c
    return R

def gf(params, T):
    # based on Fahey etal Tolias 2019 bioRxiv 
    #   https://doi.org/10.1101/745323
    Tpref = params[0] # rad, preferred direction
    w = params[1] # peak concentration or 'tuning width factor'
    g = np.exp(-w * (1 - np.cos(T - Tpref)))
    return g

def vf(params, T):
    a0 = params[2] # baseline
    a1 = params[3] # peak 1 maxiumum amplitude
    a2 = params[4] # peak 2 maxiumum amplitude
    v = a0 + (a1 * gf(params, T)) + (a2 * gf(params, T - np.pi))
    return v

def dsi_model(params, T):
    # r = Rf(params, T) # use Pattadkal etal Priebe 2022 
    r = vf(params, T) # use Fahey etal Tolias 2019
    return r

def dsi_objective(params, T, measRs):
    predRs = dsi_model(params, T)
    mse = np.square(np.subtract(predRs, measRs)).mean()
    return mse

def calculate_dsi(xs, ys):
    thetas = np.radians(xs)
    measRs = ys

    # params = [Tpref, w/beta, a0, a1, a2]
    guess_baseline = np.mean(measRs)
    guess_amplitude = np.percentile(measRs, 90)
    guess = [(np.pi / 2),
             np.radians(360 / 8), 
             guess_baseline, 
             guess_amplitude, 
             guess_amplitude]

    result = minimize(dsi_objective, guess, args=(thetas, measRs), method='L-BFGS-B')
    fit = result['x']
    Tpref = fit[0]
    #w_fit = fit[1]
    #a0_fit = fit[2]
    #a1_fit = fit[3]
    #a2_fit = fit[4]

    #plt.figure()
    #plt.plot(dsi_model(fit, np.radians(np.arange(0, 360))))
    #plt.scatter(np.degrees(thetas), measRs, s=4, facecolors='none', edgecolors='k')

    dT = thetas
    Rn = measRs + np.abs(np.min(measRs))
    dR = Rn
    # based on Pattadkal etal Priebe 2022 bioRxiv
    dsi = np.sqrt(np.sum(dR * np.sin(dT))**2 + np.sum(dR * np.cos(dT))**2) / np.sum(dR)
    # not sure how one would use it here, but 
    #   DSI can also be defined as (Rpref − Ropp)/(Rpref + Ropp)
    #   Ropp is the response in the direction opposite to the preferred direction (180)
    
    return dsi, Tpref
