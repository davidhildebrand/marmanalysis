#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize as scipy_minimize


def response_at_theta(params, theta):
    # Based on Pattadkal et al Priebe 2022 bioRxiv
    #     https://doi.org/10.1101/2022.06.23.497220
    theta_pref = params[0]  # rad, preferred direction
    beta = params[1]  # 'tuning width factor'
    const_baseline = params[2]  # baseline
    a1 = params[3]  # peak 1 maxiumum amplitude
    a2 = params[4]  # peak 2 maxiumum amplitude
    resp = (a1 * np.exp(beta * np.cos(theta - theta_pref))) + \
        (a2 * np.exp(beta * np.cos(np.pi + theta - theta_pref))) + \
        const_baseline
    return resp


def gf(params, theta):
    # Based on Fahey et al Tolias 2019 bioRxiv.
    #   https://doi.org/10.1101/745323
    theta_pref = params[0]  # rad, preferred direction
    w = params[1]  # peak concentration or 'tuning width factor'
    g = np.exp(-w * (1 - np.cos(theta - theta_pref)))
    return g


def vf(params, theta):
    a0 = params[2]  # baseline
    a1 = params[3]  # peak 1 maxiumum amplitude
    a2 = params[4]  # peak 2 maxiumum amplitude
    v = a0 + (a1 * gf(params, theta)) + (a2 * gf(params, theta - np.pi))
    return v


def dsi_model(params, theta):
    resp = response_at_theta(params, theta)  # Pattadkal et al Priebe 2022
    # r = vf(params, theta)  # Fahey et al Tolias 2019
    return resp


def dsi_objective(params, theta, responses_measured):
    responses_predicted = dsi_model(params, theta)
    mse = np.square(np.subtract(responses_predicted, responses_measured)).mean()
    return mse


def calculate_dsi(xs, ys, unit='deg', plotting=False, debugging=False):
    if unit == 'deg':
        thetas = np.radians(xs)
    elif unit == 'rad':
        thetas = xs
    responses_measured = ys

    # params = [theta_pref, w/beta, a0, a1, a2]
    guess_baseline = np.mean(responses_measured)
    guess_amplitude = np.percentile(responses_measured, 90)
    guess = [(np.pi / 2),
             np.radians(360 / 8),
             guess_baseline,
             guess_amplitude,
             guess_amplitude]

    result = scipy_minimize(dsi_objective, guess, args=(thetas, responses_measured), method='L-BFGS-B')
    fit = result['x']
    theta_pref_f = fit[0]
    w_f = fit[1]
    a0_f = fit[2]
    a1_f = fit[3]
    a2_f = fit[4]

    # Based on Pattadkal et al Priebe 2022 bioRxiv
    #     "The location of the peak of this fitted tuning curve is used as the preferred direction of each cell."
    fit_curve = dsi_model(fit, np.radians(np.arange(0, 360)))
    max_peak_arg = fit_curve.argmax()
    # peak_locs = find_peaks(fit_curve, height=0)[0]
    # peaks = fit_curve[peak_locs]
    # max_peak_loc = peak_locs[peaks.argmax()]
    theta_pref = np.radians(max_peak_arg)

    if plotting:
        import matplotlib.pyplot as plt
        deg_symbol = u'\N{DEGREE SIGN}'

        plt.figure()
        plt.scatter(np.degrees(thetas), responses_measured, s=4, facecolors='none', edgecolors='k')
        plt.plot(dsi_model(fit, np.radians(np.arange(0, 360))))
        plt.axvline(np.degrees(theta_pref), color='m')
        ax = plt.gca()
        ax.set_xlabel('Direction (' + deg_symbol + ')', fontsize=8)
        ax.set_ylabel('dF/F', fontsize=8)
        # ax.set_xlim((0,360))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([d for d in range(0, 360, np.diff(np.degrees(thetas)).max().astype('int'))])
        # ax.set_xticklabels(['', 0, '', 2, ''])
        # plt.scatter(thetas, measRs, s=4, facecolors='none', edgecolors='k')
        # plt.plot(dsi_model(fit, np.arange(0, 2*np.pi))) #np.radians(np.arange(0, 360))))
        # plt.axvline(theta_pref, color='m')
        plt.show()

    dT = thetas
    Rn = responses_measured + np.abs(np.min(responses_measured))
    dR = Rn
    # DSI based on Pattadkal et al Priebe 2022 bioRxiv
    dsi = np.sqrt(np.sum(dR * np.sin(dT))**2 + np.sum(dR * np.cos(dT))**2) / np.sum(dR)
    if unit == 'deg':
        theta_pref = np.degrees(theta_pref)
        theta_pref_f = np.degrees(theta_pref_f)

    if debugging:
        print('calculate_dsi dsi={:.2f} theta_pref={} '.format(dsi, theta_pref) +
              'theta_pref_f={:.2f} w={:.2f} a0={:.2f} '.format(theta_pref_f, w_f, a0_f) +
              'a1={:.2f} a2={:.2f}'.format(a1_f, a2_f))

    return dsi, theta_pref
