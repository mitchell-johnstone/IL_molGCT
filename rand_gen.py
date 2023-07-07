# Adjusted from https://alpynepyano.github.io/healthyNumerics/posts/sampling_arbitrary_distributions_with_python.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_distrib1(xc, count_c):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(17,5))
        plt.plot(xc,count_c, ls='--', lw=1, c='b')
        wi = np.diff(xc)[0]*0.95
        plt.bar (xc, count_c, color='gold', width=wi, alpha=0.7, label='Histogram of data')
        plt.title('Data distribution of tokenlen', fontsize=25, fontweight='bold')
        plt.show()
    return

def plot_line(X,Y,x,y):
    with plt.style.context('fivethirtyeight'):
        fig, ax1 = plt.subplots(figsize=(17,5))
        ax1.plot(X,Y, 'mo-', lw=7, label='discrete CDF', ms=20)
        ax1.legend(loc=6, frameon=False)
        ax2 = ax1.twinx()
        ax2.plot(x,y, 'co-', lw=7, label='discrete PDF', ms=20)
        ax2.legend(loc=7, frameon=False)
        ax1.set_ylabel('CDF-axis');  ax2.set_ylabel('PDF-axis');
        plt.title('Tokenlen: CDF and PDF', fontsize=25, fontweight='bold')
        plt.show()

def plot_distrib3(xc, myPDF, X):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(17,5))
        width, ms = 0.5, 20
        plt.bar(xc, X, color='blue', width=width, label='resampled PDF')
        plt.plot(xc, np.zeros_like(X) ,color='magenta', ls='-',lw=13, alpha=0.6)
        plt.plot(xc, myPDF, 'co-', lw=7, label='discrete PDF', ms=ms, alpha=0.5)
        plt.title('Tokenlen sampling from data distribution', fontsize=25, fontweight='bold')
        plt.legend(loc='upper center', frameon=False)
        plt.show()

def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF>=a)-1

def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)

def tokenlen_gen_from_data_distribution(data, nBins, size):
    count_c, bins_c, = np.histogram(data, bins=nBins)
    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    # plot_distrib1(xc, myPDF)
    # plot_line(bins_c, myCDF, xc, myPDF)
    # plot_distrib3(xc, myPDF, X)

    return tokenlen_list

def rand_gen_from_data_distribution(data, size, nBins, cond_dim):
    H, edges = np.histogramdd(data.values, bins=(nBins[0], nBins[1], nBins[2], nBins[3], nBins[4]))
    P = H/len(data)
    P_flatten = P.reshape(-1)
    
    dxc_Temperature = np.diff(edges[0])[0]
    dxc_Pressure = np.diff(edges[1])[0]
    dxc_DynViscosity = np.diff(edges[2])[0]
    dxc_Density = np.diff(edges[3])[0]
    dxc_ElecConductivity = np.diff(edges[4])[0]
    
    xc_Temperature = edges[0][:-1] + 0.5 * dxc_Temperature
    xc_Pressure = edges[1][:-1] + 0.5 * dxc_Pressure
    xc_DynViscosity = edges[2][:-1] + 0.5 * dxc_DynViscosity
    xc_Density = edges[3][:-1] + 0.5 * dxc_Density
    xc_ElecConductivity = edges[4][:-1] + 0.5 * dxc_ElecConductivity

    samples_idx = np.random.choice(len(P_flatten), size=size, p=P_flatten)
    samples_idx = np.array(np.unravel_index(samples_idx, P.shape)).T

    samples = np.zeros_like(samples_idx, dtype=np.float64)

    for i in range(len(samples_idx)):
        samples[i] = [xc_Temperature[i][0], xc_Pressure[i][1], xc_DynViscosity[i][2], xc_Density[i][3], xc_ElecConductivity[i][4]]
    
    random_noise = np.random.uniform(low=-0.5, high=0.5, size=np.shape(samples))
    random_noise[:, 0] = random_noise[:, 0] * dxc_Temperature
    random_noise[:, 1] = random_noise[:, 1] * dxc_Pressure
    random_noise[:, 2] = random_noise[:, 2] * dxc_DynViscosity
    random_noise[:, 3] = random_noise[:, 3] * dxc_Density
    random_noise[:, 4] = random_noise[:, 4] * dxc_ElecConductivity

    samples = samples + random_noise

    return samples


