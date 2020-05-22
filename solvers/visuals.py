import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import pdb

def _clean_figure(ax):
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', nbins=5)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

def plot_learning_trace(trace,num_episodes,Sigma = 10):
    """
    Helper function to plot the trace of a learning policy
        1. Number of episodes vs Episode length -- Ideally should converge to the lowest
        2. Number of episodes vs Rewards -- Ideally should converge to the lowest possible
        3. Number of episodes vs Epsilon -- Should have a exponential decay behavior
        
    INPUTS:
    -------
        opc : Optimized Control Class 
    Outputs:
    --------
        Plots above mentioned plots as subplots 
    """
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1)
    ys = gaussian_filter1d(trace['lengths'], sigma=Sigma)
    ax1.plot(np.arange(num_episodes),ys)
    ax1.set_ylabel('Episode length')
    _clean_figure(ax1)
    
    ax2 = plt.subplot(1,3,2)
    ys = gaussian_filter1d(trace['rewards'], sigma=Sigma)
    ax2.plot(np.arange(num_episodes),ys)
    ax2.set_ylabel('Episode Rewards')
    _clean_figure(ax2)
    
    ax3 = plt.subplot(1,3,3)
    ys = gaussian_filter1d(np.asarray(trace['loss'])[:,1].T, sigma=Sigma)
    ax3.plot(np.asarray(trace['loss'])[:,0],ys)
    ax3.set_ylabel('Loss')    
    _clean_figure(ax3)
    
    fig.text(0.5, -0.04, 'Number of Episodes', ha='center', fontsize = 24)
    plt.subplots_adjust(wspace = 0.35)

    sns.despine()
    sns.set_context('paper')
    plt.tight_layout()
    
    ax = [ax1,ax2,ax3]
    
    return fig, ax