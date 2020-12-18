# -----------------------------------------------------------
# Code free for public use, just acknowledge use
# Paul Chao, pchao@umich.edu
# December 18, 2020
# Data obtained at APS 2ID-BM
# Original data type: hdf (h5) file
#
# -----------------------------------------------------------

#Import packages
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import argparse

import matplotlib as mpl
import scipy.stats
from pathlib import Path
import h5py
import pandas as pd
import tomopy
from scipy import ndimage
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy import interpolate
import time

from math import log, e
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz

# Functions

# Import sinogram data
def load_fromh5(filepath, dir_structure, slice_num, strt_frm=0):
    """
    load_fromh5 will extract the sinogram from the h5 file 

    Output: the sinogram

    filepath: where the file is located in the system
    dir_structure: the h5 file directory structure
    slice_num: the slice where the singoram will be extracted
    strt_frm (optional): where the sinogram should begin
    """
    f = h5py.File(filepath, 'r')
    #["entry/data/data"]
    print(f[dir_structure].shape)
    end_frm = f[dir_structure].shape[0]
    sino = f[dir_structure][int(strt_frm):int(end_frm),int(slice_num),:] #For APS 2BM h5 file format
    return sino

def prepare_sinogram(sinogram, period=3000, downsamplescale=10, numLiq=3, keepLiq=True):
    """
    prepare_sinogram will normalize the sinogram using the initial liquid scans

    Output: Sinogram that has been normalized.

    sinogram: The sinogram data as a 2D array
    period: The number of oprojections for a 360 degree sample rotation in a tomographic scan
    downsamplescale: the slice where the singoram will be extracted
    numLiq: The number of liquid periods to be used to normalize 
    keepLiq: If the output should contain the liquid frames used for normalization
    """
    sinogram = tomopy.minus_log(sinogram)
    sino_dwn = downsample(sinogram, scalingfactor=downsamplescale)
    period = int(period) // int(downsamplescale)
    sino_dwn_subliq = sub_sino_liq(sino_dwn,size=1,numLiq=numLiq, period=period, keepLiq=keepLiq)
    return sino_dwn_subliq 

def downsample(data, scalingfactor=10):
    """
    downsample will reduce the data size by a integer factor by summing a NxN sized pixel area

    Output: the downsampled data

    data: The sinogram data as a 2D array
    scalingfactor: The amount the data will be scaled
    """
    data = data[0:(data.shape[0] // scalingfactor)*scalingfactor , 0:(data.shape[1] // scalingfactor)*scalingfactor];
    rows = scalingfactor
    cols = scalingfactor
    smaller = data.reshape(data.shape[0]//rows, rows,  data.shape[1]//cols, cols).sum(axis=1).sum(axis=2)
    return smaller

def sub_sino_liq(data, numLiq, size=20, period=3000, keepLiq=True):
    """
    sub_sino_liq will normalize the singram using the liquid periods

    Output: the normalized sinogram

    data: The sinogram data as a 2D array
    period: The number of oprojections for a 360 degree sample rotation in a tomographic scan
    size (optional): The size of the  median filter used to filter the liquid region
    numLiq: The number of liquid periods to be used to normalize 
    keepLiq: If the output should contain the liquid frames used for normalization
    """
    liquid = np.zeros((period, data.shape[1]), dtype=np.float32)
    counter = 0
    for i in range(numLiq):
        liquid = liquid + data[counter*period: counter*period+period, :]
        counter += 1
    liquid = liquid/numLiq
    liquid = ndimage.median_filter(liquid, size)

    subtract_fluid_data = []
    for iblock in np.arange(0, data.shape[0], period):
        try:
            subtract_fluid_data.append(data[iblock:iblock + period] - liquid)
        except ValueError:
            remaining = data.shape[0] - iblock
            subtract_fluid_data.append(data[iblock:iblock + remaining] - data[:remaining])
    subtract_fluid_data = np.concatenate(subtract_fluid_data)

    if keepLiq == True:
        return subtract_fluid_data
    else:
        return subtract_fluid_data[numLiq*period:]

def digitizetolevels(data, nLevels=256):
    """
    digitizetolevels will scale the data to discrete levels

    Output: the digitized data

    data: The sinogram data as a 2D array
    nLevels: The number of levels to descretize to
    """
    _min, _max = (data.min(), data.max())
    bins = np.linspace(_min, _max, nLevels)
    data_digitized = np.digitize(data,bins,right=True)
    return data_digitized

def save_sino(sinogram, fname, cmap='gray'):
    """
    save_sino will save the sinogram as an image

    Output: the saved image

    fname: The name of the saved image
    cmap: The image colormap
    """
    fig, ax = plt.subplots(figsize=(18, 2))
    ax.imshow(sinogram.T, cmap, interpolation='nearest', aspect='auto')
    fig.tight_layout()
    fig.savefig(fname)

def save_scree(pca, fname):
    """
    save_scree will save the scree plot as an image

    Output: the saved image

    pca: the pca object
    fname: The name of the saved image
    """
    var = pca.explained_variance_ratio_
    percent_variance = np.round(var* 100, decimals =2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(x= range(1,7), height=percent_variance)#, tick_label=list(principalDf.columns))
    plt.ylabel('Percentate of Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('PCA Scree Plot')
    plt.rcParams.update({'font.size': 22})
    fig.tight_layout()
    fig.savefig(fname)

def save_pcaplot(principalDf, fname):
    """
    save_pcaplot will save the PCA plot as an image

    Output: the saved image

    principalDf: the PCA dataframe
    fname: The name of the saved image
    """
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    pcaplot = ax.scatter(principalDf.loc[:,'principal component 1'],
               principalDf.loc[:,'principal component 2'],
               c = range(principalDf.shape[0]), s = 50)
    plt.colorbar(pcaplot, ax=ax)
    ax.grid()
    fig.tight_layout()
    fig.savefig(fname)

def save_fuzzyProbability(u, filtered_0, filtered_1, fname):
    """
    save_fuzzyProbability will save the fuzzy Probability plot as an image

    Output: the saved image

    u the fuzzy probability Nx2
    filtered_0 filtered probability
    filtered_1 filtered probability
    fname: The name of the saved image
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(u[0,:])
    plt.plot(u[1,:])
    #plt.plot(filtered_0)
    #plt.plot(filtered_1)
    fig.tight_layout()
    fig.savefig(fname)

# Various features to use, all optional
def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """
  n_labels = len(labels)
  if n_labels <= 1:
    return 0
  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <= 1:
    return 0
  ent = 0.
  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)
  return ent

def outlier_zscore_overall(array, threshold):
    mean_all = np.mean(array)*np.ones((1, array.shape[1]))
    stdev_all = np.std(array)*np.ones((1, array.shape[1]))

    outlier_count = np.zeros((array.shape[0],1))
    for col in np.arange(array.shape[0]):
        z_score = (array[col,:] - mean_all) / (stdev_all)
        outlier_count[col] = np.sum(z_score > threshold)
    return np.squeeze(outlier_count > threshold)

def numUnique(array):
    [unique, counts] = np.unique(array, return_counts=True)
    return counts.shape[0]

def numThreshold_ge(array, thresh):
    vals = (array>= np.ones(array.shape)*thresh).sum(axis=1);
    return vals

def numThreshold_le(array, thresh):
    vals = (array<= np.ones(array.shape)*thresh).sum(axis=1);
    return vals

def makeTarget(manualID, length):
    name = []
    for index in range(manualID):
        name.append('before?')
    for index in range(manualID,length):
        name.append('after?')
    return name

def maxDerivative(array):
    # go columnwise
    col_array = np.zeros((array.shape[0],1))
    for col in np.arange(0,array.shape[0]-1):
        current_col = array[col,:]
        next_col = array[col+1,:]
        diff = np.abs(next_col-current_col)
        col_array[col+1,:] = np.max(diff)
    return np.squeeze(col_array)

def maxDerivative_cols(array, distance=1):
    # go columnwise
    col_array = np.zeros((array.shape[0],1))
    for col in np.arange(0,array.shape[0]-distance):
        if col-distance < 0: #begining
            diff = np.abs(array[col,:].min()-array[col+distance,:].min())
            col_array[col,:] = diff #np.max(diff)
        elif col+distance > array.shape[0]: # end
            diff = np.abs(array[col,:].min()-array[col-distance,:].min())
            col_array[col,:] = diff #np.max(diff)
        else:
            diff = np.abs(array[col-distance,:].min()-array[col+distance,:].min())
            col_array[col,:] = diff #np.max(diff)
    return np.squeeze(col_array)

def argclosest(arr, K): 
     arr = np.asarray(arr) 
     idx = (np.abs(arr - K)).argmin() 
     return idx

# Function that encapsulates the algorithm described
def analyze_sinogram(sinogram, period, save=True):
    """
    analyze_sinogram will implement the algorithm

    Output: the saved image

    sinogram: the sinogram
    period: the number of projections for a 360 degree sample rotation
    save: Save the results
    """
    data = sinogram
    df = pd.DataFrame({
        "index" : np.arange(data.shape[0]),
        #"Sum" : data.sum(axis=1),
        #"Entropy"  : np.apply_along_axis(entropy2, 1, data),
        "Max" : np.apply_along_axis(np.max, 1, data),
        "Min" : np.apply_along_axis(np.min, 1, data),
        "Mean" : np.apply_along_axis(np.mean, 1, data),
        "Median" : np.apply_along_axis(np.median, 1, data),
        #"Q1" : np.quantile(data, .25, axis=1),
        #"Q3" : np.quantile(data, .75, axis=1),
        "Stdev" : np.apply_along_axis(np.std, 1, data),
        "Range" : np.apply_along_axis(np.ptp, 1, data),
        "Unique Values" : np.apply_along_axis(numUnique, 1, data),
        #"SNR" : np.apply_along_axis(np.mean, 1, data)**2/np.apply_along_axis(np.std, 1, data)**2,
        #"Vals<200" : numThreshold_le(data, 200),
        #"Vals<150" : numThreshold_le(data, 150),
        #"Vals>150" : numThreshold_ge(data, 150),
        "Vals<mean" : numThreshold_le(data, np.mean(data)),
        "Vals>mean" : numThreshold_ge(data, np.mean(data)),
        #"outlier KDE" : outlier_score(data, bandwidth=7, threshold = 1e-10),
        #"outliers zscore 1.5" : outlier_zscore_column(data, 1.5),
        #"z-score outlier" : outlier_zscore_column(data, 2),
        #"outliers zscore 2.5" : outlier_zscore_column(data, 2.5),
        #"outliers zscore 3" : outlier_zscore_column(data, 3),
        #"outliers zscore 3.5" : outlier_zscore_column(data, 3.5),
        #"max derivative" : maxDerivative(data),
        #"max derivative 5" : maxDerivative_cols(data, distance=5),
        #"max derivative 50" : maxDerivative_cols(data, distance=50),
        #"max derivative 150" : maxDerivative_cols(data, distance=150),
        #"max derivative 10%" : maxDerivative_cols(data, distance=10),
        #"max derivative 50%" : maxDerivative_cols(data, distance=100),
        #"max derivative period" : maxDerivative_cols(data, distance=300),
        #"max derivative 1000" : maxDerivative_cols(data, distance=1000),
        #"max change period" : maxNegChange_period(data, distance=period),
        #"target" : makeTarget(visual_change, data.shape[0])
    })

    features = list(df.columns[1:])
    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    #set up as new dataframe
    df_standard = pd.DataFrame(data=x,columns=features)

    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(x)

    if save: save_scree(pca, 'scree.png')

    principalDf = pd.DataFrame(data = principalComponents,
        columns = ['principal component 1', 'principal component 2', 'principal component 3',
        'principal component 4', 'principal component 5', 'principal component 6'])

    if save: save_pcaplot(principalDf, 'pcaplot.png')

    points = np.array(list(zip(principalDf.loc[:,'principal component 1'].values, principalDf.loc[:,'principal component 2'].values, df.loc[:,'index'].values)))
    points = StandardScaler().fit_transform(points)

    alldata = points.T
    ncenters = 2

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 1.8, error=0.00005, maxiter=1000, init=None)
    if int(period) % 2 == 0: windowsize = int(period+1)
    else: windowsize = int(period) #Make it odd for the savgol filter
    filtered_0 = savgol_filter(u[0,:], windowsize,1)
    filtered_1 = savgol_filter(u[1,:], windowsize,1)
    
    if save: save_fuzzyProbability(u, filtered_0, filtered_1, 'fuzzyprob.png')
    
    P = filtered_0 * filtered_1
    
    crit = np.argmax(P)
    
    print(' *** Clustering Results')
    print('Critical point: {}'.format(crit))
    print('Range of critical point (60% threshold): ({0}, {1})'.format(
        argclosest(P[:crit], 0.6*0.4), argclosest(P[crit:], 0.6*0.4) + crit ) )
    print('Range of critical point (70% threshold): ({0}, {1})'.format(
        argclosest(P[:crit], 0.7*0.3), argclosest(P[crit:], 0.7*0.3) + crit ) )
    print('Range of critical point (80% threshold): ({0}, {1})'.format(
        argclosest(P[:crit], 0.8*0.2), argclosest(P[crit:], 0.8*0.2) + crit ) )
    results = [crit, argclosest(P[:crit], 0.6*0.4), argclosest(P[crit:], 0.6*0.4) + crit ,
            argclosest(P[:crit], 0.7*0.3), argclosest(P[crit:], 0.7*0.3) + crit,
            argclosest(P[:crit], 0.8*0.2), argclosest(P[crit:], 0.8*0.2) + crit]
    
    return results

