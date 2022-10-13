print('Importing modules')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, chirp
import pickle
import glob
import os
import re

from sklearn.model_selection import train_test_split,GridSearchCV

from numpy import array
import tensorflow.keras
from tensorflow.keras.layers import Activation, Dropout, Dense
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

from tensorflow.keras.models import Sequential
#from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping

import sklearn
from sklearn import tree
from sklearn.linear_model import LinearRegression

from scipy.signal import hilbert
from scipy.signal import find_peaks
import matplotlib.mlab as mlab
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

print('all imported')

window_size=400
sample_rate=5
len_signal=window_size*sample_rate
n_samples=2000
n_files=10000

path = '/pnfs/iihe/radar/store/user/vlukic/twenty_seven_rec_root_files_n_1p78/voltage_time_traces/removed_10/'
path_time = '/pnfs/iihe/radar/store/user/vlukic/twenty_seven_rec_root_files_n_1p78/voltage_time_traces/'

data_rec1=np.load(path+'data_rec1_10000.npy')
data_rec2=np.load(path+'data_rec2_10000.npy')
data_rec3=np.load(path+'data_rec3_10000.npy')
data_rec4=np.load(path+'data_rec4_10000.npy')
data_rec5=np.load(path+'data_rec5_10000.npy')
data_rec6=np.load(path+'data_rec6_10000.npy')
data_rec7=np.load(path+'data_rec7_10000.npy')
data_rec8=np.load(path+'data_rec8_10000.npy')
data_rec9=np.load(path+'data_rec9_10000.npy')
data_rec10=np.load(path+'data_rec10_10000.npy')
data_rec11=np.load(path+'data_rec11_10000.npy')
data_rec12=np.load(path+'data_rec12_10000.npy')
data_rec13=np.load(path+'data_rec13_10000.npy')
data_rec14=np.load(path+'data_rec14_10000.npy')
data_rec15=np.load(path+'data_rec15_10000.npy')
data_rec16=np.load(path+'data_rec16_10000.npy')
data_rec17=np.load(path+'data_rec17_10000.npy')
data_rec18=np.load(path+'data_rec18_10000.npy')
data_rec19=np.load(path+'data_rec19_10000.npy')
data_rec20=np.load(path+'data_rec20_10000.npy')
data_rec21=np.load(path+'data_rec21_10000.npy')
data_rec22=np.load(path+'data_rec22_10000.npy')
data_rec23=np.load(path+'data_rec23_10000.npy')
data_rec24=np.load(path+'data_rec24_10000.npy')
data_rec25=np.load(path+'data_rec25_10000.npy')
data_rec26=np.load(path+'data_rec26_10000.npy')
data_rec27=np.load(path+'data_rec27_10000.npy')

data_receiver_1=np.load(path_time+'data_receiver_1_10000_time.npy')
data_receiver_2=np.load(path_time+'data_receiver_2_10000_time.npy')
data_receiver_3=np.load(path_time+'data_receiver_3_10000_time.npy')
data_receiver_4=np.load(path_time+'data_receiver_4_10000_time.npy')
data_receiver_5=np.load(path_time+'data_receiver_5_10000_time.npy')
data_receiver_6=np.load(path_time+'data_receiver_6_10000_time.npy')
data_receiver_7=np.load(path_time+'data_receiver_7_10000_time.npy')
data_receiver_8=np.load(path_time+'data_receiver_8_10000_time.npy')
data_receiver_9=np.load(path_time+'data_receiver_9_10000_time.npy')
data_receiver_10=np.load(path_time+'data_receiver_10_10000_time.npy')
data_receiver_11=np.load(path_time+'data_receiver_11_10000_time.npy')
data_receiver_12=np.load(path_time+'data_receiver_12_10000_time.npy')
data_receiver_13=np.load(path_time+'data_receiver_13_10000_time.npy')
data_receiver_14=np.load(path_time+'data_receiver_14_10000_time.npy')
data_receiver_15=np.load(path_time+'data_receiver_15_10000_time.npy')
data_receiver_16=np.load(path_time+'data_receiver_16_10000_time.npy')
data_receiver_17=np.load(path_time+'data_receiver_17_10000_time.npy')
data_receiver_18=np.load(path_time+'data_receiver_18_10000_time.npy')
data_receiver_19=np.load(path_time+'data_receiver_19_10000_time.npy')
data_receiver_20=np.load(path_time+'data_receiver_20_10000_time.npy')
data_receiver_21=np.load(path_time+'data_receiver_21_10000_time.npy')
data_receiver_22=np.load(path_time+'data_receiver_22_10000_time.npy')
data_receiver_23=np.load(path_time+'data_receiver_23_10000_time.npy')
data_receiver_24=np.load(path_time+'data_receiver_24_10000_time.npy')
data_receiver_25=np.load(path_time+'data_receiver_25_10000_time.npy')
data_receiver_26=np.load(path_time+'data_receiver_26_10000_time.npy')
data_receiver_27=np.load(path_time+'data_receiver_27_10000_time.npy')

data_all=np.stack((data_rec1,data_rec2,data_rec3,data_rec4,data_rec5,data_rec6,data_rec7,data_rec8,data_rec9,data_rec10,data_rec11,data_rec12,data_rec13,data_rec14,data_rec15,data_rec16,data_rec17,data_rec18,data_rec19,data_rec20,data_rec21,data_rec22,data_rec23,data_rec24,data_rec25,data_rec26,data_rec27))

data_all_time=np.stack((data_receiver_1,data_receiver_2,data_receiver_3,data_receiver_4,data_receiver_5,data_receiver_6,data_receiver_7,data_receiver_8,data_receiver_9,data_receiver_10,data_receiver_11,data_receiver_12,data_receiver_13,data_receiver_14,data_receiver_15,data_receiver_16,data_receiver_17,data_receiver_18,data_receiver_19,data_receiver_20,data_receiver_21,data_receiver_22,data_receiver_23,data_receiver_24,data_receiver_25,data_receiver_26,data_receiver_27))

###### Generate the signal properties

def intensity(data):
    """
    Calculates the intensity at each receiver, as defined by the sum of the voltage-time trace (at each time point) squared
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate)
    
    Args: data (3-d numpy array)
    
    Returns: numpy array of intensity values at each receiver
    
    """
    int_list=[]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            int_list.append(np.sum(data[j][i]**2))
    int_array=np.asarray(int_list)
    return int_array.reshape(data.shape[0],data.shape[1])

def arrival_rec(data,thres_arr_time,time_vals):
    """
    Calculates the arrival time at each receiver, by finding the time at which the absolute value of the hilbert envelope first exceeds some threshold value, that corresponds
    to some fraction of the maximum of the hilbert envelope
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate), a threshold for the arrival time,
    and a numpy array of time values
    
    Args: data (3-d numpy array)
          thres_arr_time (float)
          time_vals (numpy array)
    
    Returns: numpy array of arrival times at each receiver
    
    """
    arrival_recs=[]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            time_vec=np.linspace(time_vals[j][i],time_vals[j][i]+400,2000)
            arrival_recs.append(time_vec[np.where(np.abs(hilbert(data[j][i]))>thres_arr_time*np.max(np.abs(hilbert(data[j][i]))))[0][0]])
    arrival_recs=np.asarray(arrival_recs)
    return arrival_recs.reshape(data.shape[0],data.shape[1])

def peakFreq(data,sample_rate):
    """
    Calculates the peak frequency at each receiver. Finds the frequency at which the fft amplitude is a maximum.
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate), a sample rate for the signal (the
    number of times the signal is sampled per nanosecond)
    
    Args: data (3-d numpy array)
          sample_rate (int)
    
    Returns: numpy array of peak frequencies
    
    """
    T=1/sample_rate
    n=data.shape[2] # the length of each signal
    xf = fftfreq(n, T)[:n//2]
    peakFreq_list=[]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            peakFreq_list.append(xf[np.where(abs(fft(data[j][i])[0:n//2])==np.sort(abs(fft(data[j][i])[0:n//2]))[len(fft(data[j][i])[0:n//2])-1])][0])
    peakFreq=np.asarray(peakFreq_list)
    return peakFreq.reshape(data.shape[0],data.shape[1])

def rise_fall_time(data,time_vals,thres1,thres2):
    """
    Calculates the rise and fall time at each receiver. Rise time is calculated as the difference of time between the max of the hilbert envelope and the first threshold
    crossing (as defined by thres1). Fall time is calculated as the difference of time between the last threshold crossing and the max of the hilbert envelope (as defined by thres2)
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate), a numpy array of time values and
    a threshold from which the proportion of the maximum of the signal is calculated
    
    Args: data (3-d numpy array)
          time_vals (numpy array)
          thres1 (float)
          thres2 (float)
    
    Returns: numpy array of rise times and fall times
    
    """
    rise_time=[]
    fall_time=[]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            time_vec=np.linspace(time_vals[j][i],time_vals[j][i]+400,2000)
            rise_time.append(np.round((time_vec[np.where(np.abs(hilbert(data[j][i]))==np.max(np.abs(hilbert(data[j][i]))))[0]]-time_vec[np.where(np.abs(hilbert(data[j][i]))>thres1*np.max(np.abs(hilbert(data[j][i]))))[0][0]])[0],3))
            fall_time.append(np.round((time_vec[np.where(np.abs(hilbert(data[j][i]))>thres2*np.max(np.abs(hilbert(data[j][i]))))[0][len(np.where(np.abs(hilbert(data[j][i]))>thres2*np.max(np.abs(hilbert(data[j][i]))))[0])-1]]-time_vec[np.where(np.abs(hilbert(data[j][i]))==np.max(np.abs(hilbert(data[j][i]))))[0]])[0],3))
    rise_time=np.asarray(rise_time)
    fall_time=np.asarray(fall_time)
    return rise_time.reshape(data.shape[0],data.shape[1]), fall_time.reshape(data.shape[0],data.shape[1])
           
def bandwidth_recs(data,sample_rate,thres_bandwidth):
    """
    Calculates the bandwidth at each receiver. This is defined as the range of frequency above which the amplitude of the fft is greater than some threshold value as defined
    by the threshold
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate), a sample rate for the signal (the
    number of times the signal is sampled per nanosecond) and a threshold value that is multipled by the maximum of the fft, above which the range of frequencies that exceeds this
    threshold is calculated from
    
    Args: data (3-d numpy array)
          sample_rate (float)
          thres_bandwidth (float)
    
    Returns: numpy array of bandwidths
    
    """
    freq_vec_thres=[]
    T=1/sample_rate
    n=data.shape[2] # the length of each signal
    xf = fftfreq(n, T)[:n//2]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            freq_vec_thres.append(xf[np.where(np.abs(fft(data[j][i])[0:n//2])>thres_bandwidth*np.max((np.abs(fft(data[j][i])[0:n//2]))))[0]][len(np.where(np.abs(fft(data[j][i])[0:n//2])>thres_bandwidth*np.max((np.abs(fft(data[j][i])[0:n//2]))))[0])-1]-xf[np.where(np.abs(fft(data[j][i])[0:n//2])>thres_bandwidth*np.max((np.abs(fft(data[j][i])[0:n//2]))))[0]][0])
    freq_vec_thres=np.asarray(freq_vec_thres)
    return freq_vec_thres.reshape(data.shape[0],data.shape[1])
           
def fft_amps(data,sample_rate,tf):
    """
    Calculates the maximum of the fft amplitude, the fft amplitude at half, at and twice the transmit frequency tf, for each receiver.
    
    Takes in a 3-d numpy array of data in the form (# receivers, # data points (or runs), length of signal = window_size*sample_rate), a sample rate for the signal (the
    number of times the signal is sampled per nanosecond) and a threshold value that is multipled by the maximum of the fft, above which the range of frequencies that exceeds this
    threshold is calculated from
    
    Args: data (3-d numpy array)
          sample_rate (float)
          tf (transmit frequency)
    
    Returns: numpy arrays of the max fft amplitude, the fft amplitude at half, at and twice the transmit frequency
    
    """
    max_amp_freq=[]
    amp_twice_trans_freq=[]
    amp_trans_freq=[]
    amp_half_trans_freq=[]
    T=1/sample_rate
    n=data.shape[2] # the length of each signal
    xf = fftfreq(n, T)[:n//2]
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            max_amp_freq.append(np.max(2.0/n*np.abs(fft(data[j][i])[0:n//2])))
            #amp_half_trans_freq.append(np.abs(fft(data[j][i])[np.where(xf==0.5*tf)[0]])[0])
            amp_half_trans_freq.append(np.abs(fft(data[j][i])[np.where(np.abs((0.5*tf-xf))==np.min(np.abs((0.5*tf-xf))))[0]][0]))
            #xf[np.where(np.abs((0.5*tf-xf))==np.min(np.abs((0.5*tf-xf))))[0][0]]
            #amp_trans_freq.append(np.abs(fft(data[j][i])[np.where(xf==tf)[0]])[0])
            amp_trans_freq.append(np.abs(fft(data[j][i])[np.where(np.abs((tf-xf))==np.min(np.abs((tf-xf))))[0]][0]))
            #amp_twice_trans_freq.append(np.abs(fft(data[j][i])[np.where(xf==2*tf)[0]])[0])
            amp_twice_trans_freq.append(np.abs(fft(data[j][i])[np.where(np.abs((2*tf-xf))==np.min(np.abs((2*tf-xf))))[0]][0]))
    max_amp_freq=np.asarray(max_amp_freq)
    amp_half_trans_freq=np.asarray(amp_half_trans_freq)
    amp_trans_freq=np.asarray(amp_trans_freq)
    amp_twice_trans_freq=np.asarray(amp_twice_trans_freq)
    return max_amp_freq.reshape(data.shape[0],data.shape[1]),amp_half_trans_freq.reshape(data.shape[0],data.shape[1]),amp_trans_freq.reshape(data.shape[0],data.shape[1]),amp_twice_trans_freq.reshape(data.shape[0],data.shape[1])
        
data=data_all
       
int_array=intensity(data)
thres_arr_time=0.3
arrival_at_recs=arrival_rec(data,thres_arr_time,data_all_time)
peakfreqs=peakFreq(data,sample_rate)
thres1=0.3
thres2=0.3
rise_time,fall_time=rise_fall_time(data,data_all_time,thres1,thres2)
thres_bandwidth=0.05
bandwidths=bandwidth_recs(data,sample_rate,thres_bandwidth)
max_amp_freqs,half_amp_freqs,at_amp_freqs,twice_amp_freqs=fft_amps(data,sample_rate,0.05)

#### Directions and positions

print('Reading in positions')

pos=pd.read_csv('/user/vlukic/software/geant4/geant4.10.06.p03/nrt_build2/twenty_seven_rec_macros_n_1p78/positions_27_recs_copy.txt', skiprows=0,delim_whitespace=True,header=None)

xpos=pos[0]
ypos=pos[1]
zpos=pos[2]

print('Reading in energy')

energy=pd.read_csv('/user/vlukic/software/geant4/geant4.10.06.p03/nrt_build2/twenty_seven_rec_macros_n_1p78/energy_27_recs_copy.txt', skiprows=0,delim_whitespace=True,header=None)

print('Reading in directions')

dirs=pd.read_csv('/user/vlukic/software/geant4/geant4.10.06.p03/nrt_build2/twenty_seven_rec_macros_n_1p78/directions_27_recs_copy.txt', skiprows=0,delim_whitespace=True,header=None)

xdir=dirs[0]/np.sqrt(dirs[0]**2+dirs[1]**2+dirs[2]**2)
ydir=dirs[1]/np.sqrt(dirs[0]**2+dirs[1]**2+dirs[2]**2)
zdir=dirs[2]/np.sqrt(dirs[0]**2+dirs[1]**2+dirs[2]**2)

#### Positions of transmitter and receivers

Tx=[0,0,-0]
R1x= [250, 0, -20]
R2x= [250, 0, 0]
R3x= [250, 0, 20]
R4x= [191.511, 160.697, -20]
R5x= [191.511, 160.697,   0]
R6x= [191.511, 160.697,  20]
R7x= [43.412, 246.202, -20]
R8x= [43.412, 246.202,   0]
R9x= [43.412, 246.202,  20]
R10x= [-125, 216.506, -20]
R11x= [-125, 216.506,   0]
R12x= [-125, 216.506,  20]
R13x= [-234.923, 85.505, -20]
R14x= [-234.923, 85.505,   0]
R15x= [-234.923, 85.505,  20]
R16x= [-234.923,  -85.505,   -20]
R17x= [-234.923,  -85.505,   0]
R18x= [-234.923,  -85.505,   20]
R19x= [-125,  -216.506,  -20]
R20x= [-125,  -216.506,    0]
R21x= [-125,  -216.506,   20]
R22x= [43.412, -246.202, -20]
R23x= [43.412, -246.202,  0]
R24x= [43.412, -246.202,  20]
R25x= [191.511, -160.697,  -20]
R26x= [191.511, -160.697,  0]
R27x= [191.511, -160.697,  20]

Txc=np.sqrt((Tx[0]-xpos)**2+(Tx[1]-ypos)**2+(Tx[2]-zpos)**2)
R1xc=np.sqrt((R1x[0]-xpos)**2+(R1x[1]-ypos)**2+(R1x[2]-zpos)**2)
R2xc=np.sqrt((R2x[0]-xpos)**2+(R2x[1]-ypos)**2+(R2x[2]-zpos)**2)
R3xc=np.sqrt((R3x[0]-xpos)**2+(R3x[1]-ypos)**2+(R3x[2]-zpos)**2)
R4xc=np.sqrt((R4x[0]-xpos)**2+(R4x[1]-ypos)**2+(R4x[2]-zpos)**2)
R5xc=np.sqrt((R5x[0]-xpos)**2+(R5x[1]-ypos)**2+(R5x[2]-zpos)**2)
R6xc=np.sqrt((R6x[0]-xpos)**2+(R6x[1]-ypos)**2+(R6x[2]-zpos)**2)
R7xc=np.sqrt((R7x[0]-xpos)**2+(R7x[1]-ypos)**2+(R7x[2]-zpos)**2)
R8xc=np.sqrt((R8x[0]-xpos)**2+(R8x[1]-ypos)**2+(R8x[2]-zpos)**2)
R9xc=np.sqrt((R9x[0]-xpos)**2+(R9x[1]-ypos)**2+(R9x[2]-zpos)**2)
R10xc=np.sqrt((R10x[0]-xpos)**2+(R10x[1]-ypos)**2+(R10x[2]-zpos)**2)
R11xc=np.sqrt((R11x[0]-xpos)**2+(R11x[1]-ypos)**2+(R11x[2]-zpos)**2)
R12xc=np.sqrt((R12x[0]-xpos)**2+(R12x[1]-ypos)**2+(R12x[2]-zpos)**2)
R13xc=np.sqrt((R13x[0]-xpos)**2+(R13x[1]-ypos)**2+(R13x[2]-zpos)**2)
R14xc=np.sqrt((R14x[0]-xpos)**2+(R14x[1]-ypos)**2+(R14x[2]-zpos)**2)
R15xc=np.sqrt((R15x[0]-xpos)**2+(R15x[1]-ypos)**2+(R15x[2]-zpos)**2)
R16xc=np.sqrt((R16x[0]-xpos)**2+(R16x[1]-ypos)**2+(R16x[2]-zpos)**2)
R17xc=np.sqrt((R17x[0]-xpos)**2+(R17x[1]-ypos)**2+(R17x[2]-zpos)**2)
R18xc=np.sqrt((R18x[0]-xpos)**2+(R18x[1]-ypos)**2+(R18x[2]-zpos)**2)
R19xc=np.sqrt((R19x[0]-xpos)**2+(R19x[1]-ypos)**2+(R19x[2]-zpos)**2)
R20xc=np.sqrt((R20x[0]-xpos)**2+(R20x[1]-ypos)**2+(R20x[2]-zpos)**2)
R21xc=np.sqrt((R21x[0]-xpos)**2+(R21x[1]-ypos)**2+(R21x[2]-zpos)**2)
R22xc=np.sqrt((R22x[0]-xpos)**2+(R22x[1]-ypos)**2+(R22x[2]-zpos)**2)
R23xc=np.sqrt((R23x[0]-xpos)**2+(R23x[1]-ypos)**2+(R23x[2]-zpos)**2)
R24xc=np.sqrt((R24x[0]-xpos)**2+(R24x[1]-ypos)**2+(R24x[2]-zpos)**2)
R25xc=np.sqrt((R25x[0]-xpos)**2+(R25x[1]-ypos)**2+(R25x[2]-zpos)**2)
R26xc=np.sqrt((R26x[0]-xpos)**2+(R26x[1]-ypos)**2+(R26x[2]-zpos)**2)
R27xc=np.sqrt((R27x[0]-xpos)**2+(R27x[1]-ypos)**2+(R27x[2]-zpos)**2)

theta_Txc=np.arccos((Tx[2]-zpos)/Txc)*(180/np.pi)
theta_R1xc=np.arccos((R1x[2]-zpos)/R1xc)*(180/np.pi)
theta_R2xc=np.arccos((R2x[2]-zpos)/R2xc)*(180/np.pi)
theta_R3xc=np.arccos((R3x[2]-zpos)/R3xc)*(180/np.pi)
theta_R4xc=np.arccos((R4x[2]-zpos)/R4xc)*(180/np.pi)
theta_R5xc=np.arccos((R5x[2]-zpos)/R5xc)*(180/np.pi)
theta_R6xc=np.arccos((R6x[2]-zpos)/R6xc)*(180/np.pi)
theta_R7xc=np.arccos((R7x[2]-zpos)/R7xc)*(180/np.pi)
theta_R8xc=np.arccos((R8x[2]-zpos)/R8xc)*(180/np.pi)
theta_R9xc=np.arccos((R9x[2]-zpos)/R9xc)*(180/np.pi)
theta_R10xc=np.arccos((R10x[2]-zpos)/R10xc)*(180/np.pi)
theta_R11xc=np.arccos((R11x[2]-zpos)/R11xc)*(180/np.pi)
theta_R12xc=np.arccos((R12x[2]-zpos)/R12xc)*(180/np.pi)
theta_R13xc=np.arccos((R13x[2]-zpos)/R13xc)*(180/np.pi)
theta_R14xc=np.arccos((R14x[2]-zpos)/R14xc)*(180/np.pi)
theta_R15xc=np.arccos((R15x[2]-zpos)/R15xc)*(180/np.pi)
theta_R16xc=np.arccos((R16x[2]-zpos)/R16xc)*(180/np.pi)
theta_R17xc=np.arccos((R17x[2]-zpos)/R17xc)*(180/np.pi)
theta_R18xc=np.arccos((R18x[2]-zpos)/R18xc)*(180/np.pi)
theta_R19xc=np.arccos((R19x[2]-zpos)/R19xc)*(180/np.pi)
theta_R20xc=np.arccos((R20x[2]-zpos)/R20xc)*(180/np.pi)
theta_R21xc=np.arccos((R21x[2]-zpos)/R21xc)*(180/np.pi)
theta_R22xc=np.arccos((R22x[2]-zpos)/R22xc)*(180/np.pi)
theta_R23xc=np.arccos((R23x[2]-zpos)/R23xc)*(180/np.pi)
theta_R24xc=np.arccos((R24x[2]-zpos)/R24xc)*(180/np.pi)
theta_R25xc=np.arccos((R25x[2]-zpos)/R25xc)*(180/np.pi)
theta_R26xc=np.arccos((R26x[2]-zpos)/R26xc)*(180/np.pi)
theta_R27xc=np.arccos((R27x[2]-zpos)/R27xc)*(180/np.pi)

def phi_ang(x,y,x_c,y_c):
    if ((x-x_c)>0):
        phi=np.arctan((y-y_c)/(x-x_c))
    elif ((x-x_c)<0 and (y-y_c)>=0):
        phi=np.arctan((y-y_c)/(x-x_c))+np.pi
    elif ((x-x_c)<0 and (y-y_c)<0):
        phi=np.arctan((y-y_c)/(x-x_c))-np.pi
    return phi*(180/np.pi)

df_list=[]

for i in range(0,data.shape[0]):
    df_list.append(np.asarray([int_array[i],arrival_at_recs[i],peakfreqs[i],rise_time[i],fall_time[i],bandwidths[i],max_amp_freqs[i],half_amp_freqs[i],at_amp_freqs[i],twice_amp_freqs[i]]))

l = np.vstack(df_list)

all_df=pd.DataFrame(l.T)

### Exclude the true positions

#all_df['xpos']=xpos
#all_df['ypos']=ypos
#all_df['zpos']=zpos

### Include only the relative positions with respect to the receivers and transmitter

all_df['Txc']=Txc
all_df['R1xc']=R1xc
all_df['R2xc']=R2xc
all_df['R3xc']=R3xc
all_df['R4xc']=R4xc
all_df['R5xc']=R5xc
all_df['R6xc']=R6xc
all_df['R7xc']=R7xc
all_df['R8xc']=R8xc
all_df['R9xc']=R9xc
all_df['R10xc']=R10xc
all_df['R11xc']=R11xc
all_df['R12xc']=R12xc
all_df['R13xc']=R13xc
all_df['R14xc']=R14xc
all_df['R15xc']=R15xc
all_df['R16xc']=R16xc
all_df['R17xc']=R17xc
all_df['R18xc']=R18xc
all_df['R19xc']=R19xc
all_df['R20xc']=R20xc
all_df['R21xc']=R21xc
all_df['R22xc']=R22xc
all_df['R23xc']=R23xc
all_df['R24xc']=R24xc
all_df['R25xc']=R25xc
all_df['R26xc']=R26xc
all_df['R27xc']=R27xc

all_df['theta_Txc']=theta_Txc
all_df['theta_R1xc']=theta_R1xc
all_df['theta_R2xc']=theta_R2xc
all_df['theta_R3xc']=theta_R3xc
all_df['theta_R4xc']=theta_R4xc
all_df['theta_R5xc']=theta_R5xc
all_df['theta_R6xc']=theta_R6xc
all_df['theta_R7xc']=theta_R7xc
all_df['theta_R8xc']=theta_R8xc
all_df['theta_R9xc']=theta_R9xc
all_df['theta_R10xc']=theta_R10xc
all_df['theta_R11xc']=theta_R11xc
all_df['theta_R12xc']=theta_R12xc
all_df['theta_R13xc']=theta_R13xc
all_df['theta_R14xc']=theta_R14xc
all_df['theta_R15xc']=theta_R15xc
all_df['theta_R16xc']=theta_R16xc
all_df['theta_R17xc']=theta_R17xc
all_df['theta_R18xc']=theta_R18xc
all_df['theta_R19xc']=theta_R19xc
all_df['theta_R20xc']=theta_R20xc
all_df['theta_R21xc']=theta_R21xc
all_df['theta_R22xc']=theta_R22xc
all_df['theta_R23xc']=theta_R23xc
all_df['theta_R24xc']=theta_R24xc
all_df['theta_R25xc']=theta_R25xc
all_df['theta_R26xc']=theta_R26xc
all_df['theta_R27xc']=theta_R27xc

phi_Txc=[]
phi_R1xc=[]
phi_R2xc=[]
phi_R3xc=[]
phi_R4xc=[]
phi_R5xc=[]
phi_R6xc=[]
phi_R7xc=[]
phi_R8xc=[]
phi_R9xc=[]
phi_R10xc=[]
phi_R11xc=[]
phi_R12xc=[]
phi_R13xc=[]
phi_R14xc=[]
phi_R15xc=[]
phi_R16xc=[]
phi_R17xc=[]
phi_R18xc=[]
phi_R19xc=[]
phi_R20xc=[]
phi_R21xc=[]
phi_R22xc=[]
phi_R23xc=[]
phi_R24xc=[]
phi_R25xc=[]
phi_R26xc=[]
phi_R27xc=[]

#for i in range(0,len(xpos)):
for i in np.asarray(all_df.index):
    print(i)
    phi_Txc.append(phi_ang(Tx[0],Tx[1],xpos[i],ypos[i]))
    phi_R1xc.append(phi_ang(R1x[0],R1x[1],xpos[i],ypos[i]))
    phi_R2xc.append(phi_ang(R2x[0],R2x[1],xpos[i],ypos[i]))
    phi_R3xc.append(phi_ang(R3x[0],R3x[1],xpos[i],ypos[i]))
    phi_R4xc.append(phi_ang(R4x[0],R4x[1],xpos[i],ypos[i]))
    phi_R5xc.append(phi_ang(R5x[0],R5x[1],xpos[i],ypos[i]))
    phi_R6xc.append(phi_ang(R6x[0],R6x[1],xpos[i],ypos[i]))
    phi_R7xc.append(phi_ang(R7x[0],R7x[1],xpos[i],ypos[i]))
    phi_R8xc.append(phi_ang(R8x[0],R8x[1],xpos[i],ypos[i]))
    phi_R9xc.append(phi_ang(R9x[0],R9x[1],xpos[i],ypos[i]))
    phi_R10xc.append(phi_ang(R10x[0],R10x[1],xpos[i],ypos[i]))
    phi_R11xc.append(phi_ang(R11x[0],R11x[1],xpos[i],ypos[i]))
    phi_R12xc.append(phi_ang(R12x[0],R12x[1],xpos[i],ypos[i]))
    phi_R13xc.append(phi_ang(R13x[0],R13x[1],xpos[i],ypos[i]))
    phi_R14xc.append(phi_ang(R14x[0],R14x[1],xpos[i],ypos[i]))
    phi_R15xc.append(phi_ang(R15x[0],R15x[1],xpos[i],ypos[i]))
    phi_R16xc.append(phi_ang(R16x[0],R16x[1],xpos[i],ypos[i]))
    phi_R17xc.append(phi_ang(R17x[0],R17x[1],xpos[i],ypos[i]))
    phi_R18xc.append(phi_ang(R18x[0],R18x[1],xpos[i],ypos[i]))
    phi_R19xc.append(phi_ang(R19x[0],R19x[1],xpos[i],ypos[i]))
    phi_R20xc.append(phi_ang(R20x[0],R20x[1],xpos[i],ypos[i]))
    phi_R21xc.append(phi_ang(R21x[0],R21x[1],xpos[i],ypos[i]))
    phi_R22xc.append(phi_ang(R22x[0],R22x[1],xpos[i],ypos[i]))
    phi_R23xc.append(phi_ang(R23x[0],R23x[1],xpos[i],ypos[i]))
    phi_R24xc.append(phi_ang(R24x[0],R24x[1],xpos[i],ypos[i]))
    phi_R25xc.append(phi_ang(R25x[0],R25x[1],xpos[i],ypos[i]))
    phi_R26xc.append(phi_ang(R26x[0],R26x[1],xpos[i],ypos[i]))
    phi_R27xc.append(phi_ang(R27x[0],R27x[1],xpos[i],ypos[i]))

all_df['phi_Txc']=phi_Txc
all_df['phi_R1xc']=phi_R1xc
all_df['phi_R2xc']=phi_R2xc
all_df['phi_R3xc']=phi_R3xc
all_df['phi_R4xc']=phi_R4xc
all_df['phi_R5xc']=phi_R5xc
all_df['phi_R6xc']=phi_R6xc
all_df['phi_R7xc']=phi_R7xc
all_df['phi_R8xc']=phi_R8xc
all_df['phi_R9xc']=phi_R9xc
all_df['phi_R10xc']=phi_R10xc
all_df['phi_R11xc']=phi_R11xc
all_df['phi_R12xc']=phi_R12xc
all_df['phi_R13xc']=phi_R13xc
all_df['phi_R14xc']=phi_R14xc
all_df['phi_R15xc']=phi_R15xc
all_df['phi_R16xc']=phi_R16xc
all_df['phi_R17xc']=phi_R17xc
all_df['phi_R18xc']=phi_R18xc
all_df['phi_R19xc']=phi_R19xc
all_df['phi_R20xc']=phi_R20xc
all_df['phi_R21xc']=phi_R21xc
all_df['phi_R22xc']=phi_R22xc
all_df['phi_R23xc']=phi_R23xc
all_df['phi_R24xc']=phi_R24xc
all_df['phi_R25xc']=phi_R25xc
all_df['phi_R26xc']=phi_R26xc
all_df['phi_R27xc']=phi_R27xc

all_df['xdir']=xdir
all_df['ydir']=ydir
all_df['zdir']=zdir

all_df.to_csv(path+'all_df_10000_n1p78_27_recs_relative_position_energy_included.csv', index=False)

