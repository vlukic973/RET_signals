import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, chirp
import pickle
import glob
import os
import re
import random
        
import sklearn
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
import tensorflow
import tensorflow.keras
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
        
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    Sorting in human order
    
    Returns: List of file names sorted in natural order
    """
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
def read_signal_files(path,files):
    """
    Reads in a set of signal files from all receivers available as specified eg by '*_numbers_only_real_v2.txt' and sorts them in natural order.
    The path of the file location must also be specified
    
    Args: path (str)
          files (str)
          
    Returns: List of files sorted in natural order and list of data with all voltage-time traces from each receiver from receiver 1 to receiver n
    """
    file_list = glob.glob(path + files)
    file_list.sort(key=natural_keys)
    data = []
    for file_path in file_list:
        print('Reading in '+file_path)
        try:
            data.append(np.genfromtxt(file_path, delimiter='\n'))
        except Exception:
            pass
    return file_list, data
    
def pad_data(data,sample_size):
    """
    Compares the expected size of the signal to the true size. Pads the input data in case the signals are slightly truncated.
    
    Args: data (numpy array)
    sample_size: How many runs make up the data
    """
    data_padded=[]
    if (len(data)<sample_size*len_signal):
         data_padded=np.append(data,np.zeros(len_signal*sample_size-len(data)))
    else:
        data_padded=data
    return data_padded
    
def convert_to_3D_array(data):
    """
    Converts the list of all voltage-time traces to a 3-d numpy array with shape (# receivers, # data points (or runs), length of signal = window_size*sample_rate)
    
    Args: data (list)
    
    Returns: Data converted to numpy 3d array
    """
    data=np.asarray(data)
    return data.reshape(data.shape[0],np.int(data.shape[1]/len_signal),len_signal)
    
def read_time_vals(path,file):
    """
    Reads in the time values for the signals, assumes that they all have the same time vector
    
    Args: path (str)
          file (str)
          
    Returns: numpy array of time values
    """
    time_vals= pd.read_csv(path+file, skiprows=0,header=None)
    return np.asarray(time_vals).reshape(len_signal,)

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
            arrival_recs.append(time_vals[np.where(np.abs(hilbert(data[j][i]))>thres_arr_time*np.max(np.abs(hilbert(data[j][i]))))[0][0]])
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
            rise_time.append(np.round((time_vals[np.where(np.abs(hilbert(data[j][i]))==np.max(np.abs(hilbert(data[j][i]))))[0]]-time_vals[np.where(np.abs(hilbert(data[j][i]))>thres1*np.max(np.abs(hilbert(data[j][i]))))[0][0]])[0],3))
            fall_time.append(np.round((time_vals[np.where(np.abs(hilbert(data[j][i]))>thres2*np.max(np.abs(hilbert(data[j][i]))))[0][len(np.where(np.abs(hilbert(data[j][i]))>thres2*np.max(np.abs(hilbert(data[j][i]))))[0])-1]]-time_vals[np.where(np.abs(hilbert(data[j][i]))==np.max(np.abs(hilbert(data[j][i]))))[0]])[0],3))
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
           
def plot_properties(property,property_name,save_path,begin_rec,last_rec):
    """
    Plots a specific property eg intensity, peak frequency, for receivers as defined by begin_rec and last_rec. To plot receiver 1 only, put begin_rec=1 and last_rec=2.
    To plot all 4 receivers, put begin_rec=1 and last_rec=4
    
    Args: property (2-d numpy array of # receivers and # data points)
          begin_rec (int)
          last_rec (int)
    
    Returns: A plot of the summary property vs phi and theta
    
    """
    dir_df=pd.DataFrame()
    r=np.sqrt(xdir**2+ydir**2+zdir**2)
    theta_real = np.arccos(zdir)
    dir_df['theta']=theta_real
    phi1=phi1=np.arctan(ydir/xdir)
    dir_df['phi1']=phi1
    dir_df['xdir']=xdir
    dir_df['ydir']=ydir
    dir_df['zdir']=zdir
    dir_df['add_pi'] = np.where(dir_df['xdir'] > 0, 0, dir_df['xdir'])
    dir_df['add_pi'] = np.where(dir_df['xdir'] < 0, np.pi, dir_df['add_pi'])
    dir_df['phi_real']=dir_df['phi1']+dir_df['add_pi']
    dir_df['x_dir_check']=r*np.sin(dir_df['theta'])*np.cos(dir_df['phi_real'])
    dir_df['y_dir_check']=r*np.sin(dir_df['theta'])*np.sin(dir_df['phi_real'])
    dir_df['z_dir_check']=r*np.cos(dir_df['theta'])
    dir_df['phi_deg']=dir_df['phi_real']*180/np.pi
    dir_df['phi_deg']=np.where(dir_df['phi_deg'] < 0, dir_df['phi_deg']+360, dir_df['phi_deg'])
    dir_df['theta_deg']=dir_df['theta']*180/np.pi
    plt.subplot(121)
    for i in range(begin_rec-1,last_rec-1):
        plt.scatter(dir_df['phi_deg'],property[i],label='rec'+str(i+1))
    plt.yticks(fontsize=15)
    plt.xlabel(r'$\phi$ [deg]',fontsize=15)
    plt.ylabel(property_name, fontsize=25)
    plt.legend()
    plt.subplot(122)
    for i in range(begin_rec-1,last_rec-1):
        plt.scatter(dir_df['theta_deg'],property[i],label='rec'+str(i+1))
    plt.yticks(fontsize=15)
    plt.xlabel(r'$\theta$ [deg]',fontsize=15)
    plt.ylabel(property_name, fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + property_name + '_rec' + str(begin_rec)+'_'+str(last_rec)+'.png')
    plt.close()

def fit_predict_norm(mod, x, y, train_prop):
    """
    Fits a built-in model on the training and validation data and makes predictions on the validation data
    
    Args: mod (the model to use)
          x (Data frame containing predictor variables)
          y (Data frame containing outcome variable)
          train_prop (float, fraction of samples to use for training)
    
    Returns: mean squared error, validation labels, validation predictions
    
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = train_prop, random_state =  90)
    mod.fit(x_train,y_train)
    training_predictions = mod.predict(x_train)
    valid_predictions = mod.predict(x_valid)
    training_labels = y_train
    validation_labels = y_valid
    return sklearn.metrics.mean_squared_error(validation_labels,valid_predictions),np.asarray(validation_labels),valid_predictions
    
def fit_predict_norm_NN(mod, x, y, train_prop, epochs=100, batch_size=64):
    """
    Fits a manually constructed NN model
    
    Args: mod (the model to use)
          x (Data frame containing predictor variables)
          y (Data frame containing outcome variable)
          train_prop (fraction of samples to use for training)
          dir (string, the direction being trained on)
          epochs (int, The total number of epochs of training)
          batch_size (int, How many samples in a batch)
    
    Returns: mean squared error, validation labels, validation predictions, training loss, validation loss
    
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = train_prop, random_state =  90)
    es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
    history=mod.fit(x_train,y_train, validation_data=[x_valid,y_valid], batch_size=batch_size,epochs=epochs,verbose=1,callbacks=[es])
    training_predictions = mod.predict(x_train)
    valid_predictions = mod.predict(x_valid)
    training_labels = y_train
    validation_labels = y_valid
    return sklearn.metrics.mean_squared_error(validation_labels,valid_predictions),np.asarray(validation_labels),valid_predictions,history.history['loss'],history.history['val_loss']

def create_nn(a1):
    model = models.Sequential()
    model.add(Dense(a1.shape[1], input_dim=a1.shape[1], kernel_initializer=tensorflow.keras.initializers.lecun_uniform(), activation='relu'))
    model.add(Dense(5, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    model.add(Dense(50, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    model.add(Dense(100, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(300, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    #model.add(Dense(100, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    model.add(Dense(50, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    #model.add(Dense(300, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    model.add(Dense(10, kernel_initializer=tensorflow.keras.initializers.lecun_uniform(),activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse', optimizer='Rmsprop', metrics=['mse'])
    return model

def reconstruct(data,n_iter_all,n_epochs,batchsize,save_path,NN_train_val_loss,reconstruction_plot):
    """
    Performs training and validation, fits the trained model to the data to predict direction using linear regression and GBM (default settings), as well as a NN.
    Each ML model predicts x,y and z separately. These are then converted to theta and phi via the spherical coordinate transformation. The reconstruction accuracy
    is quantified by the opening angle alpha - the dot product of the real and reconstructed direction.
    
    Args: data (the normalised input data containing the predictor and outcomes variables.)
          n_iter_all (How many times to fit the models to the data)
          n_iter_NN (How many times to iterate through the neural network)
    
    Returns: lists of mean of means, means of std, etc
    
    """
    linreg_theta_mean_list=[]
    GBM_theta_mean_list=[]
    NN_theta_mean_list=[]
    linreg_theta_std_list=[]
    GBM_theta_std_list=[]
    NN_theta_std_list=[]
    linreg_phi_mean_list=[]
    GBM_phi_mean_list=[]
    NN_phi_mean_list=[]
    linreg_phi_std_list=[]
    GBM_phi_std_list=[]
    NN_phi_std_list=[]
    linreg_theta_diff_list=[]
    GBM_theta_diff_list=[]
    NN_theta_diff_list=[]
    linreg_phi_diff_list=[]
    GBM_phi_diff_list=[]
    NN_phi_diff_list=[]
    linreg_alpha_mean_list=[]
    GBM_alpha_mean_list=[]
    NN_alpha_mean_list=[]
    linreg_alpha_std_list=[]
    GBM_alpha_std_list=[]
    NN_alpha_std_list=[]
    invalid_linreg_theta=[]
    invalid_GBM_theta=[]
    invalid_NN_theta=[]
    invalid_linreg_phi=[]
    invalid_GBM_phi=[]
    invalid_NN_phi=[]
    invalid_alpha_linreg=[]
    invalid_alpha_GBM=[]
    invalid_alpha_NN=[]
    alpha_GBM_list=[]
    alpha_NN_list=[]
    alpha_linreg_list=[]
    for j in range(0,n_iter_all):
        print(j)
        data_trunc=data
        data_trunc=data_trunc.sample(frac=1)
        GBM_reg0 = sklearn.ensemble.GradientBoostingRegressor()
        lin_reg0 = LinearRegression()
        GBM_reg1 = sklearn.ensemble.GradientBoostingRegressor()
        lin_reg1 = LinearRegression()
        GBM_reg2 = sklearn.ensemble.GradientBoostingRegressor()
        lin_reg2 = LinearRegression()
        np.sum(np.isinf(data_trunc))
        np.sum(np.isnan(data_trunc))
        sc = sklearn.preprocessing.StandardScaler()
        data_norm = sc.fit_transform(data_trunc)
        data_norm = pd.DataFrame(data_norm, index=data_trunc.index, columns=data_trunc.columns)
        data_trunc1 = data_trunc
        trainingx_norm_col = data_norm
        n_samples_view=20
        train_prop=0.8
        #es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
        #print(es)
        n=np.int(train_prop*len(data_norm))
        #x1 = data_norm.iloc[:,:(n_rec*df_list[0].shape[0]-1)][0:n]
        x1 = data_norm.iloc[:,:(data_norm.shape[1]-3)][0:n]
        #x1 = data_norm.iloc[:,:(data_norm.shape[1])][0:n]
        y1 = data_norm.loc[:,'xdir'][0:n]
        y2 = data_norm.loc[:,'ydir'][0:n]
        y3 = data_norm.loc[:,'zdir'][0:n]
        #x2 = data_norm.iloc[:,:(n_rec*df_list[0].shape[0]-1)][n:data_norm.shape[0]]
        x2 = data_norm.iloc[:,:(data_norm.shape[1]-3)][n:data_norm.shape[0]]
        #x2 = data_norm.iloc[:,:(data_norm.shape[1])][n:data_norm.shape[0]]
        y4 = data_norm.loc[:,'xdir'][n:int(data_norm.shape[0])] ### test solutions
        y5 = data_norm.loc[:,'ydir'][n:int(data_norm.shape[0])] ### test solutions
        y6 = data_norm.loc[:,'zdir'][n:int(data_norm.shape[0])] ### test solutions
        np.sum(np.isinf(data_norm))
        np.sum(np.isnan(data_norm))
        mse_NN_list_norm1=list()
        valid_predictions_NN_list_norm1=list()
        modelnn1=create_nn(x1)
        mse_NN_norm1,validation_labels_norm1,valid_predictions_NN_norm1,train_loss_x,valid_loss_x=fit_predict_norm_NN(modelnn1,x1,y1,train_prop,n_epochs,batchsize)
        mse_NN_list_norm1.append(mse_NN_norm1)
        valid_predictions_NN_list_norm1.append(valid_predictions_NN_norm1)
        mse_NN_list_norm2=list()
        valid_predictions_NN_list_norm2=list()
        #for i in range(0,n_iter_NN):
        modelnn2=create_nn(x1)
        mse_NN_norm2,validation_labels_norm2,valid_predictions_NN_norm2,train_loss_y,valid_loss_y=fit_predict_norm_NN(modelnn2,x1,y2,train_prop,n_epochs,batchsize)
        mse_NN_list_norm2.append(mse_NN_norm2)
        valid_predictions_NN_list_norm2.append(valid_predictions_NN_norm2)
        mse_NN_list_norm3=list()
        valid_predictions_NN_list_norm3=list()
        modelnn3=create_nn(x1)
        mse_NN_norm3,validation_labels_norm3,valid_predictions_NN_norm3,train_loss_z,valid_loss_z=fit_predict_norm_NN(modelnn3,x1,y3,train_prop,n_epochs,batchsize)
        mse_NN_list_norm3.append(mse_NN_norm3)
        valid_predictions_NN_list_norm3.append(valid_predictions_NN_norm3)
        train_prop=0.8
        mse_GBM_norm1,validation_labels_norm1,valid_predictions_GBM_norm1=fit_predict_norm(GBM_reg0,x1,y1,train_prop)
        mse_linreg_norm1,validation_labels_norm1,valid_predictions_linreg_norm1=fit_predict_norm(lin_reg0,x1,y1,train_prop)
        mse_GBM_norm2,validation_labels_norm2,valid_predictions_GBM_norm2=fit_predict_norm(GBM_reg1,x1,y2,train_prop)
        mse_linreg_norm2,validation_labels_norm2,valid_predictions_linreg_norm2=fit_predict_norm(lin_reg1,x1,y2,train_prop)
        mse_GBM_norm3,validation_labels_norm3,valid_predictions_GBM_norm3=fit_predict_norm(GBM_reg2,x1,y3,train_prop)
        mse_linreg_norm3,validation_labels_norm3,valid_predictions_linreg_norm3=fit_predict_norm(lin_reg2,x1,y3,train_prop)
        GBM_test_predictions_x=GBM_reg0.predict(x2)
        linreg_test_predictions_x=lin_reg0.predict(x2)
        neuralnet_predictions_x=modelnn1.predict(x2)
        GBM_test_predictions_y=GBM_reg1.predict(x2)
        linreg_test_predictions_y=lin_reg1.predict(x2)
        neuralnet_predictions_y=modelnn2.predict(x2)
        GBM_test_predictions_z=GBM_reg2.predict(x2)
        linreg_test_predictions_z=lin_reg2.predict(x2)
        neuralnet_predictions_z=modelnn3.predict(x2)
        ##### Convert normalised values back to usual values for test set
        n=data_trunc1.shape[0]
        #from sklearn.linear_model import LinearRegression
        modelx = LinearRegression()
        modelx = LinearRegression().fit(np.asarray(trainingx_norm_col['xdir'][0:n]).reshape((-1, 1)),np.asarray(data_trunc1['xdir'][0:n]).reshape((-1, 1)))
        print('intercept:', modelx.intercept_)
        print('slope:', modelx.coef_)
        data_trunc1['x_dir_check'] = modelx.coef_[0][0]*trainingx_norm_col['xdir'] + modelx.intercept_[0]
        modely = LinearRegression()
        modely = LinearRegression().fit(np.asarray(trainingx_norm_col['ydir'][0:n]).reshape((-1, 1)),np.asarray(data_trunc1['ydir'][0:n]).reshape((-1, 1)))
        print('intercept:', modely.intercept_)
        print('slope:', modely.coef_)
        data_trunc1['y_dir_check'] = modely.coef_[0][0]*trainingx_norm_col['ydir'] + modely.intercept_[0]
        modelz = LinearRegression()
        modelz = LinearRegression().fit(np.asarray(trainingx_norm_col['zdir'][0:n]).reshape((-1, 1)),np.asarray(data_trunc1['zdir'][0:n]).reshape((-1, 1)))
        print('intercept:', modelz.intercept_)
        print('slope:', modelz.coef_)
        data_trunc1['z_dir_check'] = modelz.coef_[0][0]*trainingx_norm_col['zdir'] + modelz.intercept_[0]
    ##### return x,y,z back to original scale
        orig_x_real = modelx.coef_[0][0]*np.asarray(y4) + modelx.intercept_[0]
        orig_y_real = modely.coef_[0][0]*np.asarray(y5) + modely.intercept_[0]
        orig_z_real = modelz.coef_[0][0]*np.asarray(y6) + modelz.intercept_[0]
    # check if normalised
        orig_x_GBM = modelx.coef_[0][0]*GBM_test_predictions_x + modelx.intercept_[0]
        orig_y_GBM = modely.coef_[0][0]*GBM_test_predictions_y + modely.intercept_[0]
        orig_z_GBM = modelz.coef_[0][0]*GBM_test_predictions_z + modelz.intercept_[0]
        norm_x_GBM=[]
        norm_y_GBM=[]
        norm_z_GBM=[]
        ### Normalise the reconstructed directions
        for i in range(0,len(orig_x_GBM)):
            norm_x_GBM.append(orig_x_GBM[i]/np.sqrt(orig_x_GBM[i]**2+orig_y_GBM[i]**2+orig_z_GBM[i]**2))
            norm_y_GBM.append(orig_y_GBM[i]/np.sqrt(orig_x_GBM[i]**2+orig_y_GBM[i]**2+orig_z_GBM[i]**2))
            norm_z_GBM.append(orig_z_GBM[i]/np.sqrt(orig_x_GBM[i]**2+orig_y_GBM[i]**2+orig_z_GBM[i]**2))
        orig_x_GBM = np.asarray(norm_x_GBM)
        orig_y_GBM = np.asarray(norm_y_GBM)
        orig_z_GBM = np.asarray(norm_z_GBM)
    # check if normalised
        orig_x_NN = modelx.coef_[0][0]*neuralnet_predictions_x.reshape(len(neuralnet_predictions_x),) + modelx.intercept_[0]
        orig_y_NN = modely.coef_[0][0]*neuralnet_predictions_y.reshape(len(neuralnet_predictions_y),) + modely.intercept_[0]
        orig_z_NN = modelz.coef_[0][0]*neuralnet_predictions_z.reshape(len(neuralnet_predictions_z),) + modelz.intercept_[0]
        norm_x_NN=[]
        norm_y_NN=[]
        norm_z_NN=[]
        ### Normalise the reconstructed directions
        for i in range(0,len(orig_x_NN)):
            norm_x_NN.append(orig_x_NN[i]/np.sqrt(orig_x_NN[i]**2+orig_y_NN[i]**2+orig_z_NN[i]**2))
            norm_y_NN.append(orig_y_NN[i]/np.sqrt(orig_x_NN[i]**2+orig_y_NN[i]**2+orig_z_NN[i]**2))
            norm_z_NN.append(orig_z_NN[i]/np.sqrt(orig_x_NN[i]**2+orig_y_NN[i]**2+orig_z_NN[i]**2))
        orig_x_NN = np.asarray(norm_x_NN)
        orig_y_NN = np.asarray(norm_y_NN)
        orig_z_NN = np.asarray(norm_z_NN)
    # chek if normalised
        orig_x_linreg = modelx.coef_[0][0]*linreg_test_predictions_x + modelx.intercept_[0]
        orig_y_linreg = modely.coef_[0][0]*linreg_test_predictions_y + modely.intercept_[0]
        orig_z_linreg = modelz.coef_[0][0]*linreg_test_predictions_z + modelz.intercept_[0]
        norm_x_linreg=[]
        norm_y_linreg=[]
        norm_z_linreg=[]
        ### Normalise the reconstructed directions
        for i in range(0,len(orig_x_linreg)):
            norm_x_linreg.append(orig_x_linreg[i]/np.sqrt(orig_x_linreg[i]**2+orig_y_linreg[i]**2+orig_z_linreg[i]**2))
            norm_y_linreg.append(orig_y_linreg[i]/np.sqrt(orig_x_linreg[i]**2+orig_y_linreg[i]**2+orig_z_linreg[i]**2))
            norm_z_linreg.append(orig_z_linreg[i]/np.sqrt(orig_x_linreg[i]**2+orig_y_linreg[i]**2+orig_z_linreg[i]**2))
        orig_x_linreg = np.asarray(norm_x_linreg)
        orig_y_linreg = np.asarray(norm_y_linreg)
        orig_z_linreg = np.asarray(norm_z_linreg)
        v1dotv2_GBM=orig_x_GBM*orig_x_real+orig_y_GBM*orig_y_real+orig_z_GBM*orig_z_real
        v1dotv2_NN=orig_x_NN*orig_x_real+orig_y_NN*orig_y_real+orig_z_NN*orig_z_real
        v1dotv2_linreg=orig_x_linreg*orig_x_real+orig_y_linreg*orig_y_real+orig_z_linreg*orig_z_real
        real_vec_magnitude=np.sqrt(orig_x_real**2+orig_y_real**2+orig_z_real**2)
        alpha_GBM=np.arccos(v1dotv2_GBM/real_vec_magnitude**2)
        invalid_alpha_GBM.append(np.sum(np.isnan(alpha_GBM)))
        print(invalid_alpha_GBM)
        alpha_GBM=pd.Series(alpha_GBM)
        alpha_GBM.replace(np.nan, 100, inplace=True)
        alpha_GBM=alpha_GBM[alpha_GBM!=100]
        alpha_NN=np.arccos(v1dotv2_NN/real_vec_magnitude**2)
        invalid_alpha_NN.append(np.sum(np.isnan(alpha_NN)))
        print(invalid_alpha_NN)
        alpha_NN=pd.Series(alpha_NN)
        alpha_NN.replace(np.nan, 100, inplace=True)
        alpha_NN=alpha_NN[alpha_NN!=100]
        alpha_linreg=np.arccos(v1dotv2_linreg/real_vec_magnitude**2)
        invalid_alpha_linreg.append(np.sum(np.isnan(alpha_linreg)))
        print(invalid_alpha_linreg)
        alpha_linreg=pd.Series(alpha_linreg)
        alpha_linreg.replace(np.nan, 100, inplace=True)
        alpha_linreg=alpha_linreg[alpha_linreg!=100]
        alpha_GBM_list.append(alpha_GBM)
        alpha_NN_list.append(alpha_NN)
        alpha_linreg_list.append(alpha_linreg)
#       alpha_ML_list.append(alpha_ML)
#       ML_alpha_mean_list.append(np.round(np.mean(alpha_ML),3))
#       ML_alpha_std_list.append(np.round(np.std(alpha_ML),3))
        linreg_alpha_mean_list.append(np.round(np.mean(alpha_linreg),3))
        GBM_alpha_mean_list.append(np.round(np.mean(alpha_GBM),3))
        NN_alpha_mean_list.append(np.round(np.mean(alpha_NN),3))
        linreg_alpha_std_list.append(np.round(np.std(alpha_linreg),3))
        GBM_alpha_std_list.append(np.round(np.std(alpha_GBM),3))
        NN_alpha_std_list.append(np.round(np.std(alpha_NN),3))
        #print(NN_alpha_std_list)
    ##### Convert the real and reconstructed x,y,z values to theta and phi
        real_reco_df=pd.DataFrame()
        r=np.sqrt(orig_x_real**2+orig_y_real**2+orig_z_real**2)
        theta_real = np.arccos(orig_z_real)
        real_reco_df['r']=r
        real_reco_df['theta']=theta_real
        phi1=np.arctan(orig_y_real/orig_x_real)
        real_reco_df['phi1']=phi1
        real_reco_df['orig_x_real']=orig_x_real
        real_reco_df['add_pi'] = np.where(real_reco_df['orig_x_real'] > 0, 0, real_reco_df['orig_x_real'])
        real_reco_df['add_pi'] = np.where(real_reco_df['orig_x_real'] < 0, np.pi, real_reco_df['add_pi'])
        real_reco_df['phi_real']=real_reco_df['phi1']+real_reco_df['add_pi']
        real_reco_df['x_dir_check']=r*np.sin(real_reco_df['theta'])*np.cos(real_reco_df['phi_real'])
        real_reco_df['y_dir_check']=r*np.sin(real_reco_df['theta'])*np.sin(real_reco_df['phi_real'])
        real_reco_df['z_dir_check']=r*np.cos(real_reco_df['theta'])
        #print(np.min(real_reco_df['phi_real'])*(180/np.pi))
        #print(np.max(real_reco_df['phi_real'])*(180/np.pi))
        #print(np.min(real_reco_df['theta'])*(180/np.pi))
        #print(np.max(real_reco_df['theta'])*(180/np.pi))
        real_reco_df['theta_deg']=real_reco_df['theta']*(180/np.pi)
        real_reco_df['phi_deg']=real_reco_df['phi_real']*(180/np.pi)
        #print(np.sum(np.round(real_reco_df['x_dir_check'],2)==np.round(orig_x_real,2)))
        #print(np.sum(np.round(real_reco_df['y_dir_check'],2)==np.round(orig_y_real,2)))
        #print(np.sum(np.round(real_reco_df['z_dir_check'],2)==np.round(orig_z_real,2)))
        real_reco_df['phi_deg']=np.where(real_reco_df['phi_deg'] < 0,real_reco_df['phi_deg']+360, real_reco_df['phi_deg'])
        GBM_reco_df=pd.DataFrame()
        r=np.sqrt(orig_x_GBM**2+orig_y_GBM**2+orig_z_GBM**2)
        theta_real = np.arccos(orig_z_GBM)
        GBM_reco_df['theta']=theta_real
        phi1=np.arctan(orig_y_GBM/orig_x_GBM)
        GBM_reco_df['phi1']=phi1
        GBM_reco_df['orig_x_GBM']=orig_x_GBM
        GBM_reco_df['add_pi'] = np.where(GBM_reco_df['orig_x_GBM'] > 0, 0, GBM_reco_df['orig_x_GBM'])
        GBM_reco_df['add_pi'] = np.where(GBM_reco_df['orig_x_GBM'] < 0, np.pi, GBM_reco_df['add_pi'])
        GBM_reco_df['phi_real']=GBM_reco_df['phi1']+GBM_reco_df['add_pi']
        GBM_reco_df['x_dir_check']=r*np.sin(GBM_reco_df['theta'])*np.cos(GBM_reco_df['phi_real'])
        GBM_reco_df['y_dir_check']=r*np.sin(GBM_reco_df['theta'])*np.sin(GBM_reco_df['phi_real'])
        GBM_reco_df['z_dir_check']=r*np.cos(GBM_reco_df['theta'])
        np.min(GBM_reco_df['phi_real'])*(180/np.pi)
        np.max(GBM_reco_df['phi_real'])*(180/np.pi)
        np.min(GBM_reco_df['theta'])*(180/np.pi)
        np.max(GBM_reco_df['theta'])*(180/np.pi)
        GBM_reco_df['theta_deg']=GBM_reco_df['theta']*(180/np.pi)
        GBM_reco_df['phi_deg']=GBM_reco_df['phi_real']*(180/np.pi)
        #print(np.sum(np.round(GBM_reco_df['x_dir_check'],2)==np.round(orig_x_GBM,2)))
        #print(np.sum(np.round(GBM_reco_df['y_dir_check'],2)==np.round(orig_y_GBM,2)))
        #print(np.sum(np.round(GBM_reco_df['z_dir_check'],2)==np.round(orig_z_GBM,2)))
        GBM_reco_df['phi_deg']=np.where(GBM_reco_df['phi_deg'] < 0, GBM_reco_df['phi_deg']+360, GBM_reco_df['phi_deg'])
        NN_reco_df=pd.DataFrame()
        r=np.sqrt(orig_x_NN**2+orig_y_NN**2+orig_z_NN**2)
        theta_real = np.arccos(orig_z_NN)
        NN_reco_df['theta']=theta_real
        phi1=np.arctan(orig_y_NN/orig_x_NN)
        NN_reco_df['phi1']=phi1
        NN_reco_df['orig_x_NN']=orig_x_NN
        NN_reco_df['add_pi'] = np.where(NN_reco_df['orig_x_NN'] > 0, 0, NN_reco_df['orig_x_NN'])
        NN_reco_df['add_pi'] = np.where(NN_reco_df['orig_x_NN'] < 0, np.pi, NN_reco_df['add_pi'])
        NN_reco_df['phi_real']=NN_reco_df['phi1']+NN_reco_df['add_pi']
        NN_reco_df['x_dir_check']=r*np.sin(NN_reco_df['theta'])*np.cos(NN_reco_df['phi_real'])
        NN_reco_df['y_dir_check']=r*np.sin(NN_reco_df['theta'])*np.sin(NN_reco_df['phi_real'])
        NN_reco_df['z_dir_check']=r*np.cos(NN_reco_df['theta'])
        np.min(NN_reco_df['phi_real'])*(180/np.pi)
        np.max(NN_reco_df['phi_real'])*(180/np.pi)
        np.min(NN_reco_df['theta'])*(180/np.pi)
        np.max(NN_reco_df['theta'])*(180/np.pi)
        NN_reco_df['theta_deg']=NN_reco_df['theta']*(180/np.pi)
        NN_reco_df['phi_deg']=NN_reco_df['phi_real']*(180/np.pi)
        print(np.sum(np.round(NN_reco_df['x_dir_check'],2)==np.round(orig_x_NN,2)))
        print(np.sum(np.round(NN_reco_df['y_dir_check'],2)==np.round(orig_y_NN,2)))
        print(np.sum(np.round(NN_reco_df['z_dir_check'],2)==np.round(orig_z_NN,2)))
        NN_reco_df['phi_deg']=np.where(NN_reco_df['phi_deg'] < 0, NN_reco_df['phi_deg']+360, NN_reco_df['phi_deg'])
        linreg_reco_df=pd.DataFrame()
        r=np.sqrt(orig_x_linreg**2+orig_y_linreg**2+orig_z_linreg**2)
        theta_real = np.arccos(orig_z_linreg)
        linreg_reco_df['theta']=theta_real
        phi1=np.arctan(orig_y_linreg/orig_x_linreg)
        linreg_reco_df['phi1']=phi1
        linreg_reco_df['orig_x_linreg']=orig_x_linreg
        linreg_reco_df['add_pi'] = np.where(linreg_reco_df['orig_x_linreg'] > 0, 0, linreg_reco_df['orig_x_linreg'])
        linreg_reco_df['add_pi'] = np.where(linreg_reco_df['orig_x_linreg'] < 0, np.pi, linreg_reco_df['add_pi'])
        linreg_reco_df['phi_real']=linreg_reco_df['phi1']+linreg_reco_df['add_pi']
        linreg_reco_df['x_dir_check']=r*np.sin(linreg_reco_df['theta'])*np.cos(linreg_reco_df['phi_real'])
        linreg_reco_df['y_dir_check']=r*np.sin(linreg_reco_df['theta'])*np.sin(linreg_reco_df['phi_real'])
        linreg_reco_df['z_dir_check']=r*np.cos(linreg_reco_df['theta'])
        np.min(linreg_reco_df['phi_real'])*(180/np.pi)
        np.max(linreg_reco_df['phi_real'])*(180/np.pi)
        np.min(linreg_reco_df['theta'])*(180/np.pi)
        np.max(linreg_reco_df['theta'])*(180/np.pi)
        linreg_reco_df['theta_deg']=linreg_reco_df['theta']*(180/np.pi)
        linreg_reco_df['phi_deg']=linreg_reco_df['phi_real']*(180/np.pi)
        #print(np.sum(np.round(linreg_reco_df['x_dir_check'],2)==np.round(orig_x_linreg,2)))
        #print(np.sum(np.round(linreg_reco_df['y_dir_check'],2)==np.round(orig_y_linreg,2)))
        #print(np.sum(np.round(linreg_reco_df['z_dir_check'],2)==np.round(orig_z_linreg,2)))
        linreg_reco_df['phi_deg']=np.where(linreg_reco_df['phi_deg'] < 0, linreg_reco_df['phi_deg']+360, linreg_reco_df['phi_deg'])
        linreg_theta_diff=real_reco_df['theta_deg']-linreg_reco_df['theta_deg']
        invalid_linreg_theta.append(np.sum(np.isnan(linreg_theta_diff)))
        linreg_theta_diff.replace(np.nan, 0, inplace=True)
        linreg_theta_diff=linreg_theta_diff[linreg_theta_diff!=0]
        linreg_phi_diff=real_reco_df['phi_deg']-linreg_reco_df['phi_deg']
        linreg_phi_diff=np.where(linreg_phi_diff > 300, 360-linreg_phi_diff, linreg_phi_diff)
        linreg_phi_diff=np.where(linreg_phi_diff < -300, 360+linreg_phi_diff, linreg_phi_diff)
        linreg_phi_diff=pd.Series(linreg_phi_diff)
        invalid_linreg_phi.append(np.sum(np.isnan(linreg_phi_diff)))
        linreg_phi_diff.replace(np.nan, 0, inplace=True)
        linreg_phi_diff=linreg_phi_diff[linreg_phi_diff!=0]
        GBM_theta_diff=real_reco_df['theta_deg']-GBM_reco_df['theta_deg']
        invalid_GBM_theta.append(np.sum(np.isnan(GBM_theta_diff)))
        GBM_theta_diff.replace(np.nan, 0, inplace=True)
        GBM_theta_diff=GBM_theta_diff[GBM_theta_diff!=0]
        GBM_phi_diff=real_reco_df['phi_deg']-GBM_reco_df['phi_deg']
        GBM_phi_diff=np.where(GBM_phi_diff > 300, 360-GBM_phi_diff, GBM_phi_diff)
        GBM_phi_diff=np.where(GBM_phi_diff < -300, 360+GBM_phi_diff, GBM_phi_diff)
        GBM_phi_diff=pd.Series(GBM_phi_diff)
        invalid_GBM_phi.append(np.sum(np.isnan(GBM_phi_diff)))
        GBM_phi_diff.replace(np.nan, 0, inplace=True)
        GBM_phi_diff=GBM_phi_diff[GBM_phi_diff!=0]
        NN_theta_diff=real_reco_df['theta_deg']-NN_reco_df['theta_deg']
        invalid_NN_theta.append(np.sum(np.isnan(NN_theta_diff)))
        NN_theta_diff.replace(np.nan, 0, inplace=True)
        NN_theta_diff=NN_theta_diff[NN_theta_diff!=0]
        NN_phi_diff=real_reco_df['phi_deg']-NN_reco_df['phi_deg']
        NN_phi_diff=np.where(NN_phi_diff > 300, 360-NN_phi_diff, NN_phi_diff)
        NN_phi_diff=np.where(NN_phi_diff < -300, 360+NN_phi_diff, NN_phi_diff)
        NN_phi_diff=pd.Series(NN_phi_diff)
        invalid_NN_phi.append(np.sum(np.isnan(NN_phi_diff)))
        NN_phi_diff.replace(np.nan, 0, inplace=True)
        NN_phi_diff=NN_phi_diff[NN_phi_diff!=0]
    flat_linreg_alpha_list = [item for sublist in alpha_linreg_list for item in sublist]
    flat_GBM_alpha_list = [item for sublist in alpha_GBM_list for item in sublist]
    flat_NN_alpha_list = [item for sublist in alpha_NN_list for item in sublist]
    nbins=50
    plt.figure(figsize=(10, 7))
    #plt.title('Opening angle alpha')
    plt.hist(np.asarray(flat_linreg_alpha_list)*(180/np.pi),alpha=0.5,bins=nbins,color="green",label='linreg alpha, mean= '+str(np.round(np.mean(np.asarray(linreg_alpha_mean_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(linreg_alpha_mean_list)*180/np.pi),3))+', std= ' +str(np.round(np.mean(np.asarray(linreg_alpha_std_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(linreg_alpha_std_list)*180/np.pi),3)))
    plt.hist(np.asarray(flat_GBM_alpha_list)*(180/np.pi),alpha=0.5,bins=nbins,color="blue",label='GBM alpha, mean= '+str(np.round(np.mean(np.asarray(GBM_alpha_mean_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(GBM_alpha_mean_list)*180/np.pi),3))+', std= ' +str(np.round(np.mean(np.asarray(GBM_alpha_std_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(GBM_alpha_std_list)*180/np.pi),3)))
    plt.hist(np.asarray(flat_NN_alpha_list)*(180/np.pi),alpha=0.5,bins=nbins,color="orange",label='NN alpha, mean= '+str(np.round(np.mean(np.asarray(NN_alpha_mean_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(NN_alpha_mean_list)*180/np.pi),3))+', std= ' +str(np.round(np.mean(np.asarray(NN_alpha_std_list)*180/np.pi),3))+'+/-'+str(np.round(np.std(np.asarray(NN_alpha_std_list)*180/np.pi),3)))
    plt.xlabel('Alpha [deg]',fontsize=20)
    plt.ylabel('Number of runs',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path+reconstruction_plot+'.png')
    #plt.savefig(path+'Histogram_alpha_all_own_signal_properties_efficient_2000_4_recs_vf'+str(n_iter_all)+'_'+str(n_epochs)+'_'+str(batchsize)+'_dummy_check_23_6_22.png')
    plt.close()
    plt.subplot(311)
    plt.plot(train_loss_x,label='Training loss xdir')
    plt.plot(valid_loss_x,label='Validation loss xdir')
    plt.subplot(312)
    plt.plot(train_loss_y,label='Training loss ydir')
    plt.plot(valid_loss_y,label='Validation loss ydir')
    plt.subplot(313)
    plt.plot(train_loss_z,label='Training loss zdir')
    plt.plot(valid_loss_z,label='Validation loss zdir')
    plt.title('Training and validation loss vs epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(save_path+NN_train_val_loss+'.png')
    plt.close()
    return flat_linreg_alpha_list,flat_GBM_alpha_list,flat_NN_alpha_list

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Plot signal properties vs direction and/or perform reconstruction")
    parser.add_argument('--save_path', default='/Users/vesnalukic/Desktop/Reconstruction/',type=str)
    parser.add_argument('--plot_properties', default='True',type=str)
    parser.add_argument('--reconstruct', default='True',type=str)
    parser.add_argument('--NN_train_val_loss', default='Train_validation_loss',type=str)
    parser.add_argument('--reconstruction_plot', default='Reconstruction_histogram',type=str)
    parser.add_argument('--signal_files', default='*_numbers_only2.txt')
    parser.add_argument('--time_file', default='four_receivers_change_dir_fillbyEvent0_manual_1000_50MHz_10_events_to_run_100_GeV_primary_best_dir_sim_10_PeV_400ns_window_vpol_time.txt')
    parser.add_argument('--directions', default='four_recs_rand_dir.csv')
    parser.add_argument('--thres_arr_time', default=0.3,type=float)
    parser.add_argument('--thres_bandwidth', default=0.05, type=float)
    parser.add_argument('--window_size', default=400,type=int)
    parser.add_argument('--sample_rate', default=5,type=int)
    parser.add_argument('--thres1', default=0.3,type=float)
    parser.add_argument('--thres2', default=0.3,type=float)
    parser.add_argument('--tf', default='0.05',type=float)
    parser.add_argument('--n_samples', default='990',type=int)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

len_signal=args.window_size*args.sample_rate
          
print('Reading in signal files:')
sorted_file_list,data_list=read_signal_files(args.save_path,args.signal_files)
    
n_rec=len(sorted_file_list)
data_list_all=[]

for i in range(0,n_rec):
    data_list_all.append(pad_data(data_list[i],args.n_samples))
 
data=convert_to_3D_array(data_list_all)

print('Reading in directions')
dirs_fillbyEvent0=pd.read_csv(args.save_path+args.directions, skiprows=1,header=None)

xdir=dirs_fillbyEvent0[1][0:args.n_samples]
ydir=dirs_fillbyEvent0[2][0:args.n_samples]
zdir=dirs_fillbyEvent0[3][0:args.n_samples]
 
# Read in time file

#time_file='four_receivers_change_dir_fillbyEvent0_manual_1000_50MHz_10_events_to_run_100_GeV_primary_best_dir_sim_10_PeV_400ns_window_vpol_time.txt'

print('Reading in time file')
time_vals=read_time_vals(args.save_path,args.time_file)

n=data.shape[2]
T=1/args.sample_rate

int_array=intensity(data)
arrival_at_recs=arrival_rec(data,args.thres_arr_time,time_vals)
peakfreqs=peakFreq(data,args.sample_rate)
rise_time,fall_time=rise_fall_time(data,time_vals,args.thres1,args.thres2)
bandwidths=bandwidth_recs(data,args.sample_rate,args.thres_bandwidth)
max_amp_freqs,half_amp_freqs,at_amp_freqs,twice_amp_freqs=fft_amps(data,args.sample_rate,args.tf)
           
df_list=[]

for i in range(0,data.shape[0]):
    print(i)
    df_list.append(np.asarray([int_array[i],arrival_at_recs[i],peakfreqs[i],rise_time[i],fall_time[i],bandwidths[i],max_amp_freqs[i],max_amp_freqs[i],half_amp_freqs[i],at_amp_freqs[i],twice_amp_freqs[i]]))

l = np.vstack(df_list)

all_df=pd.DataFrame(l.T)
           
all_df['xdir']=xdir
all_df['ydir']=ydir
all_df['zdir']=zdir
           
if (args.plot_properties=='True'):
           
    print('Plotting properties')
    plot_properties(int_array,'Intensity_n_1p78_zpol',args.save_path,1,5)
    plot_properties(arrival_at_recs,'Arrival_time_n_1p78_zpol',args.save_path,1,5)
    plot_properties(peakfreqs,'peakFreq_n_1p78_zpol',args.save_path,1,5)
    plot_properties(rise_time,'Rise_time_n_1p78_zpol',args.save_path,1,5)
    plot_properties(fall_time,'Fall_time_n_1p78_zpol',args.save_path,1,5)
    plot_properties(bandwidths,'Bandwidth_n_1p78_zpol',args.save_path,1,5)
    plot_properties(max_amp_freqs,'Max_fft_amplitude_n_1p78_zpol',args.save_path,1,5)
    plot_properties(half_amp_freqs,'Amplitude_at_0.5tf_n_1p78_zpol',args.save_path,1,5)
    plot_properties(at_amp_freqs,'Amplitude_at_tf_n_1p78_zpol',args.save_path,1,5)
    plot_properties(twice_amp_freqs,'Amplitude_at_2tf_n_1p78_zpol',args.save_path,1,5)

    print('Signal properties plotted vs phi and theta')
    
if (args.reconstruct=='True'):

    print('Performing reconstruction')
    
    linreg_alpha_list,GBM_alpha_list,NN_alpha_list=reconstruct(all_df,5,200,256,args.save_path,args.NN_train_val_loss,args.reconstruction_plot)
    
else:
    print('Please specify plotting properties and/or reconstruction')
