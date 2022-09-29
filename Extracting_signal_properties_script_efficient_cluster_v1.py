import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, chirp
import pickle
import glob
import os
import re
        
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

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Plot signal properties vs direction")
    parser.add_argument('--save_path', default='/Users/vesnalukic/Desktop/Reconstruction/')
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

print('Done!')
