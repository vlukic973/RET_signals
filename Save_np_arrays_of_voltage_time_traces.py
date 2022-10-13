print('Importing modules')

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, chirp
import pickle
import glob
import os
import re
#import argparse

#print(args.file_names[0:10])

print('all imported')

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
    #
    Args: path (str)
          files (str)
          
    Returns: List of files sorted in natural order and list of data with all voltage-time traces from each receiver from receiver 1 to receiver n
    #
    """
    file_list = glob.glob(path + files)
    file_list.sort(key=natural_keys)
    data = []
    for file_path in file_list:
        print('Reading in '+file_path)
        try:
            data.append(np.genfromtxt(file_path, delimiter='\n', skip_header=10))
        except Exception:
            pass
    return file_list, data
#
#
def pad_data(data,sample_size):
    """
    Pads input data in case the signals are slightly truncated
    #
    Args: data (numpy array)
    sample_size: How many runs make up the data
    """
    data_padded=[]
    if (len(data)<sample_size*len_signal):
         data_padded=np.append(data,np.zeros(len_signal-len(data)))
    else:
        data_padded=data
    return data_padded
#
#
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

def main():
    
    import os
    import argparse

    print('Now in main')

    parser = argparse.ArgumentParser(description="Save voltage-time traces as numpy arrays")
    parser.add_argument('--file_names', default='receiver_1_27_recs_vary_pos_dir_energy_removed_10_*')
    parser.add_argument('--path_to_files', default='/pnfs/iihe/radar/store/user/vlukic/twenty_seven_rec_root_files_n_1p78/voltage_time_traces/removed_10/')
    parser.add_argument('--save_dir', default='/pnfs/iihe/radar/store/user/vlukic/twenty_seven_rec_root_files_n_1p78/voltage_time_traces/removed_10/')
    parser.add_argument('--window_size', default=400,type=int)
    parser.add_argument('--sample_rate', default=5, type=int)
    parser.add_argument('--n_samples', default=2000,type=int)
    parser.add_argument('--n_files', default=10000,type=int)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    process_data(args)

def process_data(args):

    print(args.file_names[0:10])
    window_size=args.window_size
    sample_rate=args.sample_rate
    n_samples=args.n_samples
    n_files=args.n_files
    len_signal=window_size*sample_rate
    
    path=args.path_to_files

    file=args.file_names

    sorted_file_list,data_list=read_signal_files(path,file)

    data_list_all=[]

    for i in range(0,args.n_files):
        print(i)
        data_list_all1.append(pad_data(data_list[i],args.n_samples))

    data_rec=np.asarray(data_list_all)

    np.save(args.save_dir+'data_'+args.file_names[0:10]+'_10000.npy',data_rec)

if __name__ == "__main__":
    main()

