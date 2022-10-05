# RET_signals
This repository contains code for plotting signal properties, and reconstruction

The code (Plot_signal_properties_and_reconstruct.py) is able to plot signal properties and/or perform reconstruction on data generated using RadioScatter. The signal properties derived are the intensity, arrival time at receivers, peak frequency, bandwidth, rise and fall time maximum amplitude at transmit frequency, amplitude at half, at and twice the transmit frequency.

Prior to running the python code, please use the RET_signals_env.yaml file provided by supplying the following commands on the command-line:

conda env create -f RET_signals_env.yaml
conda activate RET_signals

Also download the other data provided, given in the text and csv files.

4-receiver data

The data provided includes the signals generated across 4 receivers in the 'cross' setup, beginning with '*_numbers_only2.txt'. There are 990 runs in total, each with an associated direction (given in four_recs_rand_dir.csv) and signal. The signals in the file '*_numbers_only2.txt' are appended one after the other.

The corresponding values on the time axis are given by '*_time.txt'

With the data provided it is possible to plot the signal properties and/or perform reconstruction.

To plot the signal properties and perform reconstruction with the 4-receiver data, the following command should be run. The default settings in this function load the 4-receiver data, and have the setting on for both plotting properties and performing the reconstruction.

python Extracting_signal_properties_script_efficient_cluster_v1_1.py

The signal properties as a function of direction given in phi and theta, such as peak frequency and arrival time, are given in files ending in '*_n_1p78_zpol_rec1_5.png'.

The reconstruction results are given in 'Reconstruction_histogram.png', which shows the histogram of the opening angle alpha. The training and validation losses for the neural network are given in 'Train_validation_loss.png'

27-receiver data

The csv file 'remove_intensity_outliers_no_pos_non_zero_freqs.csv' contains the signal properties across all 27 receivers. There were 10,000 runs done in this simulation, with 100 events to a run. The direction of the cascade was varied isotropically, the energy ranged between 1e14 and 1e20 eV, and the position varied within a cube of 3km  per side (from -1.5km to 1.5km).

With the data provided it is possible to do reconstruction only. 