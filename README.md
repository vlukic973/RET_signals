# Radar Echo Telescope (RET) signal properties and reconstruction

This repository contains code for generating RadioScatter signals, plotting signal properties, and reconstruction. 

## Generating voltage-time traces

The script 'Generate_RadioScatter_macros.py' generates the macro files needed to run RadioScatter. It allows the user to specify the name of the macros, how many files they want to generate, whether to vary or keep constant the position, direction and energy of the primary. The specific example provided generates data for 27 receivers, with energy at 1e18 eV, and position (0,100,10) m, with 10000 runs in total. 

The generated macros, containing the file string twenty_seven_rec_macro_50_MHz_n_1p78_case_1_* should then run on cluster, where RadioScatter can be run on them (through Geant4), using the following example submission file.

The case_1_macros.submit script can be submitted to the ht_condor cluster using the following command:

`condor_submit case_1_macros.submit`

The submission file must be run from a folder that has the nrt or slac executable. For the example provided it submits 10000 jobs, using the $(Process) command (specifying queue 10000). What must be specified in the submission file is the location of the macros on the cluster, the desired location of the resulting .root files, as well as the name of the .log, .out and .err files.

If the files have finished running successfully on the cluster, they will produce a '*_summary.root' file. 

Assuming all the files have run successfully, one can extract the voltage-time traces from the root files.

To show how this is done by example, use a script like the provided Extract_individual_voltage_time_rec1_case_1.sh, which generates the voltage-time traces for receiver 1. There will be 10000 text files created, where there is one voltage-time trace per text file. Similarly, the other 26 scripts can be run to generate data for the remaining 26 receivers.

These scripts should all have the following pattern 'Extract_individual_voltage_time_rec\${i}\_case_1.sh', where i goes from 1 to 27. The shell script calls a function written in C, to extract the individual voltage-time traces. They are done individually because it can happen that occasionally a single voltage-time trace is truncated, therefore it needs to be padded.  

Within this script, there are 10,000 lines that look like the following:

`root -b -q 'textRunThrough.C("/pnfs/iihe/radar/store/user/vlukic/case_1/twenty_seven_rec_macro_50_MHz_n_1p78_case_1_0.root",0,1,0,0)' >> /pnfs/iihe/radar/store/user/vlukic/case_1/voltage_time_traces/receiver_1_27_recs_vary_pos_dir_energy_0.txt`

That particular line (the first one in the file) will extract the voltage-time trace for receiver 1, run 1 of 10,000.

To generate the remaining lines it is possible to write a bash loop.

The shell scripts for extracting the voltage-time traces can be run through a .submit file. For example, the one corresponding to Extract_individual_voltage_time_rec1_case_1.sh is Extract_individual_voltage_time_rec1_case_1.submit

If the position of the cascade is varied in the simulation, it is also necessary to extract the time-vector, since this will change given the changing position. To do this, run a script like the provided one 'Extract_time_rec1.sh'. It works in the same way as for extracting the voltage-time traces, but instead of extracting the Y-axis values (giving voltage), it extracts the X-axis values (giving time). In fact, it extracts only the first value of the time vector, since the rest is known, given the length of the signal and the sampling rate. Check that the time vector generated is the same length as the voltage-time trace.

After the voltage-time traces, and (if position is varied) time vector are generated, it is possible to use a python script to generate numpy arrays of the data.

This can be done using the script 'Save_np_arrays_of_voltage_time_traces.py', which produces a .npy array for every receiver individually. 

These arrays can then be read into python to produce the signal properties and perform reconstruction

## Extracting signal properties and reconstruction

For plotting the signal properties and performing reconstruction, there is data provided for 4 receivers (990 runs with varying direction only), and 27 receivers (10000 runs with varying energy, position and direction). It is possible to plot the signal properties and do reconstruction for the 4 receiver data. For the 27 receiver data, reconstruction only is possible.

The code (Plot_signal_properties_and_reconstruct.py) is able to plot signal properties and/or perform reconstruction on data generated using RadioScatter. The signal properties derived are the intensity, arrival time at receivers, peak frequency, bandwidth, rise and fall time maximum amplitude at transmit frequency, amplitude at half, at and twice the transmit frequency.

Prior to running the python code, please use the RET_signals_env.yaml file provided by supplying the following commands on the command-line:

`conda env create -f RET_signals_env.yaml`

`conda activate RET_signals`

Also download the other data provided, given in the text and csv files.

## 4-receiver data

<img width="700" alt="four_receiver_setup_v3" src="https://user-images.githubusercontent.com/42998963/194073362-01943ccf-f2ef-4254-a974-daf4daa20d16.png">

The data provided includes the signals generated across 4 receivers in the 'cross' setup, beginning with '_numbers_only2.txt'. There are 990 runs in total, each with an associated direction (given in four_recs_rand_dir.csv) and signal. The signals in the file '_numbers_only2.txt' are appended one after the other.

The corresponding values on the time axis are given by '_time.txt'

With the data provided it is possible to plot the signal properties and/or perform reconstruction.

To plot the signal properties and perform reconstruction with the 4-receiver data, the following command should be run. The default settings in this function load the 4-receiver data, and have the setting on for both plotting properties and performing the reconstruction.

`python Plot_signal_properties_and_reconstruct.py`

The signal properties as a function of direction given in phi and theta, such as peak frequency and arrival time, are given in files ending in '_n_1p78_zpol_rec1_5.png'.

The reconstruction results are given in 'Reconstruction_histogram.png', which shows the histogram of the opening angle alpha. The training and validation losses for the neural network are given in 'Train_validation_loss.png'

## 27-receiver data

<img width="900" alt="Twenty_seven_rec_setup_with_start_pos" src="https://user-images.githubusercontent.com/42998963/194078202-2aa2254c-a6c9-4b0d-836b-3d9e30a43136.png">

The csv file 'remove_intensity_outliers_no_pos_non_zero_freqs.csv' contains the signal properties across all 27 receivers. There were 10,000 runs done in this simulation, with 100 events to a run. The direction of the cascade was varied isotropically, the energy ranged between 1e14 and 1e20 eV, and the position varied within a cube of 3km  per side (from -1.5km to 1.5km).

With the data provided it is possible to do reconstruction only. The command to do so is given below

`python Plot_signal_properties_and_reconstruct.py --reconstruct_27_recs 'True' --plot_properties 'False' --reconstruct_4_recs 'False'`
