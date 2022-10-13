## This script generates some macros that are used to run RadioScatter within Geant4. The example below is for a primary with 100 events to a run, 10000 runs in total, 27 receivers, fixed position at (0,100,10) m and fixed energy of 1e18 eV

import numpy as np

########### 27 receiver setup case 1

n_entries=10001
    
for i in range(0,n_entries,1):
    print("i="+str(i))
    f = open(r"twenty_seven_rec_macro_50_MHz_n_1p78_case_1_"+str(i)+".mac", "w")
    f.write("/run/initialize \n")
    f.write("/tracking/verbose 0 \n")
    f.write("/gps/particle e- \n")
    f.write("/gps/energy 100.0 GeV \n")
    f.write("/gps/position 0 100 10 m \n")
    f.write("/RS/setNRx 27 \n")
    f.write("/RS/setTargetEnergy 1.0e+18 eV \n")
    f.write("/RS/setTxPos 0 0 0 m \n")
    f.write("/RS/setRxPos 250 0 -20 m \n")
    f.write("/RS/setRxPos 250 0 0  m \n")
    f.write("/RS/setRxPos 250 0 20  m \n")
    f.write("/RS/setRxPos 191.511 160.697 -20  m \n")
    f.write("/RS/setRxPos 191.511 160.697   0  m \n")
    f.write("/RS/setRxPos 191.511 160.697  20  m \n")
    f.write("/RS/setRxPos 43.412 246.202 -20  m \n")
    f.write("/RS/setRxPos 43.412 246.202   0  m \n")
    f.write("/RS/setRxPos 43.412 246.202  20  m \n")
    f.write("/RS/setRxPos -125 216.506 -20  m \n")
    f.write("/RS/setRxPos -125 216.506   0  m \n")
    f.write("/RS/setRxPos -125 216.506  20  m \n")
    f.write("/RS/setRxPos -234.923 85.505 -20  m \n")
    f.write("/RS/setRxPos -234.923 85.505   0  m \n")
    f.write("/RS/setRxPos -234.923 85.505  20  m \n")
    f.write("/RS/setRxPos -234.923  -85.505   -20  m \n")
    f.write("/RS/setRxPos -234.923  -85.505   0  m \n")
    f.write("/RS/setRxPos -234.923  -85.505   20  m \n")
    f.write("/RS/setRxPos -125  -216.506  -20  m \n")
    f.write("/RS/setRxPos -125  -216.506    0  m \n")
    f.write("/RS/setRxPos -125  -216.506   20  m \n")
    f.write("/RS/setRxPos 43.412 -246.202  -20  m \n")
    f.write("/RS/setRxPos 43.412 -246.202  0  m \n")
    f.write("/RS/setRxPos 43.412 -246.202  20  m \n")
    f.write("/RS/setRxPos 191.511 -160.697  -20  m \n")
    f.write("/RS/setRxPos 191.511 -160.697  0  m \n")
    f.write("/RS/setRxPos 191.511 -160.697  20  m \n")
    f.write("/RS/setScaleByEnergy 1 \n")
    f.write("/RS/setTxFreq 0.05 \n")
    f.write("/RS/setRxSampleRate 5 \n")
    f.write("/RS/setTxVoltage 223.607 \n")
    f.write("/RS/setPolarization 0 1 0 \n")
    f.write("/RS/setIndexOfRefraction 1.78 \n")
    f.write("/RS/setRecordWindowLength 400 \n")
    f.write("/RS/setShowCWFlag 0 \n")
    f.write("/RS/setMakeSummary 1 \n")
    f.write("/RS/setPlasmaLifetime 10 \n")
    f.write("/RS/setFillByEvent 0 \n")
    f.write("\n")
    f.write("\n")
    phi = random.uniform(0,2*np.pi)
    costheta = random.uniform(-1,1)
    theta = np.arccos(costheta)
    r=1
    x_dir = r * np.sin(theta)*np.cos(phi)
    y_dir = r * np.sin(theta)*np.sin(phi)
    z_dir = r * np.cos(theta)
    f.write('/gps/direction     '+str(np.round(x_dir,3))+'     '+str(np.round(y_dir,3))+'     '+str(np.round(z_dir,3))+"\n")
    f.write('/run/beamOn          100'+"\n")

#### To vary the starting position of the cascade, for example within a box of (3km)**3, do the following (don't forget to delete the original position above e.g as given by f.write("/gps/position 0 100 10 m \n")):

# xpos= random.uniform(-1500, 1500)
# ypos= random.uniform(-1500, 1500)
# zpos= random.uniform(-1500, 1500)
# f.write('/gps/position     '+str(np.round(xpos,3))+'     '+str(np.round(ypos,3))+'     '+str(np.round(zpos,3))+"   m   \n")

#### To vary the energy of the cascade, for example from energies ranging between 1e14 and 1e20, do the following (also don't forget to delete the original energy above e.g. as given by f.write("/RS/setTargetEnergy 1.0e+18 eV \n"))

# E_array_power.append(random.uniform(14, 20))
# E_vals_array=np.power(10,E_array_power)

# f.write('/RS/setTargetEnergy     ' + str(E_vals_array[i]) + "   eV  \n")
