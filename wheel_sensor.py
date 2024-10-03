#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:04:45 2024

@author: amandaschott
"""
from pathlib import Path
import pyopenephys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5 import QtWidgets
from open_ephys.analysis import Session
# custom modules
import pyfx

WHEEL_DIAMETER = 14       # diameter (cm) of the main wheel
N_WHEEL_INCREMENTS = 256  # number of increments in one wheel rotation
N_BELT_INCREMENTS = 893   # number of increments in one full "lap" of the belt
AUDIO_SPAN = 2            # area (cm) covered by each unique audio cue

WHEEL_CIRCUMFERENCE = WHEEL_DIAMETER * np.pi               # circumference (cm) of the main wheel
INCREMENT_SIZE = WHEEL_CIRCUMFERENCE / N_WHEEL_INCREMENTS  # distance (cm) per increment
BELT_LENGTH = INCREMENT_SIZE * N_BELT_INCREMENTS           # length (cm) of the belt

N_AUDIO_CUES = int(BELT_LENGTH / AUDIO_SPAN)  # number of unique audio cues
AUDIO_SPAN += (BELT_LENGTH % AUDIO_SPAN) / N_AUDIO_CUES  # evenly spaced intervals

BINS = np.arange(AUDIO_SPAN, N_AUDIO_CUES * AUDIO_SPAN, AUDIO_SPAN)


def get_Hall_edges(Hall_signal):
    idifs = np.where(np.diff(Hall_signal) != 0)[0]
    if Hall_signal[0] == 1:   # Hall sensor ON at recording start
        idifs = np.insert(idifs, 0, 0)
    if Hall_signal[-1] == 1:  # Hall sensor ON at recording end
        idifs = np.append(idifs, len(Hall_signal)-1)
        
    # get edge idx pairs (istart,iend) for each Hall sensor activation
    if len(idifs) >= 2 : edges = np.split(idifs, len(idifs)/2)
    else               : edges = np.array((), dtype=Hall_signal.dtype)
    return edges


#%%
ppath = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell/wheel_data/multiple_belt_rotations'

### PI SIGNALS (multiple full rotations of extra-long belt)
# load timestamps, calculate sampling rate
rel_times = np.load(os.path.join(ppath, 'timestamps.npy'))
dur = rel_times[-1] - rel_times[0]
times = np.linspace(0, dur, len(rel_times))
pi_fs = len(times) / dur

# load signals
A_signal = np.load(os.path.join(ppath, 'A_SIGNAL.npy'))
B_signal = np.load(os.path.join(ppath, 'B_SIGNAL.npy'))
Hall_signal = np.load(os.path.join(ppath, 'HALL_SIGNAL.npy'))
Revs = np.load(os.path.join(ppath, 'REVOLUTIONS.npy')).astype('float32')

# get edges of Hall signals
iedges = get_Hall_edges(Hall_signal)   # (istart, iend) for each signal
tedges = [ie / pi_fs for ie in iedges] # (tstart, tend) for each signal
istarts, iends = list(zip(*iedges))


#%%
from ephys import plot_signals
Revs_nan = Revs.copy()
Revs_nan[np.where(Revs_nan == 0)[0]] = np.nan

DDICT = dict(A=A_signal, B=B_signal, Hall=Hall_signal, rev=Revs_nan)
fig,ax,_ = plot_signals(times, DDICT, fs=pi_fs, hide=[], mkr=None, mkrdict=dict(rev='o'),
                colordict=dict(A='blue', B='red', Hall='green', rev='purple'), t_init=28, twin=3)
for ie in iedges:
    ax.axvspan(*times[ie], color='green', alpha=0.5, zorder=3)
#%%

##############################################################################
#################                   SCRAPS                   #################
##############################################################################


# get number of rotary encoder increments from initial position to Hall sensor
# first_istart, first_iend = edges[0]
# revs2hall = Revs[0 : first_istart].sum()

# [0, 2.02, 4.04, 6.06 ... 153.42]


# #%%
# FAKE_AUDIO = np.arange(0, N_AUDIO_CUES)

# # "live" position tracking
# i = 0

# #POS_FROM_START = 0
# DIST_FROM_HALL = None
# HALL_ON = False

# while DIST_FROM_HALL is None:
#     hall = Hall_signal[i]
#     #POS_FROM_START += Revs[i]
#     if hall == 1:
#         print("found the hall sensor!" + os.linesep)
#         DIST_FROM_HALL = 0
#         HALL_ON = True
#     i += 1

# #COMPLETE_LOOPS = 0
# # track number of increments
# #DIST_FROM_HALL = 0
# while True:
# #while DIST_FROM_HALL > -2.5:
#     rev = Revs[i]
#     hall = Hall_signal[i]
#     #DIST_FROM_HALL += INCREMENT_SIZE * Revs[i]
    
#     # belt entered Hall field
#     if HALL_ON == False and hall == 1:
#         print('entered Hall field!')
#         POS_FROM_HALL = 0  # reset position with respect to Hall sensor
#     elif HALL_ON == True and hall == 0:
#         print('exited Hall field!')
#     HALL_ON = bool(hall)
    
#     DIST_FROM_HALL += rev * INCREMENT_SIZE  # distance (cm) from Hall sensor
#     #if rev != 0:
        
#     i += 1
    
    
    
    
    
    
#     #_dist = POS_FROM_HALL * INCREMENT_SIZE  # distance (cm) from Hall sensor


# #%%

        

# #%%
# ### ADC SIGNALS FROM 2-PHOTON SETUP
# directory = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell/2_photon/JG002/JG002_2024-07-18_11-41-56/Record Node 103'
# session = Session(directory)
# recording = session.recordings[0]
# continuous = recording.continuous[0]

# # memmap with signal data stored as nsamples x channels (e.g. (8358637, 8))
# d = continuous.samples.T
# # vector of nsamples indexes (position since start of acquisition)
# idxs = continuous.sample_numbers
# # times corresponding with recording indexes
# timestamps = continuous.timestamps
# dur = timestamps[-1] - timestamps[0]
# times = np.linspace(0, dur, len(timestamps))

# # metadata (e.g. sample_rate (10000.0), num_channels (8), and channel_names [ADC1,ADC2,...ADC8]
# meta = continuous.metadata
# fs = meta['sample_rate']  # raw sampling rate

# A = d[1]
# B = d[2]
# HALL = d[3]

# # create A and B signal trains (0=LOW, 1=HIGH)
# A_train = np.zeros(len(times), dtype='int8')
# B_train = np.zeros(len(times), dtype='int8')
# Hall_train = np.zeros(len(times), dtype='int8')

# A_train[np.where(A < np.percentile(A, 99.9))[0]] = 1
# #A_train[np.where(A < A.max())[0]] = 1
# B_train[np.where(B < B.max())[0]] = 1

# #%%
# ppath = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell'

# ### PI SIGNALS (no Hall)
# gpio_df = pd.read_csv(os.path.join(ppath, 'gpio_data.csv'))
# rel_times = gpio_df.timestamp.values
# dur = rel_times[-1] - rel_times[0]
# times = np.linspace(0, dur, len(rel_times))
# A_signal = gpio_df.A.values
# B_signal = gpio_df.B.values
# pi_fs = len(times) / dur

