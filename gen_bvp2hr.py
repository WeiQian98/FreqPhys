import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from scipy.io import loadmat
import scipy
from scipy import misc, io
from scipy.signal import butter
from scipy.sparse import spdiags

import config


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def detrend(signal, Lambda):
    signal_length = signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def calculate_HR(pxx_label, frange_label, fmask_label):
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return ground_truth_HR

def calculate_metric_per_video(labels, signal='pulse', fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.66 / fs * 2, 3.0 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

#     if signal == 'pulse':
#         label_window = detrend(np.cumsum(labels), 100)

    # label_window = detrend(np.cumsum(labels), 100)

    if bpFlag:
        label_window = scipy.signal.filtfilt(b, a, np.double(labels))

    label_window = np.expand_dims(labels, 0)

    N = next_power_of_2(label_window.shape[1])
    # Labels FFT
    f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=3600, detrend=False)
    if signal == 'pulse':
        fmask_label = np.argwhere((f_label >= 0.66) & (f_label <= 3.0))  # regular Heart beat are 0.75*60 and 2.5*60
    else:
        fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
    label_window = np.take(f_label, fmask_label)
    temp_HR = calculate_HR(pxx_label, label_window, fmask_label)
    return temp_HR

def gen(bvp_path):
    bvp = loadmat(bvp_path)['bvp'].squeeze(0)
    hr_temp = calculate_metric_per_video(bvp)
    HR = {'hr':hr_temp}
    hr_path = os.path.join(f'{config.filter_path}', bvp_path.split('/')[-3], bvp_path.split('/')[-2])
    io.savemat(hr_path + f"/hr.mat", HR)


if __name__ == "__main__":
    bvp_path_list = glob.glob(f'{config.filter_path}/*/*/bvp.mat')
    with parallel_backend("loky"):
        Parallel(n_jobs=-1)(delayed(gen)(bvp_path) for bvp_path in tqdm(bvp_path_list))

