#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import librosa
import scipy.signal
import numpy as np

from .utils import read_audio

FRAME_INTERVAL = 0.01  # s


def cq_matrix(bins_per_octave, num_bins, f_min, fft_len, sr):
    """
    Compute center frequencies of the log-spaced filterbank

    Parameters
    ----------
    bins_per_octave : int
    num_bins : int
    f_min : float
    fft_len : int
    sr : float

    Returns
    -------
    c_mat
    """
    # note range goes from -1 to bpo*num_oct for boundary issues
    f_cq = f_min * 2 ** ((np.arange(-1, num_bins+1)) / bins_per_octave)
    # centers in bins
    kc = np.round(f_cq * (fft_len / sr)).astype(int)
    c_mat = np.zeros([num_bins, int(np.round(fft_len / 2))])
    for k in range(1, kc.shape[0]-1):
        l1 = kc[k]-kc[k-1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k+1]-kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack([w1[0:l1], w2[l2:]])  # concatenate two halves
        c_mat[k-1, kc[k-1]:(kc[k+1]+1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat


def onset_detection_fn(x, f_win_size, f_hop_size, f_bins_per_octave, f_octaves, f_fmin, sr, mean_filter_size):
    """
    Filter bank for onset pattern calculation
    """
    # calculate frequency constant-q transform
    f_win = scipy.signal.hanning(f_win_size)
    x_spec = librosa.stft(x,
                          n_fft=f_win_size,
                          hop_length=f_hop_size,
                          win_length=f_win_size,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    f_cq_mat = cq_matrix(f_bins_per_octave, f_octaves * f_bins_per_octave, f_fmin, f_win_size, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])

    # subtract moving mean
    b = np.concatenate([[1], np.ones(mean_filter_size, dtype=float) / -mean_filter_size])
    od_fun = scipy.signal.lfilter(b, 1, x_cq_spec, axis=1)

    # half-wave rectify
    od_fun = np.maximum(0, od_fun)

    # post-process OPs
    od_fun = np.log10(1 + 1000*od_fun)
    return od_fun, x_cq_spec


def extract_features(audio_file_path, sr=22050, channel=1):
    x, sr = read_audio(audio_file_path, mono=True, sr=sr)

    f_win_size = 1024
    f_hop_size = int(round(FRAME_INTERVAL * sr))
    f_bins_per_octave = 8
    f_octaves = 8
    f_fmin = 40
    mean_filter_size = 22

    # normalize
    x /= np.max(np.abs(x))

    od_fun, x_cq_spec = onset_detection_fn(x,
                                           f_win_size,
                                           f_hop_size,
                                           f_bins_per_octave,
                                           f_octaves,
                                           f_fmin,
                                           sr,
                                           mean_filter_size)

    logf_stft = librosa.power_to_db(x_cq_spec).astype('float32')
    od_fun = np.abs(od_fun).astype('float32')

    # reshape for model
    ms_input_array = np.moveaxis(logf_stft, 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)
    os_input_array = np.moveaxis(od_fun, 1, 0)
    os_input_array = np.clip(os_input_array / 2.25, 0, 1)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    return ms_input_array, os_input_array, sr