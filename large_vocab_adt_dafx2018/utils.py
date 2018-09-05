#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa
import resampy
import soundfile as psf


def reduce_voices(onsets, mixing_matrix):
    """
    Reduce the voices given a mixing matrix

    Parameters
    ----------
    onsets
    mixing_matrix

    Returns
    -------
    np.array
    """
    output = np.zeros([onsets.shape[0], len(mixing_matrix)])
    for k,voices in enumerate(mixing_matrix):
        output[:, k] = np.max(onsets[:, voices], axis=1)
    return output


def read_audio(filepath, sr=None, mono=True, peak_norm=False):
    """
    Read audio

    Parameters
    ----------
    filepath
    sr
    mono

    Returns
    -------
    y, sr
    """
    try:
        y, _sr = psf.read(filepath)
        y = y.T
    except RuntimeError:
        y, _sr = librosa.load(filepath, mono=False, sr=None)

    if sr is not None and sr != _sr:
        y = resampy.resample(y, _sr, sr, filter='kaiser_fast')
    else:
        sr = _sr

    if mono:
        y = librosa.to_mono(y)

    if peak_norm:
        y /= np.max(np.abs(y))

    return y, sr


def find_files_in_dirs(dirs, extensions=('.wav', '.mp3', '.aif', '.aiff', '.flac')):
    """
    Find all files in the directories `dir` and their subdirectories with `extensions`, and return the full file path

    Parameters
    ----------
    dirs : list[str]
    extensions : list[str]

    Returns
    -------
    files : list[str]
    """
    found_files = []

    for adir in dirs:
        for root, subdirs, files in os.walk(adir):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    found_files.append(os.path.join(root, f))

    return found_files
