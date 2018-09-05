#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import jams
import numpy as np
import copy
import datetime
import pandas as pd
import soundfile as psf
import librosa.util
import os

from .utils import find_files_in_dirs
from .utils import read_audio


PERC_VOICE_SET = ['kicks',
                  'snares',
                  'snares_rim',
                  'crashes',
                  'rides',
                  'open_hats',
                  'closed_hats',
                  'low_toms',
                  'mid_toms',
                  'high_toms',
                  'congas_bongos',
                  'claps',
                  'bells',
                  'claves']


REDUCED_PERC_VOICE_SET = ['kicks',
                          'snares',
                          'closed_hats']


DEFAULT_MIXING_COEFS = [1.0,] * len(PERC_VOICE_SET)


def select_audio_files_from_directory(audio_directory):
    audio_files = []
    list_of_list_of_files = []

    for perc_voice in PERC_VOICE_SET:
        list_of_list_of_files.append(find_files_in_dirs([os.path.join(audio_directory, perc_voice)]))
        audio_files.append(random.choice(list_of_list_of_files[-1]))

    return audio_files, list_of_list_of_files


def select_audio_files_from_filelist(list_of_list_of_files):
    return [random.choice(l) for l in list_of_list_of_files]


def log_to_linear_amp(x, arange=(-48., 0.)):
    """
    Convert a 0-1 log-scaled signal (whose 0 and 1 levels are defined by `arange`) to linear scale.

    Parameters
    ----------
    x : np.array
        Input signal that ranges from 0. to 1.
    arange : tuple[float]
        The range of the input in dB

    Returns
    -------
    x_linear : np.array
        Linear-scaled x

    Examples
    --------
    >>> log_to_linear_amp(np.array([1.]))
    array([ 1.])

    >>> log_to_linear_amp(np.array([0.5]), arange=(-6., 0.))
    array([ 0.70794578])

    >>> log_to_linear_amp(np.array([0.]), arange=(-6., 0.))
    array([ 0.])

    >>> log_to_linear_amp(0., arange=(-6., 0.))
    0.0
    """
    x_linear = x * (arange[1] - arange[0]) + arange[0]
    x_linear = (10.0**(x_linear/20.)) * (x > 0.)  # make sure 0 is still 0
    return x_linear


def velocity_to_amp(v, arange=(-60, 0)):
    """
    Convert a 0-1 velocity signal (whose 0 and 1 levels are defined by `arange`) to linear scale.

    Parameters
    ----------
    x : np.array
        Input signal that ranges from 0. to 1.
    arange : tuple[float]
        The range of the input in dB

    Returns
    -------
    x_linear : np.array
        Linear-scaled x

    Examples
    --------
    >>> velocity_to_amp(np.array([1.]))
    array([ 1.])

    >>> velocity_to_amp(np.array([1/127.]), arange=(-60., 0.))
    array([ 0.001])

    >>> velocity_to_amp(np.array([0.]), arange=(-6., 0.))
    array([ 0.])

    >>> velocity_to_amp(0., arange=(-6., 0.))
    0.0
    """
    r = 10**((arange[1] - arange[0]) / 20.)
    b = (127. / (126 * (r ** 0.5))) - (1 / 126.)
    m = (1 - b) / 127.0
    return (((127 * m * v) + b) ** 2) * (v > 0.)


def _write_audio(path, y, sample_rate, norm=True):
    """
    Write audio file to disk.

    Parameters
    ----------
    path : str
        File path to write audio file. Extension dictates format.
    y : np.array
        Audio signal array
    sample_rate : int
    norm : bool
        Peak-normalize `y` before writing to disk.

    Returns
    -------
    None
    """
    if norm:
        y /= np.max(np.abs(y))
    psf.write(path, y, int(sample_rate))


def _dict_of_array_to_dict_of_list(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            new_dict[k] = v.tolist()
        else:
            new_dict[k] = v
    return new_dict


def _dict_of_list_to_dict_of_array(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, list):
            new_dict[k] = np.array(v)
        else:
            new_dict[k] = v
    return new_dict


def _repeat_annotations(ann, repetitions):
    frames = [ann.data]

    for i in range(1, repetitions):
        frame = copy.deepcopy(ann.data)
        frame.time += datetime.timedelta(seconds=(ann.duration * i))
        frames.append(frame)

    frame = pd.DataFrame(pd.concat(frames, ignore_index=True))
    ann.data = jams.JamsFrame.from_dataframe(frame)
    ann.duration *= repetitions
    return ann


def _rotate_annotations(ann, rotation_sec):
    dur = datetime.timedelta(seconds=ann.duration)
    ann.data.time += datetime.timedelta(seconds=rotation_sec)
    ann.data.ix[ann.data.time >= dur, 'time'] -= dur
    ann.data = jams.JamsFrame.from_dataframe(ann.data.sort_values('time').reset_index(drop=True))
    return ann


def _trim_annotations(ann, min_sec, max_sec):
    ann.data.time -= datetime.timedelta(seconds=min_sec)
    max_sec -= min_sec
    ann.data = ann.data.ix[(ann.data.time >= datetime.timedelta(seconds=0)) &
                           (ann.data.time <= datetime.timedelta(seconds=max_sec))]
    dur = max(max_sec - min_sec, ann.data.time.max().total_seconds())
    ann.duration = dur
    ann.data = jams.JamsFrame.from_dataframe(ann.data)
    return ann


class PercussionSynthesizer(object):
    """
    Synthesize percussive audio from annotations

    Attributes
    ----------
    midi_file_path : str
        The path to the MIDI file
    """

    def __init__(self, perc_voice_set=PERC_VOICE_SET):
        self._sample_rate = 44100.
        self._min_amplitude = -60.
        self._mixing_coeffs = DEFAULT_MIXING_COEFS
        self._audio_files = None
        self._jam = None
        self._db = None
        self._downbeats = None
        self._beats = None
        self._onsets = None
        self._has_large_vocab = True
        self._perc_voice_set = perc_voice_set

    @property
    def num_active_voices(self):
        return len([len(o) > 0 for o in self.get_onsets()])

    @property
    def time_signature(self):
        raise NotImplementedError()

    @property
    def ts_num(self):
        raise NotImplementedError()

    @property
    def ts_denom(self):
        raise NotImplementedError()

    @property
    def tempo(self):
        raise NotImplementedError()

    @property
    def has_large_vocab(self):
        return self._has_large_vocab

    @property
    def perc_voice_set(self):
        return self._perc_voice_set

    @property
    def sample_rate(self):
        """
        Sampling rate. Read-only. Default is 44100.
        """
        return self._sample_rate

    @property
    def mixing_coeffs(self):
        """
        Mixing coefficients that determine the relative level for each synthesized pattern. Read-only.
        Default is [1./num_patterns, 1./num_pattern, ... ].
        """
        if self._mixing_coeffs is None:
            return np.ones(self.num_active_voices) / self.num_active_voices
        else:
            return self._mixing_coeffs

    def get_onsets(self, velocity=100):
        """
        Get onset times and amplitudes for percussion voices

        Returns
        -------
        onsets : list[list]
        """
        onsets = [(v, []) for v in self.perc_voice_set]

        for k,voice_onset_times in enumerate(self._onsets):
            for onset_time in voice_onset_times:
                onsets[k][1].append(dict(time=onset_time, velocity=velocity/127.))

        return onsets

    def get_downbeats(self):
        return self._downbeats

    def get_beats(self):
        return self._beats

    @property
    def audio_files(self):
        """
        The audio files used to synthesized the patterns in their respective order by pattern.
        Read-only. Default is None.
        """
        return self._audio_files

    @property
    def jam(self):
        """
        The JAMS annotations for the current pattern. Read-only. Default is None.
        """
        return self._jam

    @property
    def duration(self):
        return self._duration

    @property
    def min_amplitude(self):
        """
        The minimum amplitude of an onset in dB. Read-only. Default is -48.
        """
        return self._min_amplitude

    @staticmethod
    def from_onsets(duration,
                    onset_activations_array=None,
                    onset_times_list=None,
                    sampling_interval=None,
                    has_beats=True,
                    perc_voice_set=PERC_VOICE_SET,
                    **kwargs):
        """
        Synthesize from either an array of onset activations or a  list of onset times. Last two rows are downbeat
        and beat if `has_beats` is defined.

        Parameters
        ----------
        onset_activations_array : np.array
        onset_times_list : list[np.array]
        sampling_interval : float
        **kwargs : dict
            Additional keywors are passed to the peak picking algorithm

        Returns
        -------
        PercussionSynthesizer
        """
        if onset_activations_array is None and onset_times_list is None:
            raise Exception("Must define either onset_activations_array or onset_times_list")

        if sampling_interval is None:
            sampling_interval = (int(round(0.01 * 22050)) / float(22050))

        if onset_activations_array is not None:
            onset_times_list = []
            for i in range(onset_activations_array.shape[1]):
                peaks = librosa.util.peak_pick(onset_activations_array[:, i], **kwargs)
                onset_times_list.append(peaks * sampling_interval)

        perc_loop_synth = PercussionSynthesizer(perc_voice_set=perc_voice_set)
        if has_beats:
            perc_loop_synth._downbeats = onset_times_list[-2]
            perc_loop_synth._beats = onset_times_list[-1]
            perc_loop_synth._onsets = onset_times_list[:-2]
        else:
            perc_loop_synth._onsets = onset_times_list
        perc_loop_synth._duration = duration

        return perc_loop_synth

    @staticmethod
    def from_jams(jams_file_path=None, jam=None):
        """
        Load a pattern from a JAMS file and instantiate the PercussionLoopSynthesizer

        Parameters
        ----------
        jams_file_path : str
            Path to jams file
        jam : JAM
            jam. Either `jams_file_path` or `jam` must be defined.

        Returns
        -------
        rhythm_synth : RhythmSynthesizer
        """
        if jams_file_path is not None:
            jam = jams.load(jams_file_path)
        elif jam is not None:
            pass
        else:
            raise Exception('Either `jams_file_path` or `jam` must be defined.')

        num_patterns = len([ann for ann in jam.search(namespace='onset') if not ann.sandbox.base_pattern])
        sample_rate = jam.sandbox.sample_rate
        min_amplitude = jam.sandbox.min_amplitude
        mixing_coeffs = []
        audio_files = []

        for i in range(num_patterns):
            onset_ann = jam.search(pattern_index=i)[0]
            mixing_coeffs.append(onset_ann.sandbox.mixing_coeff)
            audio_files.append(onset_ann.sandbox.audio_source)

        perc_loop_synth = PercussionSynthesizer()

        perc_loop_synth._min_amplitude = min_amplitude
        perc_loop_synth._has_large_vocab = jam.sandbox.has_large_vocab
        perc_loop_synth._sample_rate = sample_rate
        perc_loop_synth._jam = jam
        perc_loop_synth._mixing_coeffs = mixing_coeffs
        perc_loop_synth._audio_files = audio_files
        perc_loop_synth._perc_voice_set = jam.sandbox.perc_voice_set
        perc_loop_synth._duration = jam.file_metadata.duration

        return perc_loop_synth

    def generate_jam(self,
                     output_jams_file=None,
                     audio_files=None,
                     audio_directory=None,
                     mixing_coeffs=None,
                     sample_rate=None,
                     min_amplitude=None,
                     duration_sec=None,
                     additional_onset_sandbox_info=None,
                     additional_global_sandbox_info=None):
        """
        Parameters
        ----------
        output_jams_file : str
            If not None, write jams file to `output_jams_file`.
        audio_files : list[str]
            A list of the audio_files to use for rendering the rhythm patterns. These should be ordered according to the
            pattern (e.g., if the patterns were constructed to be from low frequency to high frequency, a bass drum
            might be first in the list). If not None, overwrite self.audio_files. Default is None.
        audio_directory : str
            If audio_files is None, then this must be set. It will randomly sample from subdirectories in PERC_VOICE_SET
            Default is None
        mixing_coeffs : np.array
            The coefficients specifying the mixing levels for the patterns. If not None, overwrite self.mixing_coeffs.
            Default is None.
        sample_rate : float
            Sampling rate of output. If not None, overwrite self.sample_rate. Default is None.
        min_amplitude : float
            The minimum amplitude to render for the softest sound above 0 in log amplitude. If not None, overwrite
            self.min_amplitude. Default is None.
        duration_sec : float
            If not None, extend the rhythm pattern to the target duration. Default is None.
        additional_onset_sandbox_info : list[dict]
            If not None, then save this additional info in the sandbox for the corresponding onset annotation.
            Default is None.
        additional_global_sandbox_info: dict
            If not None, then save this additional info in the top-level sandbox of the JAM. Default is None.

        Returns
        -------
        jam : jams.JAM

        """
        if mixing_coeffs is not None:
            self._mixing_coeffs = mixing_coeffs

        if audio_files is None and audio_directory is None:
            raise Exception("Either `audio_files` or `audio_directory` must be defined.")

        if audio_files is None:
            audio_files = select_audio_files_from_directory(audio_directory)[0]

        self._audio_files = audio_files
        assert (len(self.audio_files) == len(self.perc_voice_set))

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if min_amplitude is not None:
            self._min_amplitude = min_amplitude

        # make JAM structure
        jam = jams.JAMS()
        jam.file_metadata.duration = self.duration
        jam.sandbox.sample_rate = self.sample_rate
        # jam.sandbox.time_signature = (self.ts_num, self.ts_denom)
        jam.sandbox.min_amplitude = self.min_amplitude
        jam.sandbox.has_large_vocab = self.has_large_vocab
        jam.sandbox.perc_voice_set = self.perc_voice_set
        if additional_global_sandbox_info is not None:
            jam.sandbox.update(**additional_global_sandbox_info)

        if self.get_beats() is not None:
            # write beat positions to jams
            base_pattern = False
            beat_ann = jams.Annotation(namespace='beat', time=0, duration=jam.file_metadata.duration)
            beat_ann.sandbox = jams.Sandbox(base_pattern=base_pattern)
            for k, t in enumerate(self.get_beats()):
                beat_ann.append(time=t,
                                duration=0.0,
                                value=k)
            jam.annotations.append(beat_ann)

        # write tempo to jams
        # tempo_ann = jams.Annotation(namespace='tempo', time=0, duration=jam.file_metadata.duration)
        # tempo_ann.append(time=0, duration=jam.file_metadata.duration, value=self.tempo, confidence=1.0)
        # jam.annotations.append(tempo_ann)

        # write onsets for each rhythm pattern
        onsets = self.get_onsets()
        for i in range(len(self.perc_voice_set)):
            base_pattern = False
            onsets_ann = jams.Annotation(namespace='onset', time=0, duration=jam.file_metadata.duration)
            onsets_ann.sandbox = jams.Sandbox(pattern_index=i,
                                              perc_voice=self.perc_voice_set[i],
                                              audio_source=self.audio_files[i],
                                              mixing_coeff=self.mixing_coeffs[i],
                                              base_pattern=base_pattern)
            if additional_onset_sandbox_info is not None:
                onsets_ann.sandbox.update(**additional_onset_sandbox_info[i])

            # write onsets
            for onset in onsets[i][1]:
                onsets_ann.append(time=onset['time'],
                                  value=velocity_to_amp(onset['velocity'], (self.min_amplitude, 0.)),
                                  duration=0)

            jam.annotations.append(onsets_ann)

        self._jam = jam
        if output_jams_file is not None:
            jam.save(output_jams_file)

        return jam

    def synthesize(self,
                   output_file,
                   **kwargs):
        """
        Synthesize the patterns from self.jam. If self.jam does not exist, pass in the arguments to `generate_jam`.

        Parameters
        ----------
        output_file : str
            The path where the rendered and mixed output signal will be written. If None, no file will be written.
        kwargs : additional keyword arguments
            Additional keyword arguments to pass to `generate_jam` if self.jam is None.

        Returns
        -------
        rhythm_audio : np.array
            The mixed output_signal
        unmixed_rhythm_audio: np.array
            An MxN array where M is the number of rhythm patterns and N is the length of the measure in samples

        See Also
        --------
        generate_jam
        """
        if self.jam is None or len(kwargs) > 0:
            self.generate_jam(**kwargs)

        onset_anns = [ann for ann in self.jam.search(namespace='onset') if not ann.sandbox.base_pattern]

        unmixed_rhythm_audio = np.zeros([len(self.perc_voice_set), int(np.ceil(self.duration * self.sample_rate))])

        for k, onset_ann in enumerate(onset_anns):
            # load audio file
            sample, _ = read_audio(self.audio_files[k], self.sample_rate, mono=True)
            sample_length = sample.shape[0]

            # render samples at onsets
            for j in range(len(onset_ann.data.index)):
                start_idx = int(np.round(onset_ann.data.ix[j, 'time'].total_seconds() * self.sample_rate))
                start_idx = min(start_idx, unmixed_rhythm_audio.shape[1]-1)
                stop_idx = start_idx + sample_length
                stop_idx = min(stop_idx, unmixed_rhythm_audio.shape[1]-1)
                unmixed_rhythm_audio[k, start_idx:stop_idx] = onset_ann.data.ix[j, 'value'] * sample[:(stop_idx -
                                                                                                       start_idx)]

        rhythm_audio = np.dot(self.mixing_coeffs, unmixed_rhythm_audio)

        if output_file is not None:
            _write_audio(output_file, rhythm_audio, int(self.sample_rate))

        return rhythm_audio, unmixed_rhythm_audio
