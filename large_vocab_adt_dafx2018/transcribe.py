#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import soundfile as psf

from keras.models import load_model, model_from_json

from large_vocab_adt_dafx2018.percussion_synth import PercussionSynthesizer, PERC_VOICE_SET, REDUCED_PERC_VOICE_SET
from large_vocab_adt_dafx2018.features import extract_features, FRAME_INTERVAL
from large_vocab_adt_dafx2018.model import init_model_config


def transcribe(model_definition_path,
               model_weights_path,
               input_audio_file,
               model_configuration_id,
               sample_audio_files,
               plot=False,
               synthesize=False,
               display_start=0,
               display_stop=500,
               peak_params=None,
               output_sample_rate=44100):
    """
    Transcribe an input audio file, creating a JAMs file with onsets and a synthesizing an audio file.

    Parameters
    ----------
    model_definition_path : str
        Path to model definitinon JSON file.
    model_weights_path : str
        Path to model weights HDF5 file.
    input_audio_file : str
        Path to input audio file
    model_configuration_id : str
        Model configuration ID.
    sample_audio_files : dict
        Audio files used to resynthesize the audio.
        Should have keys for
        ['kicks',
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
    plot : bool
        If True, plot output. Default is False.
    synthesize : bool
        Synthesize audio using `sample_audio_files`
    display_start : int
        The step index to start display when plotting.
    display_stop : int
        The step index to stop display when plotting.
    peak_params : dict
        Dictionary of peak picking parameters. See `librosa.util.peak_pick`.
        Defaults to
            dict(pre_max=2,
                 post_max=2,
                 pre_avg=2,
                 post_avg=2,
                 delta=0.05,
                 wait=5)
    output_sample_rate : int
        Sample rate for output audio files. Default is 44100.

    Returns
    -------
    full_voice_mixed : np.array
        Mixed audio for 14-voice transcription
    full_voice_unmixed : np.array
        Unmixed audio for 14-voice transcription
    reduced_voice_mixed : np.array
        Mixed audio for 3-voice transcription
    reduced_voice_unmixed : np.array
        Unmixed audio for 3-voice transcription
    """

    with open(model_definition_path, 'r') as f:
        json_string = f.read()
        mdl = model_from_json(json_string)

    mdl.load_weights(model_weights_path)

    mn = str(model_configuration_id)
    cfg = init_model_config(model_configuration_id)

    ms_input_array, os_input_array, sr = extract_features(input_audio_file)

    y_hat = mdl.predict([np.array([ms_input_array, ]),
                         np.array([os_input_array, ]), ])

    if plot:
        import matplotlib.pyplot as plt

        x1 = np.squeeze(ms_input_array, 2)
        x2 = np.squeeze(os_input_array, 2)

        y_hat1 = np.squeeze(y_hat[0], 0)
        y_hat2 = np.squeeze(y_hat[1], 0)
        y_hat3 = np.squeeze(y_hat[2], 0)

        plt.figure(figsize=(15, 2.5 * 5))
        plt.subplot(5, 1, 1)
        plt.imshow(x1[display_start:display_stop, :].T, aspect='auto', origin='lower', interpolation='none')
        plt.xticks([])
        plt.title('{} X1'.format(mn))
        plt.colorbar()
        plt.subplot(5, 1, 2)
        plt.imshow(x2[display_start:display_stop, :].T, aspect='auto', origin='lower', interpolation='none')
        plt.xticks([])
        plt.title('{} X2'.format(mn))
        plt.colorbar()
        plt.subplot(5, 1, 3)
        plt.imshow(y_hat1[display_start:display_stop, :].T, aspect='auto', origin='lower', interpolation='none')
        plt.xticks([])
        plt.title('{} Y1_hat'.format(mn))
        plt.colorbar()
        plt.subplot(5, 1, 4)
        plt.imshow(y_hat2[display_start:display_stop, :].T, aspect='auto', origin='lower', interpolation='none')
        plt.title('{} Y2_hat'.format(mn))
        plt.colorbar()
        plt.xticks([])
        plt.subplot(5, 1, 5)
        plt.imshow(y_hat3[display_start:display_stop, :].T, aspect='auto', origin='lower', interpolation='none')
        plt.title('{} Y3_hat'.format(mn))
        plt.colorbar()
        plt.xticks([])
        plt.show()

    if peak_params is None:
        peak_params = dict(pre_max=2,
                           post_max=2,
                           pre_avg=2,
                           post_avg=2,
                           delta=0.05,
                           wait=5)

    sampling_interval = int(round(FRAME_INTERVAL * sr)) / float(sr)
    duration = ms_input_array.shape[0] * sampling_interval

    output = dict()
    for name, perc_voice_set, index in [('14v', PERC_VOICE_SET, cfg.full_transcription_output_index),
                                        ('3v', REDUCED_PERC_VOICE_SET, cfg.reduced_transcription_output_index)]:
        if index < 0:
            continue

        if isinstance(y_hat, list):
            y_hat_full = np.squeeze(y_hat[index], 0)
        else:
            print('Single output')
            y_hat_full = np.squeeze(y_hat, 0)

        if cfg.model_creation_fn_name == 'make_crnn_trans_nc_noskip2_weight_mdl':
            y_hat_full = y_hat_full.T

        perc_synth = PercussionSynthesizer.from_onsets(duration,
                                                       y_hat_full,
                                                       None,
                                                       sampling_interval,
                                                       has_beats=False,
                                                       pre_max=peak_params['pre_max'],
                                                       post_max=peak_params['post_max'],
                                                       pre_avg=peak_params['pre_avg'],
                                                       post_avg=peak_params['post_avg'],
                                                       delta=peak_params['delta'],
                                                       wait=peak_params['wait'],
                                                       perc_voice_set=perc_voice_set)

        jam = perc_synth.generate_jam(audio_files=[sample_audio_files[v] for v in perc_voice_set],
                                      sample_rate=output_sample_rate,
                                      mixing_coeffs=np.ones(len(perc_voice_set)) / len(perc_voice_set))

        if synthesize:
            mixed, unmixed = perc_synth.synthesize(None)
        else:
            mixed, unmixed = None, None
        output[name] = dict(mixed_audio=mixed,
                            unmixed_audio=unmixed,
                            onset_activations=y_hat_full)

    return output