#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# def input_encoding23(data_dict):
#     input_dict = data_dict['input']
#     ms_input_array = np.moveaxis(input_dict['logf_stft'], 1, 0)
#     ms_input_array = np.expand_dims(ms_input_array, axis=2)
#
#     os_input_array = np.moveaxis(input_dict['od_fun'], 1, 0)
#     os_input_array = np.expand_dims(os_input_array, axis=2)
#     return [ms_input_array, os_input_array]
#
#
# def input_encoding25(data_dict):
#     length = 799
#     input_dict = data_dict['input']
#     ms_input_array = np.moveaxis(input_dict['logf_stft'][:, :length], 1, 0)
#     ms_input_array = np.expand_dims(ms_input_array, axis=2)
#
#     os_input_array = np.moveaxis(input_dict['od_fun'][:, :length], 1, 0)
#     os_input_array = np.expand_dims(os_input_array, axis=2)
#
#     assert(ms_input_array.shape[0]==length)
#     assert(os_input_array.shape[0]==length)
#     return [ms_input_array, os_input_array]
#
#
# def target_encoding36(data_dict):
#     length = 799
#     rhythm_attrs = data_dict['output']
#     mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]
#
#     if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
#         out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
#         weights1 = np.ones(out1.shape[0], dtype='float32')
#     else:
#         out1 = np.zeros([length, 14])
#         weights1 = np.zeros(out1.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_onsets']:
#         out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
#         weights2 = np.ones(out2.shape[0], dtype='float32')
#     else:
#         out2 = np.zeros([length, 3])
#         weights2 = np.zeros(out2.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_beats']:
#         out3 = rhythm_attrs['sampled_beats'].T[:length, :]
#         weights3 = np.ones(out3.shape[0], dtype='float32')
#     else:
#         out3 = np.zeros([length, 2])
#         weights3 = np.zeros(out3.shape[0], dtype='float32')
#
#     return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]
#
#
# def target_encoding60(data_dict):
#     length = 799
#     rhythm_attrs = data_dict['output']
#     mixing_matrix = [[0, ], [1, ], [7, 8, 9], [5, 6], [3, ], [4, ], [12, ], [13, ]]
#
#     if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
#         out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
#         weights1 = np.ones(out1.shape[0], dtype='float32')
#     else:
#         out1 = np.zeros([length, 14])
#         weights1 = np.zeros(out1.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_onsets']:
#         out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
#         weights2 = np.ones(out2.shape[0], dtype='float32')
#     else:
#         out2 = np.zeros([length, 8])
#         weights2 = np.zeros(out2.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_beats']:
#         out3 = rhythm_attrs['sampled_beats'].T[:length, :]
#         weights3 = np.ones(out3.shape[0], dtype='float32')
#     else:
#         out3 = np.zeros([length, 2])
#         weights3 = np.zeros(out3.shape[0], dtype='float32')
#
#     return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]
#
#
# def target_encoding61(data_dict):
#     # 50 full
#     rhythm_attrs = data_dict['output']
#     mixing_matrix = [[0, ], [1, ], [7, 8, 9], [5, 6], [3, ], [4, ], [12, ], [13, ]]
#
#     if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
#         out1 = rhythm_attrs['sampled_onsets'].T
#         weights1 = np.ones(out1.shape[0], dtype='float32')
#     else:
#         out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
#         weights1 = np.zeros(out1.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_onsets']:
#         out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
#         weights2 = np.ones(out2.shape[0], dtype='float32')
#     else:
#         out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
#         weights2 = np.zeros(out2.shape[0], dtype='float32')
#
#     if rhythm_attrs['has_beats']:
#         out3 = rhythm_attrs['sampled_beats'].T
#         weights3 = np.ones(out3.shape[0], dtype='float32')
#     else:
#         out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
#         weights3 = np.zeros(out3.shape[0], dtype='float32')
#
#     return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]

def input_encoding22(data_dict):
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'][:, :601], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'][:, :601], 1, 0)
    os_input_array = np.clip(os_input_array / 2.25, 0, 1)
    os_input_array = np.expand_dims(os_input_array, axis=2)
    return [ms_input_array, os_input_array]


def input_encoding23(data_dict):
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)
    return [ms_input_array, os_input_array]


def input_encoding24(data_dict):
    rhythm_attrs = data_dict['output']
    sampled_onsets = rhythm_attrs['sampled_onsets'].T
    return [sampled_onsets, ]


def input_encoding25(data_dict):
    length = 799
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'][:, :length], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'][:, :length], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    assert(ms_input_array.shape[0]==length)
    assert(os_input_array.shape[0]==length)
    return [ms_input_array, os_input_array]


def input_encoding26(data_dict):
    length = 799
    input_dict = data_dict['input']
    pcen_input_array = np.moveaxis(input_dict['pcen_stft'][:, :length], 1, 0)
    pcen_input_array = np.expand_dims(pcen_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'][:, :length], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    assert(pcen_input_array.shape[0]==length)
    assert(os_input_array.shape[0]==length)
    return [pcen_input_array, os_input_array]


def input_encoding27(data_dict):
    input_dict = data_dict['input']
    pcen_input_array = np.moveaxis(input_dict['pcen_stft'], 1, 0)
    pcen_input_array = np.expand_dims(pcen_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)
    return [pcen_input_array, os_input_array]


def input_encoding28(data_dict):
    length = 799
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'][:, :length], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'][:, :length], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    if data_dict['output']['has_base_pattern']:
        decoder_input_array, _, _ = bp_beats_to_pattern_state(data_dict['output']['bp_sampled_beats'],
                                                              data_dict['output']['bp_sampled_onsets'], decoder_input=True)
        decoder_input_array = np.moveaxis(decoder_input_array, 1, 0)
    else:
        decoder_input_array = np.zeros([799, 19], dtype='float32')

    assert(ms_input_array.shape[0] == length)
    assert(os_input_array.shape[0] == length)
    return [ms_input_array, os_input_array, decoder_input_array]


def input_encoding29(data_dict):
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    if data_dict['output']['has_base_pattern']:
        decoder_input_array, _, _ = bp_beats_to_pattern_state(data_dict['output']['bp_sampled_beats'],
                                                              data_dict['output']['bp_sampled_onsets'], decoder_input=True).T
        decoder_input_array = np.moveaxis(decoder_input_array, 1, 0)
    else:
        decoder_input_array = np.zeros([799, 19], dtype='float32')

    return [ms_input_array, os_input_array, decoder_input_array]


def input_encoding30(data_dict):
    length = 799
    input_dict = data_dict['input']
    ms_input_array = np.moveaxis(input_dict['logf_stft'][:, :length], 1, 0)
    ms_input_array = np.expand_dims(ms_input_array, axis=2)

    os_input_array = np.moveaxis(input_dict['od_fun'][:, :length], 1, 0)
    os_input_array = np.expand_dims(os_input_array, axis=2)

    if data_dict['output']['has_base_pattern']:
        beats = data_dict['output']['bp_sampled_beats']
        onsets = data_dict['output']['bp_sampled_onsets']
        beats, onsets = convert_to_beat_intervals(beats, onsets)
        decoder_input_array, _, _ = bp_beats_to_pattern_state(beats, onsets, decoder_input=True)
        decoder_input_array = np.moveaxis(decoder_input_array, 1, 0)
    else:
        decoder_input_array = np.zeros([799, 19], dtype='float32')

    assert(ms_input_array.shape[0] == length)
    assert(os_input_array.shape[0] == length)
    return [ms_input_array, os_input_array, decoder_input_array]


def input_encoding32(data_dict):
    length = 799
    gt_beat_trans = np.vstack([data_dict['output']['sampled_onsets'][:, :length], data_dict['output']['sampled_beats'][:, :length]])
    gt_beat_trans = np.moveaxis(gt_beat_trans, 1, 0)
    gt_beat_trans = np.expand_dims(gt_beat_trans, axis=2)

    if data_dict['output']['has_base_pattern']:
        beats = data_dict['output']['bp_sampled_beats']
        onsets = data_dict['output']['bp_sampled_onsets']
        beats, onsets = convert_to_beat_intervals(beats, onsets)
        decoder_input_array, _, _ = bp_beats_to_pattern_state(beats, onsets, decoder_input=True)
        decoder_input_array = np.moveaxis(decoder_input_array, 1, 0)
    else:
        decoder_input_array = np.zeros([799, 19], dtype='float32')

    assert(gt_beat_trans.shape[0] == length)
    return [gt_beat_trans, decoder_input_array]


def input_encoding33(data_dict):
    mixing_matrix = [[0], [1], [3], [4], [5], [6], [7], [8], [9]]

    length = 199
    gt_beat_trans = np.vstack([data_dict['output']['sampled_onsets'][:, :length], data_dict['output']['sampled_beats'][:, :length]])
    gt_beat_trans = np.moveaxis(gt_beat_trans, 1, 0)
    gt_beat_trans = np.expand_dims(gt_beat_trans, axis=2)

    if data_dict['output']['has_base_pattern']:
        beats = data_dict['output']['bp_sampled_beats']
        onsets = reduce_voices(data_dict['output']['bp_sampled_onsets'].T, mixing_matrix).T
        beats, onsets = convert_to_beat_intervals(beats, onsets)
        decoder_input_array, _, _ = bp_beats_to_pattern_state(beats, onsets, decoder_input=True)
        decoder_input_array = np.moveaxis(decoder_input_array, 1, 0)
        decoder_input_array = decoder_input_array[:length,:]
    else:
        decoder_input_array = np.zeros([length, 14], dtype='float32')

    assert(gt_beat_trans.shape[0] == length)
    return [gt_beat_trans, decoder_input_array]


def target_encoding25(data_dict):
    rhythm_attrs = data_dict['output']
    out1 = np.squeeze(input_encoding22(data_dict)[1], axis=(2,))
    out2 = rhythm_attrs['sampled_onsets'].T[:601, :]
    out3 = rhythm_attrs['bp_sampled_onsets'].T[:601, :]
    weights1 = np.ones(out1.shape[0])
    weights2 = weights1
    weights3 = np.zeros(out2.shape[0])
    weights3[:rhythm_attrs['cycle_length_samples']] = 1.0
    return [[out1, out2, out3,], [weights1, weights2, weights3,]]


def target_encoding26(data_dict):
    rhythm_attrs = data_dict['output']
    out1 = np.squeeze(input_encoding22(data_dict)[1], axis=(2,))
    weights1 = np.ones(out1.shape[0])
    return [[out1, ], [weights1, ]]


def target_encoding27(data_dict):
    rhythm_attrs = data_dict['output']
    out1 = rhythm_attrs['sampled_onsets'].T
    out2 = rhythm_attrs['sampled_beats'].T
    weights1 = np.ones(out1.shape[0])
    weights2 = np.ones(out2.shape[0])
    return [[out1, out2], [weights1, weights2]]


def target_encoding28(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], [-2, ], [-1], ]

    out1 = np.squeeze(input_encoding22(data_dict)[1], axis=(2,))
    out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:601, :], mixing_matrix)
    out3 = reduce_voices(rhythm_attrs['bp_sampled_onsets'].T[:601, :], mixing_matrix)
    weights1 = np.ones(out1.shape[0])
    weights2 = weights1
    weights3 = np.zeros(out2.shape[0])
    weights3[:rhythm_attrs['cycle_length_samples']] = 1.0
    return [[out1, out2, out3,], [weights1, weights2, weights3,]]


def target_encoding29(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], [-2, ], [-1], ]

    out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
    weights2 = np.ones(out2.shape[0])
    return [[None, out2, None,], [None, weights2, None,]]


def target_encoding30(data_dict):
    rhythm_attrs = data_dict['output']

    out3= rhythm_attrs['bp_sampled_onsets'].T
    weights3 = np.ones(out3.shape[0])
    return [[None, None, out3], [None, None, weights3,]]


def target_encoding31(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
    out2 = rhythm_attrs['sampled_onsets'].T[:799, :]
    out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:799, :], mixing_matrix)
    out4 = rhythm_attrs['sampled_beats'].T[:799, :]
    weights1 = np.ones(out1.shape[0])
    weights2 = weights1
    weights3 = weights1
    weights4 = weights1
    return [[out1, out2, out3, out4], [weights1, weights2, weights3, weights4]]


def target_encoding32(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    out1 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:799, :], mixing_matrix)
    out2 = rhythm_attrs['sampled_beats'].T[:799, :]
    weights1 = np.ones(out1.shape[0])
    weights2 = weights1
    return [[out1, out2, ], [weights1, weights2, ]]


def target_encoding33(data_dict):
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    rhythm_attrs = data_dict['output']
    out1 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
    out2 = rhythm_attrs['sampled_beats'].T
    weights1 = np.ones(out1.shape[0])
    weights2 = np.ones(out2.shape[0])
    return [[out1, out2], [weights1, weights2]]


def target_encoding34(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
    out2 = rhythm_attrs['sampled_onsets'].T[:799, :]
    out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:799, :], mixing_matrix)
    out4 = rhythm_attrs['sampled_beats'].T[:799, :]

    if rhythm_attrs['has_denoised']:
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        weights4 = np.ones(out4.shape[0], dtype='float32')
    else:
        weights4 = np.zeros(out4.shape[0], dtype='float32')

    return [[out1, out2, out3, out4,], [weights1, weights2, weights3, weights4,]]


def target_encoding35(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
    out2 = rhythm_attrs['sampled_onsets'].T[:799, :]
    out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:799, :], mixing_matrix)
    out4 = rhythm_attrs['sampled_beats'].T[:799, :]
    out5 = rhythm_attrs['bp_sampled_onsets'].T[:799, :]
    out6 = rhythm_attrs['bp_sampled_beats'].T[:799, :]

    weights1 = np.ones(out1.shape[0])
    weights2 = weights1
    weights3 = weights1
    weights4 = weights1
    weights5 = weights1
    weights6 = weights1
    return [[out1, out2, out3, out4, out5, out6], [weights1, weights2, weights3, weights4, weights5, weights6]]


def target_encoding36(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]


def target_encoding37(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets']:
        out1 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 3])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out2 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 2])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    return [[out1, out2, ], [weights1, weights2, ]]


def target_encoding38(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    return [[out1, out2, ], [weights1, weights2, ]]


def target_encoding39(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets']:
        out1 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 3])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding40(data_dict):
    # 36 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]


def target_encoding41(data_dict):
    # 37 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    return [[out2, out3, ], [weights2, weights3, ]]


def target_encoding42(data_dict):
    # 38 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    return [[out1, out2, ], [weights1, weights2, ]]


def target_encoding43(data_dict):
    # 39 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    return [[out2, ], [weights2, ]]


def target_encoding44(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding45(data_dict):
    # 44 full
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding46(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_beats']:
        out1 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1= np.zeros([length, 2])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding47(data_dict):
    ## 46 full
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_beats']:
        out1 = rhythm_attrs['sampled_beats'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding48(data_dict):
    length = 799
    weights = [1.0, 1.3221026092, 15.3404558913, 5.25943991047, 2.17277924354, 5.79943287406, 1.40574770645,
               7.6096545423, 4.45459325191, 6.89835544313, 5.91528291973, 43.4788247265, 50.0, 12.4145447049]
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'][:, :length]
        weights1 = np.array(weights)
    else:
        out1 = np.zeros([14, length])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding49(data_dict):
    # 48 full
    weights = [1.0, 1.3221026092, 15.3404558913, 5.25943991047, 2.17277924354, 5.79943287406, 1.40574770645,
               7.6096545423, 4.45459325191, 6.89835544313, 5.91528291973, 43.4788247265, 50.0, 12.4145447049]
    rhythm_attrs = data_dict['output']

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets']
        weights1 = np.array(weights)
    else:
        out1 = np.zeros([14, rhythm_attrs['sampled_onsets'].shape[1]])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    return [[out1, ], [weights1, ]]


def target_encoding50(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_denoised']:
        out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out2 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights2 = np.ones(out1.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 14])
        weights2 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights3 = np.ones(out2.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 3])
        weights3 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out4 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights4 = np.ones(out3.shape[0], dtype='float32')
    else:
        out4 = np.zeros([length, 2])
        weights4 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, out4,], [weights1, weights2, weights3, weights4]]


def target_encoding51(data_dict):
    # 50 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_denoised']:
        out1 = np.squeeze(input_encoding25(data_dict)[1], axis=(2,))
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 64])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out2 = rhythm_attrs['sampled_onsets'].T
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out4 = rhythm_attrs['sampled_beats'].T
        weights4 = np.ones(out4.shape[0], dtype='float32')
    else:
        out4 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights4 = np.zeros(out4.shape[0], dtype='float32')

    return [[out1, out2, out3, out4,], [weights1, weights2, weights3, weights4]]


def target_encoding52(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_denoised']:
        out1 = np.squeeze(input_encoding26(data_dict)[1], axis=(2,))
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.squeeze(input_encoding26(data_dict)[1], axis=(2,))
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out2 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights2 = np.ones(out1.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 14])
        weights2 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights3 = np.ones(out2.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 3])
        weights3 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out4 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights4 = np.ones(out3.shape[0], dtype='float32')
    else:
        out4 = np.zeros([length, 2])
        weights4 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, out4,], [weights1, weights2, weights3, weights4]]


def target_encoding53(data_dict):
    # 52 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_denoised']:
        out1 = np.squeeze(input_encoding26(data_dict)[1], axis=(2,))
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 64])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out2 = rhythm_attrs['sampled_onsets'].T
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out3 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out4 = rhythm_attrs['sampled_beats'].T
        weights4 = np.ones(out4.shape[0], dtype='float32')
    else:
        out4 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights4 = np.zeros(out4.shape[0], dtype='float32')

    return [[out1, out2, out3, out4,], [weights1, weights2, weights3, weights4]]


def target_encoding54(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_base_pattern']:
        out4, weights4, weights5 = bp_beats_to_pattern_state(data_dict['output']['bp_sampled_beats'],
                                                             data_dict['output']['bp_sampled_onsets'],
                                                             decoder_input=False)
        out4 = out4.T[:length]
        out5 = rhythm_attrs['bp_sampled_onsets'].T[:length]
    else:
        out4 = np.zeros([length, 5])
        weights4 = np.zeros(out4.shape[0], dtype='float32')
        out5 = np.zeros([length, 14])
        weights5 = np.zeros(out5.shape[0], dtype='float32')

    return [[out1, out2, out3, out4, out5], [weights1, weights2, weights3, weights4, weights5]]


def target_encoding55(data_dict):
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_base_pattern']:
        out4, weights4, weights5 = bp_beats_to_pattern_state(data_dict['output']['bp_sampled_beats'],
                                                             data_dict['output']['bp_sampled_onsets'],
                                                             decoder_input=False)
        out4 = out4.T[:length]
        out5 = rhythm_attrs['bp_sampled_onsets'].T[:length]
    else:
        out4 = np.zeros([length, 5])
        weights4 = np.zeros(out4.shape[0], dtype='float32')
        out5 = np.zeros([length, 14])
        weights5 = np.zeros(out5.shape[0], dtype='float32')

    return [[out1, out2, out3, out4, out5], [weights1, weights2, weights3, weights4, weights5]]


def target_encoding56(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, 2, 11], [5, 6], ]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    if rhythm_attrs['has_base_pattern']:
        beats = data_dict['output']['bp_sampled_beats']
        onsets = data_dict['output']['bp_sampled_onsets']
        beats, onsets = convert_to_beat_intervals(beats, onsets)

        out4, weights4, weights5 = bp_beats_to_pattern_state(beats, onsets, decoder_input=False)
        out4 = out4.T[:length]
        out5 = onsets.T[:length]
    else:
        out4 = np.zeros([length, 5])
        weights4 = np.zeros(out4.shape[0], dtype='float32')
        out5 = np.zeros([length, 14])
        weights5 = np.zeros(out5.shape[0], dtype='float32')

    return [[out1, out2, out3, out4, out5], [weights1, weights2, weights3, weights4, weights5]]


def target_encoding58(data_dict):
    length = 799

    beats = data_dict['output']['bp_sampled_beats']
    onsets = data_dict['output']['bp_sampled_onsets']
    beats, onsets = convert_to_beat_intervals(beats, onsets)

    out1, weights1, weights2 = bp_beats_to_pattern_state(beats, onsets, decoder_input=False)
    out1 = out1.T[:length]
    out2 = onsets.T[:length]

    return [[out1, out2], [weights1, weights2]]


def target_encoding59(data_dict):
    length = 199
    mixing_matrix = [[0], [1], [3], [4], [5], [6], [7], [8], [9]]

    beats = data_dict['output']['bp_sampled_beats']
    onsets = reduce_voices(data_dict['output']['bp_sampled_onsets'].T, mixing_matrix).T
    beats, onsets = convert_to_beat_intervals(beats, onsets)

    out1, weights1, weights2 = bp_beats_to_pattern_state(beats, onsets, decoder_input=False)
    out1 = out1.T[:length]
    out2 = onsets.T[:length]
    weights1 = weights1[:length]
    weights2 = weights2[:length]

    return [[out1, out2], [weights1, weights2]]


def target_encoding60(data_dict):
    length = 799
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, ], [7, 8, 9], [5, 6], [3, ], [4, ], [12, ], [13, ]]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T[:length, :]
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([length, 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T[:length, :], mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([length, 8])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T[:length, :]
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([length, 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]


def target_encoding61(data_dict):
    # 50 full
    rhythm_attrs = data_dict['output']
    mixing_matrix = [[0, ], [1, ], [7, 8, 9], [5, 6], [3, ], [4, ], [12, ], [13, ]]

    if rhythm_attrs['has_onsets'] and rhythm_attrs['has_large_vocab']:
        out1 = rhythm_attrs['sampled_onsets'].T
        weights1 = np.ones(out1.shape[0], dtype='float32')
    else:
        out1 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 14])
        weights1 = np.zeros(out1.shape[0], dtype='float32')

    if rhythm_attrs['has_onsets']:
        out2 = reduce_voices(rhythm_attrs['sampled_onsets'].T, mixing_matrix)
        weights2 = np.ones(out2.shape[0], dtype='float32')
    else:
        out2 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 3])
        weights2 = np.zeros(out2.shape[0], dtype='float32')

    if rhythm_attrs['has_beats']:
        out3 = rhythm_attrs['sampled_beats'].T
        weights3 = np.ones(out3.shape[0], dtype='float32')
    else:
        out3 = np.zeros([rhythm_attrs['sampled_onsets'].T.shape[0], 2])
        weights3 = np.zeros(out3.shape[0], dtype='float32')

    return [[out1, out2, out3, ], [weights1, weights2, weights3, ]]


def target_encoding62(data_dict):
    # transcription, not pattern
    length = 199
    mixing_matrix = [[0], [1], [3], [4], [5], [6], [7], [8], [9]]

    beats = data_dict['output']['bp_sampled_beats']
    onsets = reduce_voices(data_dict['output']['bp_sampled_onsets'].T, mixing_matrix).T
    beats, onsets = convert_to_beat_intervals(beats, onsets)

    out1, weights1, weights2 = bp_beats_to_pattern_state(beats, onsets, decoder_input=False)
    out1 = out1.T[:length]
    out2 = onsets.T[:length]
    weights1 = weights1[:length]
    weights2 = weights2[:length]

    return [[out1, out2], [weights1, weights2]]