#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from keras.layers import Input, BatchNormalization, Concatenate, Conv2D, Activation, Dropout, Reshape, Lambda, \
    Bidirectional, LSTM, TimeDistributed, Dense, GRU, SimpleRNN, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import keras

import model_configurations
import encodings
models = sys.modules[__name__]


def init_model_config(model_configuration_id):
    """
    Init model configuration.

    Parameters
    ----------
    model_configuration_id

    Returns
    -------
    cfg
        Model configuration
    """
    cfg = getattr(model_configurations, 'ModelConfiguration{}'.format(model_configuration_id))()
    cfg.input_encoding_fn = getattr(encodings, cfg.input_encoding_fn_name)
    cfg.target_encoding_fn = getattr(encodings, cfg.target_encoding_fn_name)
    cfg.model_creation_fn = getattr(models, cfg.model_creation_fn_name)
    cfg.model_args.update(dict(optimizer=eval(cfg.optimizer_fn_call)))
    try:
        cfg.model_args.update(dict(loss=[eval(l) for l in cfg.model_args['loss']]))
    except KeyError as e:
        print('No loss string in config')

    return cfg


def convert_full_h5_model_to_def_and_weights(model_configuration_id,
                                             output_model_name,
                                             model_h5_path,
                                             output_dir):
    cfg = init_model_config(model_configuration_id)
    cfg.model_args['input_shape'] = (None, cfg.model_args['input_shape'][1], cfg.model_args['input_shape'][2])
    mdl = cfg.model_creation_fn(**cfg.model_args)
    trained_mdl = load_model(model_h5_path)

    mdl.set_weights(trained_mdl.get_weights())

    model_weights_path = os.path.join(output_dir, '{}_weights.h5'.format(output_model_name))
    model_def_path = os.path.join(output_dir, '{}_def.json'.format(output_model_name))

    mdl.save_weights(model_weights_path)
    with open(model_def_path, 'w') as f:
        f.write(mdl.to_json())


def _add_context(x, context_frames, pad_frames):
    x = K.temporal_padding(x, (pad_frames, pad_frames))
    to_concat = [x[:, offset:-(context_frames - offset - 1), :] for offset in range(context_frames - 1)]
    to_concat.append(x[:, (context_frames - 1):, :])
    x = K.concatenate(to_concat, axis=2)
    return x


# def make_crnn_trans_mdl(input_shape,
#                         full_output_pattern_voices=14,
#                         reduced_output_pattern_voices=3,
#                         use_mag_spec_input=True,
#                         use_onset_signal_input=True,
#                         has_denoised_output=True,
#                         has_transcript_full_output=True,
#                         has_transcript_reduced_output=True,
#                         has_beat_output=True,
#                         rnn_units=(64, 64,),
#                         rnn_layers=(1, 1,),
#                         rnn_types=('blstm', 'blstm'),
#                         loss=('mse', 'mse', 'mse', 'mse',),
#                         metrics=(('denoised', 'mae',),
#                                  ('transcribed_full', 'mae'),
#                                  ('transcribed_reduce', 'mae'),
#                                  ('beat', 'mae'),),
#                         sample_weight_mode='temporal',
#                         optimizer=Adam(),
#                         loss_weights=(1, 1, 1, 1,),
#                         context_frames=1,
#                         conv_layers=(1,),
#                         conv_shapes=((7, 7),),
#                         conv_filters=(32,),
#                         conv_max_pooling_size=(None,),
#                         conv_dropout=(0,)):
#     # no cycle output
#     mag_spec_in = Input(input_shape)
#     onset_signals_in = Input(input_shape)
#     inputs = [mag_spec_in, onset_signals_in]
#
#     mag_spec = BatchNormalization()(mag_spec_in)
#     onset_signals = BatchNormalization()(onset_signals_in)
#
#     concat = []
#     if use_mag_spec_input:
#         concat.append(mag_spec)
#     if use_onset_signal_input:
#         concat.append(onset_signals)
#     if len(concat) > 1:
#         x = Concatenate(axis=3)(concat)
#     else:
#         x = concat[0]
#
#     # denoising block
#     if rnn_layers[0] > 0:
#         x = Reshape(target_shape=(-1, 128))(x)
#         rnn1 = x
#         for i in range(rnn_layers[0]):
#             if rnn_types[0] == 'lstm':
#                 rnn1 = LSTM(units=rnn_units[0], return_sequences=True)(rnn1)
#             elif rnn_types[0] == 'gru':
#                 rnn1 = GRU(units=rnn_units[0], return_sequences=True)(rnn1)
#             elif rnn_types[0] == 'rnn':
#                 rnn1 = SimpleRNN(units=rnn_units[0], return_sequences=True)(rnn1)
#             elif rnn_types[0] == 'blstm':
#                 rnn1 = Bidirectional(LSTM(units=rnn_units[0], return_sequences=True))(rnn1)
#             elif rnn_types[0] == 'bgru':
#                 rnn1 = Bidirectional(GRU(units=rnn_units[0], return_sequences=True))(rnn1)
#             else:
#                 raise Exception('Unknown RNN layer')
#         denoised_output = TimeDistributed(Dense(64, activation='sigmoid'), name='denoised_output')(rnn1)
#         denoised_output_reshape = Reshape(target_shape=(-1, 64, 1))(denoised_output)
#
#         x = denoised_output_reshape
#
#     conv1 = x
#     for i, num_layers in enumerate(conv_layers):
#         for layer_idx in range(conv_layers[i]):
#             conv1 = Conv2D(conv_filters[i], conv_shapes[i], padding='same')(conv1)
#         conv1 = BatchNormalization()(conv1)
#         conv1 = Activation('relu')(conv1)  # not in vogl model?
#         if conv_max_pooling_size[i] is not None:
#             conv1 = MaxPooling2D(pool_size=(conv_max_pooling_size[i]), strides=(1, 1), padding='same')(conv1)
#         if conv_dropout[i] > 0:
#             conv1 = Dropout(rate=conv_dropout[i])(conv1)
#
#     conv2 = Conv2D(64, (1, int(conv1.shape[2])), padding='valid')(conv1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Activation('relu')(conv2)
#     conv2_reshape = Reshape(target_shape=(-1, 64, 1))(conv2)
#
#     x = conv2_reshape
#     # transcription block
#     x = Reshape(target_shape=(-1, 64))(x)
#
#     # context windows
#     pad_frames = (context_frames - 1) / 2
#
#     if context_frames > 1:
#         rnn2 = Lambda(_add_context, arguments={'context_frames': context_frames,
#                                                 'pad_frames': pad_frames})(x)
#     else:
#         rnn2 = x
#
#     for i in range(rnn_layers[1]):
#         if rnn_types[1] == 'lstm':
#             rnn2 = LSTM(units=rnn_units[1], return_sequences=True)(rnn2)
#         elif rnn_types[1] == 'gru':
#             rnn2 = GRU(units=rnn_units[1], return_sequences=True)(rnn2)
#         elif rnn_types[1] == 'rnn':
#             rnn2 = SimpleRNN(units=rnn_units[1], return_sequences=True)(rnn2)
#         elif rnn_types[1] == 'blstm':
#             rnn2 = Bidirectional(LSTM(units=rnn_units[1], return_sequences=True))(rnn2)
#         elif rnn_types[1] == 'bgru':
#             rnn2 = Bidirectional(GRU(units=rnn_units[1], return_sequences=True))(rnn2)
#         else:
#             raise Exception('Unknown RNN layer')
#
#     outputs = []
#
#     if has_denoised_output:
#         outputs.append(denoised_output)
#
#     if has_transcript_full_output:
#         transcript_full_output = TimeDistributed(Dense(units=full_output_pattern_voices, activation='sigmoid'),
#                                                  name='transcript_full_output')(rnn2)
#         outputs.append(transcript_full_output)
#
#     if has_transcript_reduced_output:
#         transcript_reduced_output = TimeDistributed(Dense(units=reduced_output_pattern_voices, activation='sigmoid'), name='transcript_reduced_output')(rnn2)
#         outputs.append(transcript_reduced_output)
#
#     if has_beat_output:
#         beat_output = TimeDistributed(Dense(units=2, activation='sigmoid'), name='beat_output')(rnn2)
#         outputs.append(beat_output)
#
#     model = Model(inputs=inputs, outputs=outputs)
#
#     model.summary()
#
#     model.compile(loss=list(loss),
#                   optimizer=optimizer,
#                   metrics=dict(metrics),
#                   loss_weights=list(loss_weights),
#                   sample_weight_mode=sample_weight_mode)
#
#     return model


def make_crnn_trans_nc_mdl(input_shape,
                           full_output_pattern_voices=14,
                           reduced_output_pattern_voices=3,
                           use_mag_spec_input=True,
                           use_onset_signal_input=True,
                           has_denoised_output=True,
                           has_transcript_full_output=True,
                           has_transcript_reduced_output=True,
                           has_beat_output=True,
                           rnn_units=(64, 64, ),
                           rnn_layers=(1, 1, ),
                           rnn_types=('blstm', 'blstm'),
                           loss=('mse', 'mse', 'mse', 'mse', ),
                           metrics=(('denoised', 'mae',),
                                    ('transcribed_full', 'mae'),
                                    ('transcribed_reduce', 'mae'),
                                    ('beat', 'mae'), ),
                           sample_weight_mode='temporal',
                           optimizer=Adam(),
                           loss_weights=(1, 1, 1, 1, ),
                           context_frames=1,
                           conv_layers=(1,),
                           conv_shapes=((7, 7),),
                           conv_filters=(32,),
                           conv_max_pooling_size=(None,),
                           conv_dropout=(0,)):
    # no cycle output
    mag_spec_in = Input(input_shape)
    onset_signals_in = Input(input_shape)
    inputs = [mag_spec_in, onset_signals_in]

    mag_spec = BatchNormalization()(mag_spec_in)
    onset_signals = BatchNormalization()(onset_signals_in)

    concat = []
    if use_mag_spec_input:
        concat.append(mag_spec)
    if use_onset_signal_input:
        concat.append(onset_signals)
    if len(concat) > 1:
        x = Concatenate(axis=3)(concat)
    else:
        x = concat[0]

    # denoising block
    if rnn_layers[0] > 0:
        x = Reshape(target_shape=(-1, 128))(x)
        rnn1 = x
        for i in range(rnn_layers[0]):
            if rnn_types[0] == 'lstm':
                rnn1 = LSTM(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'gru':
                rnn1 = GRU(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'rnn':
                rnn1 = SimpleRNN(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'blstm':
                rnn1 = Bidirectional(LSTM(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'bgru':
                rnn1 = Bidirectional(GRU(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'brnn':
                rnn1 = Bidirectional(SimpleRNN(units=rnn_units[0], return_sequences=True))(rnn1)
            else:
                raise Exception('Unknown RNN layer')
        denoised_output = TimeDistributed(Dense(64, activation='sigmoid'), name='denoised_output')(rnn1)
        denoised_output_reshape = Reshape(target_shape=(-1, 64, 1))(denoised_output)

        concat = []
        if use_mag_spec_input:
            concat.append(mag_spec)
        if use_onset_signal_input:
            concat.append(onset_signals)
        concat.append(denoised_output_reshape)
        if len(concat) > 1:
            x = Concatenate()(concat)
        else:
            x = concat[0]

    conv1 = x
    for i, num_layers in enumerate(conv_layers):
        for layer_idx in range(conv_layers[i]):
            conv1 = Conv2D(conv_filters[i], conv_shapes[i], padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)  # not in vogl model?
        if conv_max_pooling_size[i] is not None:
            conv1 = MaxPooling2D(pool_size=(conv_max_pooling_size[i]), strides=(1, 1), padding='same')(conv1)
        if conv_dropout[i] > 0:
            conv1 = Dropout(rate=conv_dropout[i])(conv1)

    conv2 = Conv2D(64, (1, int(conv1.shape[2])), padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_reshape = Reshape(target_shape=(-1, 64, 1))(conv2)

    if rnn_layers[0] > 0:
        concat = []
        if use_mag_spec_input:
            concat.append(mag_spec)
        if use_onset_signal_input:
            concat.append(onset_signals)
        concat.extend([denoised_output_reshape, conv2_reshape])
        if len(concat) > 1:
            x = Concatenate(axis=2)(concat)
        else:
            x = concat[0]
        x_size = len(concat)
    else:
        concat = []
        if use_mag_spec_input:
            concat.append(mag_spec)
        if use_onset_signal_input:
            concat.append(onset_signals)
        concat.append(conv2_reshape)
        if len(concat) > 1:
            x = Concatenate(axis=2)(concat)
        else:
            x = concat[0]
        x_size = len(concat)

    # transcription block
    x = Reshape(target_shape=(-1, 64*x_size))(x)

    # context windows
    pad_frames = (context_frames - 1) / 2

    if context_frames > 1:
        rnn2 = Lambda(_add_context, arguments={'context_frames': context_frames,
                                                'pad_frames': pad_frames})(x)
    else:
        rnn2 = x

    for i in range(rnn_layers[1]):
        if rnn_types[1] == 'lstm':
            rnn2 = LSTM(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'gru':
            rnn2 = GRU(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'rnn':
            rnn2 = SimpleRNN(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'blstm':
            rnn2 = Bidirectional(LSTM(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'bgru':
            rnn2 = Bidirectional(GRU(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'brnn':
            rnn2 = Bidirectional(SimpleRNN(units=rnn_units[1], return_sequences=True))(rnn2)
        else:
            raise Exception('Unknown RNN layer')

    outputs = []

    if has_denoised_output:
        outputs.append(denoised_output)

    if has_transcript_full_output:
        transcript_full_output = TimeDistributed(Dense(units=full_output_pattern_voices, activation='sigmoid'),
                                                 name='transcript_full_output')(rnn2)
        outputs.append(transcript_full_output)

    if has_transcript_reduced_output:
        transcript_reduced_output = TimeDistributed(Dense(units=reduced_output_pattern_voices, activation='sigmoid'), name='transcript_reduced_output')(rnn2)
        outputs.append(transcript_reduced_output)

    if has_beat_output:
        beat_output = TimeDistributed(Dense(units=2, activation='sigmoid'), name='beat_output')(rnn2)
        outputs.append(beat_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=list(loss),
                  optimizer=optimizer,
                  metrics=dict(metrics),
                  loss_weights=list(loss_weights),
                  sample_weight_mode=sample_weight_mode)

    return model


def make_crnn_trans_nc_noskip_mdl(input_shape,
                                  full_output_pattern_voices=14,
                                  reduced_output_pattern_voices=3,
                                  use_mag_spec_input=True,
                                  use_onset_signal_input=True,
                                  has_denoised_output=True,
                                  has_transcript_full_output=True,
                                  has_transcript_reduced_output=True,
                                  has_beat_output=True,
                                  rnn_units=(64, 64,),
                                  rnn_layers=(1, 1,),
                                  rnn_types=('blstm', 'blstm'),
                                  loss=('mse', 'mse', 'mse', 'mse',),
                                  metrics=(('denoised', 'mae',),
                                           ('transcribed_full', 'mae'),
                                           ('transcribed_reduce', 'mae'),
                                           ('beat', 'mae'),),
                                  sample_weight_mode='temporal',
                                  optimizer=Adam(),
                                  loss_weights=(1, 1, 1, 1,),
                                  context_frames=1,
                                  conv_layers=(1,),
                                  conv_shapes=((7, 7),),
                                  conv_filters=(32,),
                                  conv_max_pooling_size=(None,),
                                  conv_dropout=(0,)):
    # no cycle output
    mag_spec_in = Input(input_shape)
    onset_signals_in = Input(input_shape)
    inputs = [mag_spec_in, onset_signals_in]

    mag_spec = BatchNormalization()(mag_spec_in)
    onset_signals = BatchNormalization()(onset_signals_in)

    concat = []
    if use_mag_spec_input:
        concat.append(mag_spec)
    if use_onset_signal_input:
        concat.append(onset_signals)
    if len(concat) > 1:
        x = Concatenate(axis=2)(concat)
    else:
        x = concat[0]

    # denoising block
    if rnn_layers[0] > 0:
        x = Reshape(target_shape=(-1, 128))(x)
        rnn1 = x
        for i in range(rnn_layers[0]):
            if rnn_types[0] == 'lstm':
                rnn1 = LSTM(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'gru':
                rnn1 = GRU(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'rnn':
                rnn1 = SimpleRNN(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'blstm':
                rnn1 = Bidirectional(LSTM(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'bgru':
                rnn1 = Bidirectional(GRU(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'brnn':
                rnn1 = Bidirectional(SimpleRNN(units=rnn_units[0], return_sequences=True))(rnn1)
            else:
                raise Exception('Unknown RNN layer')
        denoised_output = TimeDistributed(Dense(64, activation='sigmoid'), name='denoised_output')(rnn1)
        denoised_output_reshape = Reshape(target_shape=(-1, 64, 1))(denoised_output)

        x = denoised_output_reshape

    conv1 = x
    for i, num_layers in enumerate(conv_layers):
        for layer_idx in range(conv_layers[i]):
            conv1 = Conv2D(conv_filters[i], conv_shapes[i], padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)  # not in vogl model?
        if conv_max_pooling_size[i] is not None:
            conv1 = MaxPooling2D(pool_size=(conv_max_pooling_size[i]), strides=(1, 1), padding='same')(conv1)
        if conv_dropout[i] > 0:
            conv1 = Dropout(rate=conv_dropout[i])(conv1)

    conv2 = Conv2D(64, (1, int(conv1.shape[2])), padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_reshape = Reshape(target_shape=(-1, 64, 1))(conv2)

    x = conv2_reshape
    # transcription block
    x = Reshape(target_shape=(-1, 64))(x)

    # context windows
    pad_frames = (context_frames - 1) / 2

    if context_frames > 1:
        rnn2 = Lambda(_add_context, arguments={'context_frames': context_frames,
                                                'pad_frames': pad_frames})(x)
    else:
        rnn2 = x

    for i in range(rnn_layers[1]):
        if rnn_types[1] == 'lstm':
            rnn2 = LSTM(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'gru':
            rnn2 = GRU(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'rnn':
            rnn2 = SimpleRNN(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'blstm':
            rnn2 = Bidirectional(LSTM(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'bgru':
            rnn2 = Bidirectional(GRU(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'brnn':
            rnn2 = Bidirectional(SimpleRNN(units=rnn_units[1], return_sequences=True))(rnn2)
        else:
            raise Exception('Unknown RNN layer')

    outputs = []

    if has_denoised_output:
        outputs.append(denoised_output)

    if has_transcript_full_output:
        transcript_full_output = TimeDistributed(Dense(units=full_output_pattern_voices, activation='sigmoid'),
                                                 name='transcript_full_output')(rnn2)
        outputs.append(transcript_full_output)

    if has_transcript_reduced_output:
        transcript_reduced_output = TimeDistributed(Dense(units=reduced_output_pattern_voices, activation='sigmoid'), name='transcript_reduced_output')(rnn2)
        outputs.append(transcript_reduced_output)

    if has_beat_output:
        beat_output = TimeDistributed(Dense(units=2, activation='sigmoid'), name='beat_output')(rnn2)
        outputs.append(beat_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=list(loss),
                  optimizer=optimizer,
                  metrics=dict(metrics),
                  loss_weights=list(loss_weights),
                  sample_weight_mode=sample_weight_mode)

    return model


def make_crnn_trans_nc_noskip2_mdl(input_shape,
                                   full_output_pattern_voices=14,
                                   reduced_output_pattern_voices=3,
                                   use_mag_spec_input=True,
                                   use_onset_signal_input=True,
                                   has_denoised_output=True,
                                   has_transcript_full_output=True,
                                   has_transcript_reduced_output=True,
                                   has_beat_output=True,
                                   rnn_units=(64, 64,),
                                   rnn_layers=(1, 1,),
                                   rnn_types=('blstm', 'blstm'),
                                   loss=('mse', 'mse', 'mse', 'mse',),
                                   metrics=(('denoised', 'mae',),
                                            ('transcribed_full', 'mae'),
                                            ('transcribed_reduce', 'mae'),
                                            ('beat', 'mae'),),
                                   sample_weight_mode='temporal',
                                   optimizer=Adam(),
                                   loss_weights=(1, 1, 1, 1,),
                                   context_frames=1,
                                   conv_layers=(1,),
                                   conv_shapes=((7, 7),),
                                   conv_filters=(32,),
                                   conv_max_pooling_size=(None,),
                                   conv_dropout=(0,)):
    # no cycle output
    mag_spec_in = Input(input_shape)
    onset_signals_in = Input(input_shape)
    inputs = [mag_spec_in, onset_signals_in]

    mag_spec = BatchNormalization()(mag_spec_in)
    onset_signals = BatchNormalization()(onset_signals_in)

    concat = []
    if use_mag_spec_input:
        concat.append(mag_spec)
    if use_onset_signal_input:
        concat.append(onset_signals)
    if len(concat) > 1:
        x = Concatenate(axis=3)(concat)
    else:
        x = concat[0]

    # denoising block
    if rnn_layers[0] > 0:
        x = Reshape(target_shape=(-1, 128))(x)
        rnn1 = x
        for i in range(rnn_layers[0]):
            if rnn_types[0] == 'lstm':
                rnn1 = LSTM(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'gru':
                rnn1 = GRU(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'rnn':
                rnn1 = SimpleRNN(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'blstm':
                rnn1 = Bidirectional(LSTM(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'bgru':
                rnn1 = Bidirectional(GRU(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'brnn':
                rnn1 = Bidirectional(SimpleRNN(units=rnn_units[0], return_sequences=True))(rnn1)
            else:
                raise Exception('Unknown RNN layer')
        denoised_output = TimeDistributed(Dense(64, activation='sigmoid'), name='denoised_output')(rnn1)
        denoised_output_reshape = Reshape(target_shape=(-1, 64, 1))(denoised_output)

        x = denoised_output_reshape

    conv1 = x
    for i, num_layers in enumerate(conv_layers):
        for layer_idx in range(conv_layers[i]):
            conv1 = Conv2D(conv_filters[i], conv_shapes[i], padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)  # not in vogl model?
        if conv_max_pooling_size[i] is not None:
            conv1 = MaxPooling2D(pool_size=(conv_max_pooling_size[i]), strides=(1, 1), padding='same')(conv1)
        if conv_dropout[i] > 0:
            conv1 = Dropout(rate=conv_dropout[i])(conv1)

    conv2 = Conv2D(64, (1, int(conv1.shape[2])), padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_reshape = Reshape(target_shape=(-1, 64, 1))(conv2)

    x = conv2_reshape
    # transcription block
    x = Reshape(target_shape=(-1, 64))(x)

    # context windows
    pad_frames = (context_frames - 1) / 2

    if context_frames > 1:
        rnn2 = Lambda(_add_context, arguments={'context_frames': context_frames,
                                                'pad_frames': pad_frames})(x)
    else:
        rnn2 = x

    for i in range(rnn_layers[1]):
        if rnn_types[1] == 'lstm':
            rnn2 = LSTM(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'gru':
            rnn2 = GRU(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'rnn':
            rnn2 = SimpleRNN(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'blstm':
            rnn2 = Bidirectional(LSTM(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'bgru':
            rnn2 = Bidirectional(GRU(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'brnn':
            rnn2 = Bidirectional(SimpleRNN(units=rnn_units[1], return_sequences=True))(rnn2)
        else:
            raise Exception('Unknown RNN layer')

    outputs = []

    if has_denoised_output:
        outputs.append(denoised_output)

    if has_transcript_full_output:
        transcript_full_output = TimeDistributed(Dense(units=full_output_pattern_voices, activation='sigmoid'),
                                                 name='transcript_full_output')(rnn2)
        outputs.append(transcript_full_output)

    if has_transcript_reduced_output:
        transcript_reduced_output = TimeDistributed(Dense(units=reduced_output_pattern_voices, activation='sigmoid'), name='transcript_reduced_output')(rnn2)
        outputs.append(transcript_reduced_output)

    if has_beat_output:
        beat_output = TimeDistributed(Dense(units=2, activation='sigmoid'), name='beat_output')(rnn2)
        outputs.append(beat_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=list(loss),
                  optimizer=optimizer,
                  metrics=dict(metrics),
                  loss_weights=list(loss_weights),
                  sample_weight_mode=sample_weight_mode)

    return model


def make_crnn_trans_nc_noskip2_weight_mdl(input_shape,
                                          full_output_pattern_voices=14,
                                          reduced_output_pattern_voices=3,
                                          use_mag_spec_input=True,
                                          use_onset_signal_input=True,
                                          has_denoised_output=True,
                                          has_transcript_full_output=True,
                                          has_transcript_reduced_output=True,
                                          has_beat_output=True,
                                          rnn_units=(64, 64,),
                                          rnn_layers=(1, 1,),
                                          rnn_types=('blstm', 'blstm'),
                                          loss=('mse', 'mse', 'mse', 'mse',),
                                          metrics=(('denoised', 'mae',),
                                                   ('transcribed_full', 'mae'),
                                                   ('transcribed_reduce', 'mae'),
                                                   ('beat', 'mae'),),
                                          sample_weight_mode='temporal',
                                          optimizer=Adam(),
                                          loss_weights=(1, 1, 1, 1,),
                                          context_frames=1,
                                          conv_layers=(1,),
                                          conv_shapes=((7, 7),),
                                          conv_filters=(32,),
                                          conv_max_pooling_size=(None,),
                                          conv_dropout=(0,)):
    # no cycle output
    mag_spec_in = Input(input_shape)
    onset_signals_in = Input(input_shape)
    inputs = [mag_spec_in, onset_signals_in]

    mag_spec = BatchNormalization()(mag_spec_in)
    onset_signals = BatchNormalization()(onset_signals_in)

    concat = []
    if use_mag_spec_input:
        concat.append(mag_spec)
    if use_onset_signal_input:
        concat.append(onset_signals)
    if len(concat) > 1:
        x = Concatenate(axis=3)(concat)
    else:
        x = concat[0]

    # denoising block
    if rnn_layers[0] > 0:
        x = Reshape(target_shape=(-1, 128))(x)
        rnn1 = x
        for i in range(rnn_layers[0]):
            if rnn_types[0] == 'lstm':
                rnn1 = LSTM(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'gru':
                rnn1 = GRU(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'rnn':
                rnn1 = SimpleRNN(units=rnn_units[0], return_sequences=True)(rnn1)
            elif rnn_types[0] == 'blstm':
                rnn1 = Bidirectional(LSTM(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'bgru':
                rnn1 = Bidirectional(GRU(units=rnn_units[0], return_sequences=True))(rnn1)
            elif rnn_types[0] == 'brnn':
                rnn1 = Bidirectional(SimpleRNN(units=rnn_units[0], return_sequences=True))(rnn1)
            else:
                raise Exception('Unknown RNN layer')
        denoised_output = TimeDistributed(Dense(64, activation='sigmoid'), name='denoised_output')(rnn1)
        denoised_output_reshape = Reshape(target_shape=(-1, 64, 1))(denoised_output)

        x = denoised_output_reshape

    conv1 = x
    for i, num_layers in enumerate(conv_layers):
        for layer_idx in range(conv_layers[i]):
            conv1 = Conv2D(conv_filters[i], conv_shapes[i], padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)  # not in vogl model?
        if conv_max_pooling_size[i] is not None:
            conv1 = MaxPooling2D(pool_size=(conv_max_pooling_size[i]), strides=(1, 1), padding='same')(conv1)
        if conv_dropout[i] > 0:
            conv1 = Dropout(rate=conv_dropout[i])(conv1)

    conv2 = Conv2D(64, (1, int(conv1.shape[2])), padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_reshape = Reshape(target_shape=(-1, 64, 1))(conv2)

    x = conv2_reshape
    # transcription block
    x = Reshape(target_shape=(-1, 64))(x)

    # context windows
    pad_frames = (context_frames - 1) / 2

    if context_frames > 1:
        rnn2 = Lambda(_add_context, arguments={'context_frames': context_frames,
                                                'pad_frames': pad_frames})(x)
    else:
        rnn2 = x

    for i in range(rnn_layers[1]):
        if rnn_types[1] == 'lstm':
            rnn2 = LSTM(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'gru':
            rnn2 = GRU(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'rnn':
            rnn2 = SimpleRNN(units=rnn_units[1], return_sequences=True)(rnn2)
        elif rnn_types[1] == 'blstm':
            rnn2 = Bidirectional(LSTM(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'bgru':
            rnn2 = Bidirectional(GRU(units=rnn_units[1], return_sequences=True))(rnn2)
        elif rnn_types[1] == 'brnn':
            rnn2 = Bidirectional(SimpleRNN(units=rnn_units[1], return_sequences=True))(rnn2)
        else:
            raise Exception('Unknown RNN layer')

    outputs = []

    if has_denoised_output:
        outputs.append(denoised_output)

    if has_transcript_full_output:
        transcript_full_output = TimeDistributed(Dense(units=full_output_pattern_voices, activation='sigmoid'),
                                                 name='transcript_full_output')(rnn2)
        transcript_full_output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(transcript_full_output)
        outputs.append(transcript_full_output)

    if has_transcript_reduced_output:
        transcript_reduced_output = TimeDistributed(Dense(units=reduced_output_pattern_voices, activation='sigmoid'), name='transcript_reduced_output')(rnn2)
        transcript_reduced_output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(transcript_reduced_output)
        outputs.append(transcript_reduced_output)

    if has_beat_output:
        beat_output = TimeDistributed(Dense(units=2, activation='sigmoid'), name='beat_output')(rnn2)
        beat_output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(beat_output)
        outputs.append(beat_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=list(loss),
                  optimizer=optimizer,
                  metrics=dict(metrics),
                  loss_weights=list(loss_weights),
                  sample_weight_mode=sample_weight_mode)

    return model