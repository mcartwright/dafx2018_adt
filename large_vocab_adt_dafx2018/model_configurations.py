#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseModelConfiguration(object):
    """
    Model configuration class that specifies variations of a model when training

    Attributes
    ----------
    name : str
        The configuration name
    config_id : int
        The id of the configuration
    model_args : dict
        The model arguments to override
    input_encoding_id : int
        The id of the function to take an input dict from an example and output a numpy array
    target_encoding_id : int
        The id of the function to take an output dict from an example and output a numpy array
    model_creation_fn_name : str
        The function that defines the Keras model
    batch_size : int
    steps_per_epoch : int
    num_epoch : int
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.config_id = int(self.name[len('ModelConfiguration'):])
        self.model_args = dict()
        self.batch_size = None
        self.steps_per_epoch = None
        self.num_epoch = None
        self.input_encoding_id = None
        self.target_encoding_id = None
        self.feature_configuration_id = None
        self.output_parameters_id = None
        self.model_creation_fn_name = None
        self.model_encoder_decoder_creation_fn_name = "None"
        self._augmentation_id = None
        self.augmentation_id = None
        self.train_dataset_id = None
        self.regularization = None
        self.loss_weights = None

    def get_env_vars(self):
        """
        Set the environment variables for the Makefile
        """
        if self.augmentation_id is not None:
            assert (self.augmentation_id == self.train_dataset_id)

        return dict(MODEL_CONFIG_ID=str(self.config_id),
                    INPUT_ENCODING_ID=str(self.input_encoding_id),
                    TARGET_ENCODING_ID=str(self.target_encoding_id),
                    FEATURE_CONFIGURATION_ID=str(self.feature_configuration_id),
                    OUTPUT_PARAMETERS_ID=str(self.output_parameters_id),
                    AUGMENTATION_ID=str(self.augmentation_id),
                    TRAIN_DATASET_ID=str(self.train_dataset_id))

    @property
    def input_encoding_fn_name(self):
        return 'input_encoding{}'.format(self.input_encoding_id)

    @property
    def target_encoding_fn_name(self):
        return 'target_encoding{}'.format(self.target_encoding_id)

    @property
    def feature_configuration_fn_name(self):
        return 'feature_configuration{}'.format(self.feature_configuration_id)

    @property
    def augmentation_id(self):
        return self._augmentation_id

    @augmentation_id.setter
    def augmentation_id(self, value):
        # make sure that if augmentation_id is set, that dataset_id is set to the same
        self._augmentation_id = value
        self.train_dataset_id = value


# class ModelConfiguration291(BaseModelConfiguration):
#     def __init__(self):
#         super(ModelConfiguration291, self).__init__()
#         self.batch_size = 8
#         self.dataset_size = 200000
#         self.steps_per_epoch = int(self.dataset_size / self.batch_size)
#         self.num_epoch = 5
#         self.num_inputs = 2
#         self.num_outputs = 3
#         self.interval_dataset_ids = [9, 10, 11, 12]
#         self.batch_dataset_ids = [8, ]
#         self.num_validation_samples = 2000
#         self.feature_configuration_id = 6
#         self.output_parameters_id = 8
#         self.augmentation_id = 8
#         self.input_encoding_id = 25
#         self.target_encoding_id = 36
#         self.mixed_length_input_encoding = 23
#         self.mixed_length_target_encoding = 40
#         self.full_transcription_output_index = 0
#         self.reduced_transcription_output_index = 1
#         self.beat_output_index = 2
#         self.pattern_output_index = -1
#         self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
#         self.description = "all real + synthesized. full + reduced + beat"
#         self.model_creation_fn_name = 'make_crnn_trans_mdl'
#         self.model_args = dict(input_shape=(799, 64, 1),
#                                sample_weight_mode='temporal',
#                                full_output_pattern_voices=14,
#                                reduced_output_pattern_voices=3,
#                                use_mag_spec_input=True,
#                                use_onset_signal_input=True,
#                                has_denoised_output=False,
#                                has_transcript_full_output=True,
#                                has_transcript_reduced_output=True,
#                                has_beat_output=True,
#                                rnn_units=(0, 64, ),
#                                rnn_layers=(0, 3, ),
#                                rnn_types=('blstm', 'blstm',),
#                                loss=('keras.losses.binary_crossentropy',
#                                      'keras.losses.binary_crossentropy',
#                                      'keras.losses.binary_crossentropy',),
#                                loss_weights=(0.53, 0.16, 0.31),
#                                context_frames=13,
#                                conv_layers=(2, 2,),
#                                conv_shapes=((3, 3), (3, 3,)),
#                                conv_max_pooling_size=(None, None,),
#                                conv_filters=(32, 64,),
#                                conv_dropout=(0.3, 0.3),
#                                metrics=(('transcribed_reduce', 'binary_accuracy'),
#                                         ('beat', 'binary_accuracy'),),)


class ModelConfiguration289(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration289, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = [9, 10, 11, 12]
        self.batch_dataset_ids = []
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real. full + reduced + beat."
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 64, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration290(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration290, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = []
        self.batch_dataset_ids = [8,]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "synthesized. full + reduced + beat"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 64, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration291(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration291, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = [9, 10, 11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. full + reduced + beat"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 64, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration292(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration292, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = [9, 10, 11, 12]
        self.batch_dataset_ids = []
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real. full + reduced + beat. large."
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 256, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(128, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration293(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration293, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = []
        self.batch_dataset_ids = [8,]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "synthesized. full + reduced + beat. large."
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 256, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(128, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration294(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration294, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 3
        self.interval_dataset_ids = [9, 10, 11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 36
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 40
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = 1
        self.beat_output_index = 2
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. full + reduced + beat. large."
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=True,
                               has_beat_output=True,
                               rnn_units=(0, 256, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',
                                     'keras.losses.binary_crossentropy',),
                               loss_weights=(0.53, 0.16, 0.31),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(128, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration295(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration295, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 1
        self.interval_dataset_ids = [11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 44
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 45
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = -1
        self.beat_output_index = -1
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. full. (8, 11, 12)"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=False,
                               has_beat_output=False,
                               rnn_units=(0, 64, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',),
                               loss_weights=(1,),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration296(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration296, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 1
        self.interval_dataset_ids = [11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 44
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 45
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = -1
        self.beat_output_index = -1
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. full. large. (8, 11, 12)"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=False,
                               has_beat_output=False,
                               rnn_units=(0, 256, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',),
                               loss_weights=(1,),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(128, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)


class ModelConfiguration301(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration301, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 1
        self.interval_dataset_ids = [9, 10, 11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 39
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 43
        self.full_transcription_output_index = -1
        self.reduced_transcription_output_index = 0
        self.beat_output_index = -1
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. reduced. (8, 9, 10, 11, 12)"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=False,
                               has_transcript_reduced_output=True,
                               has_beat_output=False,
                               rnn_units=(0, 64,),
                               rnn_layers=(0, 3,),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',),
                               loss_weights=(1,),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),), )


class ModelConfiguration302(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration302, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 1
        self.interval_dataset_ids = [9, ]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 46
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 47
        self.full_transcription_output_index = -1
        self.reduced_transcription_output_index = -1
        self.beat_output_index = 0
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. beat. (8, 9, )"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=False,
                               has_transcript_reduced_output=False,
                               has_beat_output=True,
                               rnn_units=(0, 64,),
                               rnn_layers=(0, 3,),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',),
                               loss_weights=(1,),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),), )


class ModelConfiguration303(BaseModelConfiguration):
    def __init__(self):
        super(ModelConfiguration303, self).__init__()
        self.batch_size = 8
        self.dataset_size = 200000
        self.steps_per_epoch = int(self.dataset_size / self.batch_size)
        self.num_epoch = 5
        self.num_inputs = 2
        self.num_outputs = 1
        self.interval_dataset_ids = [11, 12]
        self.batch_dataset_ids = [8, ]
        self.num_validation_samples = 2000
        self.feature_configuration_id = 6
        self.output_parameters_id = 8
        self.augmentation_id = 8
        self.input_encoding_id = 25
        self.target_encoding_id = 48
        self.mixed_length_input_encoding = 23
        self.mixed_length_target_encoding = 49
        self.full_transcription_output_index = 0
        self.reduced_transcription_output_index = -1
        self.beat_output_index = -1
        self.pattern_output_index = -1
        self.optimizer_fn_call = 'Adam(clipvalue=1.0)'
        self.description = "all real + synthesized. full. (just 8, 11, 12)"
        self.model_creation_fn_name = 'make_crnn_trans_nc_noskip2_weight_mdl'
        self.model_args = dict(input_shape=(799, 64, 1),
                               sample_weight_mode='temporal',
                               full_output_pattern_voices=14,
                               reduced_output_pattern_voices=3,
                               use_mag_spec_input=True,
                               use_onset_signal_input=True,
                               has_denoised_output=False,
                               has_transcript_full_output=True,
                               has_transcript_reduced_output=False,
                               has_beat_output=False,
                               rnn_units=(0, 64, ),
                               rnn_layers=(0, 3, ),
                               rnn_types=('blstm', 'blstm',),
                               loss=('keras.losses.binary_crossentropy',),
                               loss_weights=(1,),
                               context_frames=13,
                               conv_layers=(2, 2,),
                               conv_shapes=((3, 3), (3, 3,)),
                               conv_max_pooling_size=(None, None,),
                               conv_filters=(32, 64,),
                               conv_dropout=(0.3, 0.3),
                               metrics=(('transcribed_reduce', 'binary_accuracy'),
                                        ('beat', 'binary_accuracy'),),)