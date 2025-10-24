
"""Assortment of CNN (and GNN) architectures for tracking single cells"""




import ast
import math

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv3D, LSTM
from tensorflow.keras.layers import Input, Concatenate, InputLayer
from tensorflow.keras.layers import Subtract, Dense, Reshape
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import Activation, Softmax
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Lambda
from tensorflow.python.framework import tensor_shape

from tensorflow.keras.layers import Layer, InputSpec

from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras.utils import conv_utils

from spektral.layers import GCSConv, GCNConv, GATConv

class ImageNormalization2D(Layer):
    """Image Normalization layer for 2D data.

    Args:
        norm_method (str): Normalization method to use, one of:
            "std", "max", "whole_image", None.
        filter_size (int): The length of the convolution window.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        activation (function): Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: ``a(x) = x``).
        use_bias (bool): Whether the layer uses a bias.
        kernel_initializer (function): Initializer for the ``kernel`` weights
            matrix, used for the linear transformation of the inputs.
        bias_initializer (function): Initializer for the bias vector. If None,
            the default initializer will be used.
        kernel_regularizer (function): Regularizer function applied to the
            ``kernel`` weights matrix.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
    """
    def __init__(self,
                 norm_method='std',
                 filter_size=61,
                 data_format=None,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.valid_modes = {'std', 'max', None, 'whole_image'}
        if norm_method not in self.valid_modes:
            raise ValueError(f'Invalid `norm_method`: "{norm_method}". '
                             f'Use one of {self.valid_modes}.')
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super().__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)  # hardcoded for 2D data

        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3  # hardcoded for 2D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4, '
                             'received input shape: %s' % input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})

        kernel_shape = (self.filter_size, self.filter_size, input_dim, 1)
        # self.kernel = self.add_weight(
        #     name='kernel',
        #     shape=kernel_shape,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     trainable=False,
        #     dtype=self.compute_dtype)

        W = K.ones(kernel_shape, dtype=self.compute_dtype)
        W = W / K.cast(K.prod(K.int_shape(W)), dtype=self.compute_dtype)
        self.kernel = W
        # self.set_weights([W])

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filter_size, self.filter_size),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=False,
                dtype=self.compute_dtype)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        # Depthwise convolution on CPU is only supported for NHWC format
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 1])
        outputs = tf.nn.depthwise_conv2d(inputs, self.kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')
        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 3, 1, 2])
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(K.square(inputs))
        output = K.sqrt(c2 - c1 * c1) + epsilon
        return output

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'whole_image':
            axes = [2, 3] if self.channel_axis == 1 else [1, 2]
            outputs = inputs - K.mean(inputs, axis=axes, keepdims=True)
            outputs = outputs / (K.std(inputs, axis=axes, keepdims=True) + K.epsilon())

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs = outputs / self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / K.max(inputs)
            outputs = outputs - self._average_filter(outputs)

        else:
            raise NotImplementedError(f'"{self.norm_method}" is not a valid norm_method')

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Comparison(Layer):
    """Layer for comparing two sequences of inputs."""
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]

        x = tf.expand_dims(x, 3)
        multiples = [1, 1, 1, tf.shape(y)[2], 1]
        x = tf.tile(x, multiples)

        y = tf.expand_dims(y, 2)
        multiples = [1, 1, tf.shape(x)[2], 1, 1]
        y = tf.tile(y, multiples)

        return tf.concat([x, y], axis=-1)
    

class DeltaReshape(Layer):
    """Reshape changes between current and future frames"""
    def call(self, inputs):
        current = inputs[0]
        future = inputs[1]
        current = tf.expand_dims(current, axis=3)
        multiples = [1, 1, 1, tf.shape(future)[2], 1]
        output = tf.tile(current, multiples)
        return output
    

class Unmerge(Layer):
    """Unmerge temporal inputs"""
    def __init__(self, track_length, max_cells, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.track_length = track_length
        self.max_cells = max_cells
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        new_shape = [-1, self.track_length, self.max_cells, self.embedding_dim]
        output = tf.reshape(inputs, new_shape)
        return output

    def get_config(self):
        config = {
            'track_length': self.track_length,
            'max_cells': self.max_cells,
            'embedding_dim': self.embedding_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class TemporalMerge(Layer):
    """Layer for merging the time dimension of a Tensor.

    Args:
        encoder_dim (int): desired encoder dimension.
    """
    def __init__(self, encoder_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.lstm = tf.keras.layers.LSTM(
            self.encoder_dim,
            return_sequences=True,
            name=f'{self.name}_lstm')

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        # reshape away the temporal axis
        x = tf.reshape(inputs, [-1, input_shape[1], self.encoder_dim])
        x = self.lstm(x)
        output_shape = [-1, input_shape[1], input_shape[2], self.encoder_dim]
        x = tf.reshape(x, output_shape)
        return x

    def get_config(self):
        config = {
            'encoder_dim': self.encoder_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GNNTrackingModel:
    """Creates a tracking model based on Graph Neural Networks(GNNs).

    Args:
        max_cells (int): maximum number of tracks per movie in dataset
        track_length (int): track length (parameter defined in dataset obj)
        n_filters (int): Number of filters
        encoder_dim (int): Dimension of encoder
        embedding_dim (int): Dimension of embedding
        n_layers (int): number of layers
        graph_layer (str): Must be one of {'gcs', 'gcn', 'gat'}
            Additional kwargs for the graph layers can be encoded in the following format
            ``<layer name>-kwarg:value-kwarg:value``
        appearance_shape (tuple): shape of each object's appearance tensor
        norm_layer (str): Must be one of {'layer', 'batch'}
        appearance_norm (bool): Whether to apply an input normalization layer
            to the appearance head
    """
    def __init__(self,
                 max_cells=39,
                 track_length=8,
                 n_filters=64,
                 encoder_dim=64,
                 embedding_dim=64,
                 n_layers=3,
                 graph_layer='gcs',
                 appearance_shape=(32, 32, 1),
                 norm_layer='batch',
                 appearance_norm=True):

        self.n_filters = n_filters
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.max_cells = max_cells
        self.track_length = track_length
        self.appearance_norm = appearance_norm

        if len(appearance_shape) != 3:
            raise ValueError('appearanace_shape should be a '
                             'tuple of length 3.')
        log2 = math.log(appearance_shape[0], 2)
        if appearance_shape[0] != appearance_shape[1] or int(log2) != log2:
            raise ValueError('appearance_shape should have square dimensions '
                             'and each side should be a power of 2.')

        graph_layer_name = str(graph_layer.split('-')[0]).lower()
        if graph_layer_name not in {'gcn', 'gcs', 'gat'}:
            raise ValueError(f'Invalid graph_layer: {graph_layer_name}')
        self.graph_layer = graph_layer

        norm_options = {'layer', 'batch'}
        if norm_layer not in norm_options:
            raise ValueError('Invalid normalization layer {}. Must be one of {}.'.format(
                norm_layer, norm_options))
        if norm_layer == 'layer':
            self.norm_layer = LayerNormalization
            self.norm_layer_prefix = 'ln'
        elif norm_layer == 'batch':
            self.norm_layer = BatchNormalization
            self.norm_layer_prefix = 'bn'

        # Use inputs to build expected shapes
        base_shape = [self.track_length, self.max_cells]
        self.appearance_shape = tuple(base_shape + list(appearance_shape))
        self.morphology_shape = tuple(base_shape + [3])
        self.centroid_shape = tuple(base_shape + [2])
        self.adj_shape = tuple(base_shape + [self.max_cells])

        # Create encoders and decoders
        self.unmerge_embeddings_model = self.get_unmerge_embeddings_model()
        self.unmerge_centroids_model = self.get_unmerge_centroids_model()
        self.embedding_temporal_merge_model = self.get_embedding_temporal_merge_model()
        self.delta_temporal_merge_model = self.get_delta_temporal_merge_model()
        self.appearance_encoder = self.get_appearance_encoder()
        self.morphology_encoder = self.get_morphology_encoder()
        self.centroid_encoder = self.get_centroid_encoder()
        self.delta_encoder, self.delta_across_frames_encoder = self.get_delta_encoders()
        self.neighborhood_encoder = self.get_neighborhood_encoder()
        self.tracking_decoder = self.get_tracking_decoder()

        # Create branches
        self.training_branch = self.get_training_branch()
        self.inference_branch = self.get_inference_branch()

        # Create model
        self.training_model, self.inference_model = self.get_models()

    def get_embedding_temporal_merge_model(self):
        inputs = Input(shape=(None, None, self.embedding_dim),
                       name='embedding_temporal_merge_input')

        x = TemporalMerge(self.embedding_dim, name='emb_tm')(inputs)
        return Model(inputs=inputs, outputs=x, name='embedding_temporal_merge')

    def get_delta_temporal_merge_model(self):
        inputs = Input(shape=(None, None, self.encoder_dim),
                       name='centroid_temporal_merge_input')

        x = inputs
        x = TemporalMerge(self.encoder_dim, name='delta_tm')(inputs)
        return Model(inputs=inputs, outputs=x, name='delta_temporal_merge')

    def get_appearance_encoder(self):
        app_shape = tuple([None] + list(self.appearance_shape)[2:])
        inputs = Input(shape=app_shape, name='encoder_app_input')

        x = inputs
        if self.appearance_norm:
            x = TimeDistributed(ImageNormalization2D(norm_method='whole_image',
                                                    name='imgnrm_ae'))(x)

        for i in range(int(math.log(app_shape[1], 2))):
            x = Conv3D(self.n_filters,
                       (1, 3, 3),
                       strides=1,
                       padding='same',
                       use_bias=False, name=f'conv3d_ae{i}')(x)
            x = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_ae{i}')(x)
            x = Activation('relu', name=f'relu_ae{i}')(x)
            x = MaxPool3D(pool_size=(1, 2, 2))(x)
        x = Lambda(lambda t: tf.squeeze(t, axis=(2, 3)))(x)
        x = Dense(self.encoder_dim, name='dense_aeout')(x)
        x = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_aeout')(x)
        x = Activation('relu', name='appearance_embedding')(x)
        return Model(inputs=inputs, outputs=x)

    def get_morphology_encoder(self):
        morph_shape = (None, self.morphology_shape[-1])
        inputs = Input(shape=morph_shape, name='encoder_morph_input')
        x = inputs
        x = Dense(self.encoder_dim, name='dense_me')(x)
        x = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_me')(x)
        x = Activation('relu', name='morphology_embedding')(x)
        return Model(inputs=inputs, outputs=x)

    def get_centroid_encoder(self):
        centroid_shape = (None, self.centroid_shape[-1])
        inputs = Input(shape=centroid_shape, name='encoder_centroid_input')
        x = inputs
        x = Dense(self.encoder_dim, name='dense_ce')(x)
        x = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_ce')(x)
        x = Activation('relu', name='centroid_embedding')(x)
        return Model(inputs=inputs, outputs=x)

    def get_delta_encoders(self):
        inputs = Input(shape=(None, None, self.centroid_shape[-1]),
                       name='encoder_delta_input')

        inputs_across_frames = Input(shape=(None, None, None, self.centroid_shape[-1]),
                                     name='encoder_delta_across_frames_input')

        d = Dense(self.encoder_dim, name='dense_des')
        a = Activation('relu', name='relu_des')

        x_0 = d(inputs)
        x_0 = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_des0')(x_0)
        x_0 = a(x_0)

        x_1 = d(inputs_across_frames)
        x_1 = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_des1')(x_1)
        x_1 = a(x_1)

        delta_encoder = Model(inputs=inputs, outputs=x_0)
        delta_across_frames_encoder = Model(inputs=inputs_across_frames, outputs=x_1)

        return delta_encoder, delta_across_frames_encoder

    def get_neighborhood_encoder(self):
        app_input = self.appearance_encoder.input
        morph_input = self.morphology_encoder.input
        centroid_input = self.centroid_encoder.input

        adj_input = Input(shape=(None, None), name='encoder_adj_input')

        app_features = self.appearance_encoder.output
        morph_features = self.morphology_encoder.output
        centroid_features = self.centroid_encoder.output

        adj = adj_input

        # Concatenate features
        node_features = Concatenate(axis=-1)([app_features, morph_features, centroid_features])
        node_features = Dense(self.n_filters, name='dense_ne0')(node_features)
        node_features = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_ne0'
                                        )(node_features)
        node_features = Activation('relu', name='relu_ne0')(node_features)

        # Apply graph convolution
        # Extract and define layer name
        graph_layer_name = str(self.graph_layer.split('-')[0]).lower()
        # Extract layer kwargs
        split = self.graph_layer.split('-')
        layer_kwargs = {}
        if len(split) > 1:
            for item in split[1:]:
                k, v = item.split(':')
                # Cast value to correct type
                try:
                    layer_kwargs[k] = ast.literal_eval(v)
                except ValueError:
                    layer_kwargs[k] = v
        for i in range(self.n_layers):
            name = f'{graph_layer_name}{i}'
            if graph_layer_name == 'gcn':
                graph_layer = GCNConv(self.n_filters, activation=None, name=name, **layer_kwargs)
            elif graph_layer_name == 'gcs':
                graph_layer = GCSConv(self.n_filters, activation=None, name=name, **layer_kwargs)
            elif graph_layer_name == 'gat':
                graph_layer = GATConv(self.n_filters, activation=None, name=name, **layer_kwargs)
            else:
                raise ValueError(f'Unexpected graph_layer: {graph_layer_name}')

            node_features = graph_layer([node_features, adj])
            node_features = self.norm_layer(axis=-1,
                                            name=f'{self.norm_layer_prefix}_ne{i + 1}'
                                            )(node_features)
            node_features = Activation('relu', name=f'relu_ne{i + 1}')(node_features)

        concat = Concatenate(axis=-1)([app_features, morph_features, node_features])
        node_features = Dense(self.embedding_dim, name='dense_nef')(concat)
        node_features = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_nef'
                                        )(node_features)
        node_features = Activation('relu', name='relu_nef')(node_features)

        inputs = [app_input, morph_input, centroid_input, adj_input]
        outputs = [node_features, centroid_input]

        return Model(inputs=inputs, outputs=outputs, name='neighborhood_encoder')

    def get_unmerge_embeddings_model(self):
        inputs = Input(shape=(self.appearance_shape[1], self.embedding_dim),
                       name='unmerge_embeddings_input')
        x = inputs
        x = Unmerge(self.track_length,
                    self.appearance_shape[1],
                    self.embedding_dim,
                    name='unmerge_embeddings')(x)
        return Model(inputs=inputs, outputs=x, name='unmerge_embeddings_model')

    def get_unmerge_centroids_model(self):
        inputs = Input(shape=(self.centroid_shape[1], self.centroid_shape[-1]),
                       name='unmerge_centroids_input')
        x = inputs
        x = Unmerge(self.track_length,
                    self.centroid_shape[1],
                    self.centroid_shape[2],
                    name='unmerge_centroids')(x)

        return Model(inputs=inputs, outputs=x, name='unmerge_centroids_model')

    def _get_deltas(self, x):
        """Convert raw positions to deltas"""
        deltas = Lambda(lambda t: t[:, 1:] - t[:, 0:-1])(x)
        deltas = Lambda(lambda t: tf.pad(t, tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])))(deltas)
        return deltas

    def _get_deltas_across_frames(self, centroids):
        """Find deltas across frames"""
        centroid_current = Lambda(lambda t: t[:, 0:-1])(centroids)
        centroid_future = Lambda(lambda t: t[:, 1:])(centroids)
        centroid_current = Lambda(lambda t: tf.expand_dims(t, 3))(centroid_current)
        centroid_future = Lambda(lambda t: tf.expand_dims(t, 2))(centroid_future)
        deltas_across_frames = Subtract()([centroid_future, centroid_current])
        return deltas_across_frames

    def get_training_branch(self):
        # Define inputs
        app_input = Input(shape=self.appearance_shape, name='appearances')
        morph_input = Input(shape=self.morphology_shape, name='morphologies')
        centroid_input = Input(shape=self.centroid_shape, name='centroids')
        adj_input = Input(shape=self.adj_shape, name='adj_matrices')
        inputs = [app_input, morph_input, centroid_input, adj_input]

        # Merge batch and temporal dimensions
        new_app_shape = tuple([-1] + list(self.appearance_shape)[1:])
        reshaped_app_input = Lambda(lambda t: tf.reshape(t, new_app_shape),
                                    name='reshaped_appearances')(app_input)

        new_morph_shape = tuple([-1] + list(self.morphology_shape)[1:])
        reshaped_morph_input = Lambda(lambda t: tf.reshape(t, new_morph_shape),
                                      name='reshaped_morphologies')(morph_input)

        new_centroid_shape = tuple([-1] + list(self.centroid_shape)[1:])
        reshaped_centroid_input = Lambda(lambda t: tf.reshape(t, new_centroid_shape),
                                         name='reshaped_centroids')(centroid_input)

        new_adj_shape = [-1, self.adj_shape[1], self.adj_shape[2]]
        reshaped_adj_input = Lambda(lambda t: tf.reshape(t, new_adj_shape),
                                    name='reshaped_adj_matrices')(adj_input)

        reshaped_inputs = [
            reshaped_app_input,
            reshaped_morph_input,
            reshaped_centroid_input,
            reshaped_adj_input
        ]

        x, centroids = self.neighborhood_encoder(reshaped_inputs)

        # Reshape embeddings to add back temporal dimension
        x = self.unmerge_embeddings_model(x)
        centroids = self.unmerge_centroids_model(centroids)

        # Get current and future embeddings
        x_current = Lambda(lambda t: t[:, 0:-1])(x)
        x_future = Lambda(lambda t: t[:, 1:])(x)

        # Integrate temporal information for embeddings and compare
        x_current = self.embedding_temporal_merge_model(x_current)
        x = Comparison(name='training_embedding_comparison')([x_current, x_future])

        # Convert centroids to deltas
        deltas_current = self._get_deltas(centroids)
        deltas_future = self._get_deltas_across_frames(centroids)

        deltas_current = Activation(tf.math.abs, name='act_dc_tb')(deltas_current)
        deltas_future = Activation(tf.math.abs, name='act_df_tb')(deltas_future)

        deltas_current = self.delta_encoder(deltas_current)
        deltas_future = self.delta_across_frames_encoder(deltas_future)

        deltas_current = Lambda(lambda t: t[:, 0:-1])(deltas_current)
        deltas_current = self.delta_temporal_merge_model(deltas_current)
        deltas_current = Lambda(lambda t: tf.expand_dims(t, 3))(deltas_current)
        multiples = [1, 1, 1, self.centroid_shape[1], 1]
        deltas_current = Lambda(lambda t: tf.tile(t, multiples))(deltas_current)

        deltas = Concatenate(axis=-1)([deltas_current, deltas_future])

        outputs = [x, deltas]

        # Create submodel
        return Model(inputs=inputs, outputs=outputs, name='training_branch')

    def get_inference_branch(self):
        # batch size, tracks
        current_embedding = Input(shape=(None, None, self.embedding_dim),
                                  name='current_embeddings')
        current_centroids = Input(shape=(None, None, self.centroid_shape[-1]),
                                  name='current_centroids')

        future_embedding = Input(shape=(1, None, self.embedding_dim),
                                 name='future_embeddings')
        future_centroids = Input(shape=(1, None, self.centroid_shape[-1]),
                                 name='future_centroids')

        inputs = [current_embedding, current_centroids,
                  future_embedding, future_centroids]

        # Embeddings - Integrate temporal information
        x_current = self.embedding_temporal_merge_model(current_embedding)

        # Embeddings - Get final frame from current track
        x_current = Lambda(lambda t: t[:, -1:])(x_current)

        x = Comparison(name='inference_comparison')([x_current, future_embedding])

        # Centroids - Get deltas
        deltas_current = self._get_deltas(current_centroids)
        deltas_current = Activation(tf.math.abs, name='act_dc_ib')(deltas_current)

        deltas_current = self.delta_encoder(deltas_current)
        deltas_current = self.delta_temporal_merge_model(deltas_current)
        deltas_current = Lambda(lambda t: t[:, -1:])(deltas_current)

        # Centroids - Get deltas across frames
        centroid_current_end = Lambda(lambda t: t[:, -1:])(current_centroids)
        centroid_current_end = Lambda(lambda t: tf.expand_dims(t, 3))(centroid_current_end)
        centroid_future = Lambda(lambda t: tf.expand_dims(t, 2))(future_centroids)
        deltas_future = Subtract()([centroid_future, centroid_current_end])
        deltas_future = Activation(tf.math.abs, name='act_df_ib')(deltas_future)
        deltas_future = self.delta_across_frames_encoder(deltas_future)

        deltas_current = DeltaReshape(name='delta_reshape')([deltas_current, future_centroids])

        deltas = Concatenate(axis=-1)([deltas_current, deltas_future])

        outputs = [x, deltas]

        return Model(inputs=inputs, outputs=outputs, name='inference_branch')

    def get_tracking_decoder(self):
        embedding_input = Input(shape=(None, None, None, 2 * self.embedding_dim))
        deltas_input = Input(shape=(None, None, None, 2 * self.encoder_dim))

        embedding = Concatenate(axis=-1)([embedding_input, deltas_input])

        embedding = Dense(self.n_filters, name='dense_td0')(embedding)
        embedding = self.norm_layer(axis=-1, name=f'{self.norm_layer_prefix}_td0'
                                    )(embedding)
        embedding = Activation('relu', name='relu_td0')(embedding)

        # TODO: set to n_classes
        embedding = Dense(3, name='dense_outembed')(embedding)

        # Add classification head
        output = Softmax(axis=-1, name='softmax_comparison')(embedding)

        return Model(inputs=[embedding_input, deltas_input],
                     outputs=output,
                     name='tracking_decoder')

    def get_models(self):
        # Create inputs
        training_inputs = self.training_branch.input
        inference_inputs = self.inference_branch.input

        # Apply decoder
        training_output = self.tracking_decoder(self.training_branch.output)
        inference_output = self.tracking_decoder(self.inference_branch.output)

        # Name the training output layer
        training_output = Lambda(lambda t: t, name='temporal_adj_matrices')(training_output)

        training_model = Model(inputs=training_inputs, outputs=training_output)
        inference_model = Model(inputs=inference_inputs, outputs=inference_output)

        return training_model, inference_model
    

# def siamese_model(input_shape=None,
#                   features=None,
#                   neighborhood_scale_size=10,
#                   reg=1e-5,
#                   init='he_normal',
#                   filter_size=61):
#     """Creates a tracking model based on Siamese Neural Networks(SNNs).

#     Args:
#         input_shape (tuple): If no input tensor, create one with this shape.
#         features (list): Number of output features
#         neighborhood_scale_size (int): number of input channels
#         reg (int): regularization value
#         init (str): Method for initalizing weights
#         filter_size (int): the receptive field of the neural network

#     Returns:
#         tensorflow.keras.Model: 2D FeatureNet
#     """
#     def compute_input_shape(feature):
#         if feature == 'appearance':
#             return input_shape
#         elif feature == 'distance':
#             return (None, 2)
#         elif feature == 'neighborhood':
#             return (None, 2 * neighborhood_scale_size + 1,
#                     2 * neighborhood_scale_size + 1,
#                     input_shape[-1])
#         elif feature == 'regionprop':
#             return (None, 3)
#         else:
#             raise ValueError('siamese_model.compute_input_shape: '
#                              f'Unknown feature `{feature}`')

#     def compute_reshape(feature):
#         if feature == 'appearance':
#             return (64,)
#         elif feature == 'distance':
#             return (2,)
#         elif feature == 'neighborhood':
#             return (64,)
#         elif feature == 'regionprop':
#             return (3,)
#         else:
#             raise ValueError('siamese_model.compute_output_shape: '
#                              f'Unknown feature `{feature}`')

#     def compute_feature_extractor(feature, shape):
#         if feature == 'appearance':
#             # This should not stay: channels_first/last should be used to
#             # dictate size (1 works for either right now)
#             N_layers = np.int_(np.floor(np.log2(input_shape[1])))
#             feature_extractor = Sequential()
#             feature_extractor.add(InputLayer(input_shape=shape))
#             # feature_extractor.add(ImageNormalization2D('std', filter_size=32))
#             for layer in range(N_layers):
#                 feature_extractor.add(Conv3D(64, (1, 3, 3),
#                                              kernel_initializer=init,
#                                              padding='same',
#                                              kernel_regularizer=l2(reg)))
#                 feature_extractor.add(BatchNormalization(axis=channel_axis))
#                 feature_extractor.add(Activation('relu'))
#                 feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))

#             feature_extractor.add(Reshape((-1, 64)))
#             return feature_extractor

#         elif feature == 'distance':
#             return None
#         elif feature == 'neighborhood':
#             N_layers_og = np.int_(np.floor(np.log2(2 * neighborhood_scale_size + 1)))
#             feature_extractor_neighborhood = Sequential()
#             feature_extractor_neighborhood.add(
#                 InputLayer(input_shape=shape)
#             )
#             for layer in range(N_layers_og):
#                 feature_extractor_neighborhood.add(Conv3D(64, (1, 3, 3),
#                                                           kernel_initializer=init,
#                                                           padding='same',
#                                                           kernel_regularizer=l2(reg)))
#                 feature_extractor_neighborhood.add(BatchNormalization(axis=channel_axis))
#                 feature_extractor_neighborhood.add(Activation('relu'))
#                 feature_extractor_neighborhood.add(MaxPool3D(pool_size=(1, 2, 2)))

#             feature_extractor_neighborhood.add(Reshape((-1, 64)))

#             return feature_extractor_neighborhood
#         elif feature == 'regionprop':
#             return None
#         else:
#             raise ValueError('siamese_model.compute_feature_extractor: '
#                              f'Unknown feature `{feature}`')

#     if features is None:
#         raise ValueError('siamese_model: No features specified.')

#     if K.image_data_format() == 'channels_first':
#         channel_axis = 1
#         raise ValueError('siamese_model: Only channels_last is supported.')
#     else:
#         channel_axis = -1

#     input_shape = tuple([None] + list(input_shape))

#     features = sorted(features)

#     inputs = []
#     outputs = []
#     for feature in features:
#         in_shape = compute_input_shape(feature)
#         re_shape = compute_reshape(feature)
#         feature_extractor = compute_feature_extractor(feature, in_shape)

#         layer_1 = Input(shape=in_shape, name=f'{feature}_input1')
#         layer_2 = Input(shape=in_shape, name=f'{feature}_input2')

#         inputs.extend([layer_1, layer_2])

#         # apply feature_extractor if it exists
#         if feature_extractor is not None:
#             layer_1 = feature_extractor(layer_1)
#             layer_2 = feature_extractor(layer_2)

#         # LSTM on 'left' side of network since that side takes in stacks of features
#         layer_1 = LSTM(64)(layer_1)
#         layer_2 = Reshape(re_shape)(layer_2)

#         outputs.append([layer_1, layer_2])

#     dense_merged = []
#     for layer_1, layer_2 in outputs:
#         merge = Concatenate(axis=channel_axis)([layer_1, layer_2])
#         dense_merge = Dense(128)(merge)
#         bn_merge = BatchNormalization(axis=channel_axis)(dense_merge)
#         dense_relu = Activation('relu')(bn_merge)
#         dense_merged.append(dense_relu)

#     # Concatenate outputs from both instances
#     merged_outputs = Concatenate(axis=channel_axis)(dense_merged)

#     # Add dense layers
#     dense1 = Dense(128)(merged_outputs)
#     bn1 = BatchNormalization(axis=channel_axis)(dense1)
#     relu1 = Activation('relu')(bn1)
#     dense2 = Dense(128)(relu1)
#     bn2 = BatchNormalization(axis=channel_axis)(dense2)
#     relu2 = Activation('relu')(bn2)
#     dense3 = Dense(3, activation='softmax', name='classification', dtype=K.floatx())(relu2)

#     # Instantiate model
#     final_layer = dense3
#     model = Model(inputs=inputs, outputs=final_layer)

#     return model

