import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras.utils import conv_utils


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