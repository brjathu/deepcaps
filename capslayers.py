from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import layers, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.layers import InputSpec
from keras.utils.conv_utils import conv_output_length

cf = K.image_data_format() == '..'
useGPU = True


def squeeze(s):
    sq = K.sum(K.square(s), axis=-1, keepdims=True)
    return (sq / (1 + sq)) * (s / K.sqrt(sq + K.epsilon()))


class ConvertToCaps(layers.Layer):

    def __init__(self, **kwargs):
        super(ConvertToCaps, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=2)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(1 if cf else len(output_shape), 1)
        return tuple(output_shape)

    def call(self, inputs):
        return K.expand_dims(inputs, 1 if cf else -1)

    def get_config(self):
        config = {
            'input_spec': 5
        }
        base_config = super(ConvertToCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FlattenCaps(layers.Layer):

    def __init__(self, **kwargs):
        super(FlattenCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=4)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1])

    def call(self, inputs):
        shape = K.int_shape(inputs)
        return K.reshape(inputs, (-1, np.prod(shape[1:-1]), shape[-1]))


class CapsToScalars(layers.Layer):

    def __init__(self, **kwargs):
        super(CapsToScalars, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs + K.epsilon()), axis=-1))


class Conv2DCaps(layers.Layer):

    def __init__(self, ch_j, n_j,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 r_num=1,
                 b_alphas=[8, 8, 8],
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(Conv2DCaps, self).__init__(**kwargs)
        rank = 2
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.r_num = r_num
        self.b_alphas = b_alphas
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = (1, 1)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=rank + 3)

    def build(self, input_shape):

        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.ah_j, self.aw_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                              self.kernel_size[i],
                                                              padding=self.padding,
                                                              stride=1,
                                                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.w_shape = self.kernel_size + (self.ch_i, self.n_i,
                                           self.ch_j, self.n_j)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs):
        if self.r_num == 1:
            # if there is no routing (and this is so when r_num is 1 and all c are equal)
            # then this is a common convolution
            outputs = K.conv2d(K.reshape(inputs, (-1, self.h_i, self.w_i,
                                                  self.ch_i * self.n_i)),
                               K.reshape(self.w, self.kernel_size +
                                         (self.ch_i * self.n_i, self.ch_j * self.n_j)),
                               data_format='channels_last',
                               strides=self.strides,
                               padding=self.padding,
                               dilation_rate=self.dilation_rate)

            outputs = squeeze(K.reshape(outputs, ((-1, self.h_j, self.w_j,
                                                   self.ch_j, self.n_j))))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h_j, self.w_j, self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'b_alphas': self.b_alphas,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(Conv2DCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


class Mask_CID(layers.Layer):

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, a = inputs
            mask = K.argmax(a, 1)
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.argmax(x, 1)

        increasing = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)
        m = tf.stack([increasing, tf.cast(mask, tf.int32)], axis=1)
        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        # x1 = tf.transpose(inputs, (0))
        masked = tf.gather_nd(inputs, m)

        return masked

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):  # true label provided
            return tuple([None, input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[2]])


class ConvCapsuleLayer3D(layers.Layer):

    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='valid', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer3D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_atoms, self.kernel_size, self.kernel_size, 1, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[self.num_capsule, self.num_atoms, 1, 1],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):

        input_transposed = tf.transpose(input_tensor, [0, 3, 4, 1, 2])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_tensor, [input_shape[0], 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width])

        input_tensor_reshaped.set_shape((None, 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width))

        # conv = Conv3D(input_tensor_reshaped, self.W, (self.strides, self.strides),
        #                 padding=self.padding, data_format='channels_first')

        conv = K.conv3d(input_tensor_reshaped, self.W, strides=(self.input_num_atoms, self.strides, self.strides), padding=self.padding, data_format='channels_first')

        votes_shape = K.shape(conv)
        _, _, _, conv_height, conv_width = conv.get_shape()
        conv = tf.transpose(conv, [0, 2, 1, 3, 4])
        votes = K.reshape(conv, [input_shape[0], self.input_num_capsule, self.num_capsule, self.num_atoms, votes_shape[3], votes_shape[4]])
        votes.set_shape((None, self.input_num_capsule, self.num_capsule, self.num_atoms, conv_height.value, conv_width.value))

        logit_shape = K.stack([input_shape[0], self.input_num_capsule, self.num_capsule, votes_shape[3], votes_shape[4]])
        biases_replicated = K.tile(self.b, [1, 1, conv_height.value, conv_width.value])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        a2 = tf.transpose(activations, [0, 3, 4, 1, 2])
        return a2

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(space[i], self.kernel_size, padding=self.padding, stride=self.strides, dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [3, 0, 1, 2, 4, 5]
        r_t_shape = [1, 2, 3, 0, 4, 5]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        a,b,c,d,e = logits.get_shape()
        a = logit_shape[0]
        b = logit_shape[1]
        c = logit_shape[2]
        d = logit_shape[3]
        e = logit_shape[4]
        print(logit_shape)
        logit_temp = tf.reshape(logits, [a,b,-1])
        route_temp = tf.nn.softmax(logit_temp, dim=-1)
        route = tf.reshape(route_temp, [a, b, c, d, e])
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        # activation = _squash(preactivate)
        activation = squash(preactivate, axis=[-1, -2, -3])
        activations = activations.write(i, activation)

        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)
    a = K.cast(activations.read(num_routing - 1), dtype='float32')
    return K.cast(activations.read(num_routing - 1), dtype='float32')


class DenseCaps(layers.Layer):

    def __init__(self, ch_j, n_j,
                 r_num=1,
                 b_alphas=[8, 8, 8],
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseCaps, self).__init__(**kwargs)
        self.ch_j = ch_j  # number of capsules in layer J
        self.n_j = n_j  # number of neurons in a capsule in J
        self.r_num = r_num
        self.b_alphas = b_alphas
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        self.ch_i, self.n_i = input_shape[1:]

        self.w_shape = (self.ch_i, self.n_i, self.ch_j, self.n_j)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs):
        if self.r_num == 1:
            outputs = K.dot(K.reshape(inputs, (-1, self.ch_i * self.n_i)),
                            K.reshape(self.w, (self.ch_i * self.n_i,
                                               self.ch_j * self.n_j)))
            outputs = squeeze(K.reshape(outputs, (-1, self.ch_j, self.n_j)))
        else:
            wr = K.reshape(self.w, (self.ch_i, self.n_i, self.ch_j * self.n_j))

            u = tf.transpose(tf.matmul(tf.transpose(inputs, [1, 0, 2]), wr), [1, 0, 2])

            u = K.reshape(u, (-1, self.ch_i, self.ch_j, self.n_j))

            def rt(ub):
                ub = K.reshape(ub, (-1, self.ch_i, self.ch_j, self.n_j))
                ub_wo_g = K.stop_gradient(ub)
                b = 0.0
                for r in range(self.r_num):
                    if r > 0:
                        c = K.expand_dims(K.softmax(b * self.b_alphas[r])) * self.ch_j  # distribution of weighs of capsules in I across capsules in J
                        c = K.stop_gradient(c)
                    else:
                        c = 1.0

                    if r == self.r_num - 1:
                        cub = c * ub
                    else:
                        cub = c * ub_wo_g
                    s = K.sum(cub, axis=-3)  # vectors of capsules in J
                    v = squeeze(s)  # squeezed vectors of capsules in J
                    if r == self.r_num - 1:
                        break

                    v = K.stop_gradient(v)

                    a = tf.einsum('bjk,bijk->bij', v, ub)  # a = v dot u
                    # a = K.matmul(K.reshape(v, (-1, 1, J, 1, n_j)),
                    #             K.reshape(u, (-1, I, J, n_j, 1))).reshape((-1, I, J))

                    b = b + a  # increase those b[i,j] where v[j] dot b[i,j] is larger
                return v

            u = K.reshape(u, (-1, self.ch_i * self.ch_j * self.n_j))

            global useGPU

            if useGPU:
                outputs = rt(u)
            else:
                outputs = tf.map_fn(rt, u,
                                    parallel_iterations=100, back_prop=True,
                                    infer_shape=False)

            outputs = K.reshape(outputs, (-1, self.ch_j, self.n_j))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'r_num': self.r_num,
            'b_alphas': self.b_alphas,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DenseCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsule, dim_capsule, channels, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        if(self.channels != 0):
            assert int(self.input_num_capsule / self.channels) / (self.input_num_capsule / self.channels) == 1, "error"
            self.W = self.add_weight(shape=[self.num_capsule, self.channels,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')

            self.B = self.add_weight(shape=[self.num_capsule, self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')
        else:
            self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')
            self.B = self.add_weight(shape=[self.num_capsule, self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        if(self.channels != 0):
            W2 = K.repeat_elements(self.W, int(self.input_num_capsule / self.channels), 1)
        else:
            W2 = self.W
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, W2, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]) + self.B)  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


