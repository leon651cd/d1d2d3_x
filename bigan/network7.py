import tensorflow as tf
"""
E,G,D1,D2
修改网络结构，attention(D1,D2)
输入数据经过pca降维处理，因此所有数据都是一维的
进一步修改，只对D2加2层lstm+attention

"""

learning_rate = 0.01
# batch_size = 5665
# batch_size_a=30 #异常样本的batch数目
batch_size = 30
batch_size_a=99 #异常样本的batch数目
latent_dim = 16
characters=10  #写反了，，为了方便这样改
times=10
init_kernel = tf.contrib.layers.xavier_initializer()
attention_size= 16

def attention(inputs, attention_size, time_major=False,is_training=False, getter=None, reuse=False):
    """
    inputs.shape = batch,times,hidden_n
    """
    with tf.variable_scope('attention', reuse=reuse, custom_getter=getter):
        inputs_shape = inputs.shape
        sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)  # reduce_sum --> reduce_mean
        # 只返回权值
        return alphas

def encoder(x_inp, is_training=False, getter=None, reuse=False):
    """ Encoder architecture in tensorflow
    Maps the data into the latent space
    Args:
        x_inp (tensor): input data for the encoder. x_inp.shape= [batch_size,1,characters]
        reuse (bool): sharing variables or not
    Returns:
        (tensor): last activation layer of the encoder
    """
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):

        name_net = 'enc_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=12,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)

        name_net = 'enc_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=latent_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
        return net             #

def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow
        Generates data from the latent space
        Args:
            z_inp (tensor): variable in the latent space  c.shape= [batch_size,1,latent_dim] 这里的embedding dim为1
            reuse (bool): sharing variables or not
        Returns:
            (tensor): last activation layer of the generator
     """
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'dec_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=16,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'dec_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=12,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net=tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

        name_net = 'dec_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=characters,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)
    return net

def discriminator_1(z_inp, x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow
        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            z_inp (tensor): variable in the latent space
            x_inp (tensor): input data for the encoder.
            reuse (bool): sharing variables or not
        Returns:
            logits (tensor): last activation layer of the discriminator (shape 1)
            intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator_1', reuse=reuse, custom_getter=getter):

        with tf.variable_scope("densex"):
            net_x = tf.layers.dense(x_inp,
                                  units=12,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net_x = leakyReLu(net_x)
            net_x = tf.layers.dropout(net_x, rate=0.2, name='dropout', training=is_training)

        with tf.variable_scope("densez"):

            net_z = tf.layers.dense(z_inp,
                                  units=12,
                                  kernel_initializer=init_kernel,
                                  name='fc1')
            net_z = tf.nn.relu(net_z)

            final=tf.concat([net_x,net_z],axis=2)

        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(final,
                                16,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)


    return logits

def discriminator_2(z_inp, x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow
        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            z_inp (tensor): variable in the latent space
            x_inp (tensor): input data for the encoder.
            reuse (bool): sharing variables or not
        Returns:
            logits (tensor): last activation layer of the discriminator (shape 1)
            intermediate_layer (tensor): intermediate layer for feature matching
        """
    with tf.variable_scope('discriminator_2', reuse=reuse, custom_getter=getter):
        with tf.variable_scope("dis2_lstm"):

            x_prior = tf.unstack(x_inp, characters, 2)  #
            lstm_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicRNNCell(1)) for _ in range(2)])
            res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32)
            """
            weights = tf.Variable(tf.random_normal([4, 1]));
            biases = tf.Variable(tf.random_normal([1]));
            #print(len(res))
            for i in range(len(res)):
                res[i] = tf.nn.tanh(tf.matmul(res[i], weights) + biases);
            """
            tensor_a = tf.convert_to_tensor(res)
            tensor_b = tf.transpose(tensor_a, perm=[1, 2, 0])


        with tf.variable_scope("densex2"):
            net_x = tf.layers.dense(tensor_b,
                                    units=12,
                                    kernel_initializer=init_kernel,
                                    name='fc')
            net_x = leakyReLu(net_x)
            net_x = tf.layers.dropout(net_x, rate=0.2, name='dropout', training=is_training)

        with tf.variable_scope("densez2"):
            net_z = tf.layers.dense(z_inp,
                                    units=12,
                                    kernel_initializer=init_kernel,
                                    name='fc1')
            net_z = tf.nn.relu(net_z)

            final = tf.concat([net_x, net_z], axis=2)

        name_y = 'y_fc_12'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(final,
                                16,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = y

        name_y = 'y_fc_logits2'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)
    return logits

def leakyReLu(x, alpha=0.1, name='leaky_relu'):
    """ Leaky relu """
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


# x=tf.constant(5., shape=[13,1,10])
# z=tf.constant(3., shape=[13,1,16])
#
# out,a=discriminator_2(z,x)
# print(out)
# print(a.shape)