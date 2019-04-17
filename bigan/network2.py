import tensorflow as tf
"""
没加attention
Generator (decoder), encoder and discriminator.
只有一个dis，train数据为正常的数据
"""

learning_rate = 0.00001
batch_size = 50
layer = 1
latent_dim = 10
characters=56
times=10
init_kernel = tf.contrib.layers.xavier_initializer()
attention_size= 16

def encoder(x_inp, is_training=False, getter=None, reuse=False):
    """ Encoder architecture in tensorflow
    Maps the data into the latent space
    Args:
        x_inp (tensor): input data for the encoder. x_inp.shape= [batch_size,characters,times]
        reuse (bool): sharing variables or not
    Returns:
        (tensor): last activation layer of the encoder
    """
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        x_prior = tf.unstack(x_inp, times, 2);  #
        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(25)) for _ in range(2)]);
        with tf.variable_scope("enc"):
            res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32);
            weights = tf.Variable(tf.random_normal([25, 1]));
            biases = tf.Variable(tf.random_normal([1]));
            #print(len(res))
            for i in range(len(res)):
                res[i] = tf.nn.tanh(tf.matmul(res[i], weights) + biases);
            tensor_a = tf.convert_to_tensor(res)
            tensor_b = tf.transpose(tensor_a,perm=[1,2,0])

            name_net = 'enc_1'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(tensor_b,
                                      units=64,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = leakyReLu(net)

            name_net = 'enc_2'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(net,
                                      units=10,
                                      kernel_initializer=init_kernel,
                                      name='fc')

        return net

def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow
        Generates data from the latent space
        Args:
            z_inp (tensor): variable in the latent space  c.shape= [batch_size,1,times] 这里的embedding dim为1
            reuse (bool): sharing variables or not
        Returns:
            (tensor): last activation layer of the generator
     """


    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'dec_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'dec_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=10,
                                  kernel_initializer=init_kernel,
                                  name='fc')
        z_prior = tf.unstack(net, times, 2);
        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(25)) for _ in range(2)]);
        with tf.variable_scope("gen"):
            res, states = tf.contrib.rnn.static_rnn(lstm_cell, z_prior, dtype=tf.float32);
            weights = tf.Variable(tf.random_normal([25, characters]));
            biases = tf.Variable(tf.random_normal([characters]));
            #print(len(res))

            for i in range(len(res)):
                res[i] = tf.nn.tanh(tf.matmul(res[i], weights) + biases);

            tensor_a = tf.convert_to_tensor(res)
            tensor_a = tf.transpose(tensor_a,perm=[1,2,0])
    return tensor_a

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
        x_prior = tf.unstack(x_inp, times, 2);  #
        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(25)) for _ in range(2)]);
        with tf.variable_scope("dis_x"):
            res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32);

            weights = tf.Variable(tf.random_normal([25, 1]));
            biases = tf.Variable(tf.random_normal([1]));
            # print(len(res))

            for i in range(len(res)):
                res[i] = tf.nn.tanh(tf.matmul(res[i], weights) + biases);

            tensor_a = tf.convert_to_tensor(res)
            tensor_a = tf.transpose(tensor_a, perm=[1, 2, 0])

        z_prior = tf.unstack(z_inp, times, 2);  #
        lstm_cell_ = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(25)) for _ in range(2)]);
        with tf.variable_scope("dis_z"):
            res_, states_ = tf.contrib.rnn.static_rnn(lstm_cell_, z_prior, dtype=tf.float32);

            for i in range(len(res)):
                res_[i] = tf.nn.tanh(tf.matmul(res_[i], weights) + biases);

            tensor_a_ = tf.convert_to_tensor(res_)
            tensor_a_ = tf.transpose(tensor_a_, perm=[1, 2, 0])

        final = tf.concat([tensor_a, tensor_a_], axis=2)
        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(final,
                                32,
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

def leakyReLu(x, alpha=0.1, name='leaky_relu'):
    """ Leaky relu """
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

