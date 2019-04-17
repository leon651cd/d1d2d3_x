import tensorflow as tf
"""
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

def attention(inputs, attention_size, time_major=False):
    """
    inputs.shape = batch,times,hidden_n
    """
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

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
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    #只返回权值
    return alphas

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
    return tensor_b


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
        z_prior = tf.unstack(z_inp, times, 2);
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

            tensor_a = tf.convert_to_tensor(res)
            tensor_b = tf.transpose(tensor_a, perm=[1, 2, 0])   #tensor b.shape =batch, hidden_n, times
            tensor_b = leakyReLu(tensor_b)
            tensor_c = tf.transpose(tensor_b, perm=[0, 2, 1])
            alphas= attention(tensor_c,attention_size)
            tensor_b_mean=tf.reduce_mean(tensor_b,1) #batch,times
            #tem_res=tf.diag_part(tf.matmul(tensor_b_mean,tf.transpose(alphas))) #shape = batch,1
            tem_res= tensor_b_mean * alphas

        z_prior = tf.unstack(z_inp, times, 2);  #
        lstm_cell_ = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(25)) for _ in range(2)]);
        with tf.variable_scope("dis_z"):
            res_, states_ = tf.contrib.rnn.static_rnn(lstm_cell_, z_prior, dtype=tf.float32);

            tensor_a_ = tf.convert_to_tensor(res_)
            tensor_b_ = tf.transpose(tensor_a_, perm=[1, 2, 0])   #tensor b.shape =batch, hidden_n, times
            tensor_b_ = leakyReLu(tensor_b_)
            tensor_c_ = tf.transpose(tensor_b_, perm=[0, 2, 1])
            alphas_= attention(tensor_c_,attention_size)
            tensor_b_mean_=tf.reduce_mean(tensor_b_,1) #batch,times
            #tem_res_=tf.diag_part(tf.matmul(tensor_b_mean_,tf.transpose(alphas_))) #shape = batch,1
            tem_res_ =tensor_b_mean_*alphas_

        final = tf.concat([tem_res,tem_res_],axis=1)
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

            #测试不加attention


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

def test():
    z = tf.constant(5., shape=[13, 1, 10])
    x = tf.constant(4., shape=[13, 56, 10])
    output=discriminator_1(z,x)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print(output.shape)
    print(sess.run(output))




