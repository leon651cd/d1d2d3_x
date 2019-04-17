import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.network7 as network
import train_data_bigan.data4 as data
from sklearn.metrics import precision_recall_fscore_support
"""
加入D2,pca适配一维数据

testing增加attention的对比展示
"""

RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]
def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the KDD dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("BiGAN.train.kdd.{}".format(method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    input_pl_a=tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input_a")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")


    # Data
    trainx, trainy = data.get_train()
    trainx_copy = trainx.copy()

    testx, testy = data.get_test()
    test_label=data.get_test_label()
    test_ip=data.get_test_ip()


    trainx_a,trainy_a=data.get_train_a()
    trainx_a_copy=trainx_a.copy()


    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    batch_size_a=network.batch_size_a

    times=network.times
    characters=network.characters
    latent_dim=network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    print(str(nr_batches_train) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    nr_batches_test = int(testx.shape[0] / batch_size)

    ar_batches_train=int(trainx_a.shape[0]/batch_size_a)
    print(str(ar_batches_train) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    nr_batches_train= min(nr_batches_train,ar_batches_train)

    print(str(nr_batches_train)+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(str(nr_batches_test)+"---------------------------")


    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder
    dis_1 = network.discriminator_1
    dis_2= network.discriminator_2



    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)
        z_gen_a=enc(input_pl_a,is_training=is_training_pl,reuse=True)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, 1,latent_dim])
        x_gen = gen(z, is_training=is_training_pl)

        z_a=tf.random_normal([batch_size_a,1, latent_dim])
        x_gen_a = gen(z_a, is_training=is_training_pl,reuse=True)

    with tf.variable_scope('discriminator_1_model'):
        l_encoder = dis_1(z_gen, input_pl, is_training=is_training_pl)
        l_generator = dis_1(z, x_gen, is_training=is_training_pl, reuse=True)

    with tf.variable_scope('discriminator_2_model'):
        l_encoder_a, _ =dis_2(z_gen_a,input_pl_a,is_training=is_training_pl)
        l_generator_a, _ =dis_2(z_a,x_gen_a,is_training=is_training_pl,reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator_1
        loss_dis_enc = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder), logits=l_encoder))
        loss_dis_gen = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator), logits=l_generator))
        loss_discriminator_1 = loss_dis_gen + loss_dis_enc

        # discriminator_2
        loss_dis_enc = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder_a), logits=l_encoder_a))
        loss_dis_gen = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator_a), logits=l_generator_a))
        loss_discriminator_2 = loss_dis_gen + loss_dis_enc

        # generator
        lg1=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator)
        lg2=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator_a), logits=l_generator_a)
        loss_generator = 0.8*tf.reduce_mean(lg1) - 0.2*tf.reduce_mean(lg2)

        # encoder
        le1=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder)
        le2=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder_a), logits=l_encoder_a)
        loss_encoder = 0.8*tf.reduce_mean(le1) - 0.2*tf.reduce_mean(le2)


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_1_model' in var.name]
        dvars_a = [var for var in tvars if 'discriminator_2_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_1_model' in x.name)]
        update_ops_dis_a = [x for x in update_ops if ('discriminator_2_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_dis_a = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer_a')  #调成0.9结果并没有变化
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')#不需要手动调优学习速率α，抗噪声能力强
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator_1, var_list=dvars)
        with tf.control_dependencies(update_ops_dis_a):
            dis_op_a = optimizer_dis_a.minimize(loss_discriminator_2, var_list=dvars_a)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)
        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        dis_ema_a = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis_a = dis_ema_a.apply(dvars_a)
        with tf.control_dependencies([dis_op_a]):
            train_dis_op_a = tf.group(maintain_averages_op_dis_a)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)
        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)
        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator1', loss_discriminator_1, ['dis'])
            tf.summary.scalar('loss_discriminator2', loss_discriminator_2, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')


    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_1_model'):
        l_encoder_ema = dis_1(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema  = dis_1(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)

    with tf.variable_scope('discriminator_2_model'):
        l_encoder_ema_a,alphas = dis_2(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema_a),
                                                 reuse=True)
        l_generator_ema_a, _  = dis_2(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema_a),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            # delta =tf.reshape(delta,[-1,1,delta.shape[1]])
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                                keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_1_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(l_encoder_ema), logits=l_encoder_ema)

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Discriminator_2_loss'):
            if method == "cross-e":
                dis_score_a = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(l_encoder_ema_a), logits=l_encoder_ema_a)

            dis_score_a = tf.squeeze(dis_score_a)

        with tf.variable_scope('Score'):
            list_scores = 0.6*gen_score + 0.1* dis_score+ 0.3*dis_score_a           #0.4 0.1 0.5调整后差别不大

    logdir = create_logdir(weight, method, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)



    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        train_dis_only=1
        recall_max=0.
        precision_max=0.
        f1_max=0.
        max_epoch=0




        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling ccset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            trainx_a = trainx_a[rng.permutation(trainx_a.shape[0])]  # shuffling dataset
            trainx_a_copy = trainx_a_copy[rng.permutation(trainx_a.shape[0])]

            train_loss_dis1, train_loss_dis2,train_loss_gen, train_loss_enc = [0, 0, 0, 0]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                ran_from_a= t * batch_size_a
                ran_to_a= (t+1) *batch_size_a

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             input_pl_a:trainx_a[ran_from_a:ran_to_a],
                             is_training_pl: True,
                             learning_rate: lr}

                _,_, ld1,ld2, sm = sess.run([train_dis_op,
                                            train_dis_op_a,
                                            loss_discriminator_1,
                                            loss_discriminator_2,
                                            sum_op_dis],
                                            feed_dict=feed_dict)
                train_loss_dis1 += ld1
                train_loss_dis2 += ld2
                writer.add_summary(sm, train_batch)

                #训练gen需要满足下列条件
                if(train_dis_only>20 and train_dis_only%100!=0):
                    # train generator and encoder
                    feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                                 input_pl_a: trainx_a_copy[ran_from_a:ran_to_a],
                                 is_training_pl: True,
                                 learning_rate: lr}
                    tg, te, le, lg, sm = sess.run([train_gen_op,
                                                 train_enc_op,
                                                 loss_encoder,
                                                 loss_generator,
                                                 sum_op_gen],
                                                feed_dict=feed_dict)
                    train_loss_gen += lg
                    train_loss_enc += le
                    writer.add_summary(sm, train_batch)


                train_batch += 1
                train_dis_only+=1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis1 /= nr_batches_train
            train_loss_dis2/=nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis1 = %.4f | loss dis2 = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis1,train_loss_dis2))

            #testing
            if((epoch+1)%1==0):
                logger.warn('Testing evaluation...')

                inds = rng.permutation(testx.shape[0])
                testx = testx[inds]  # shuffling  dataset
                testy = testy[inds]  # shuffling  dataset
                test_ip=test_ip[inds]
                test_label=test_label[inds]
                scores = []
                inference_time = []
                labels = np.ones(shape=[testx.shape[0],characters])

                # Create scores
                for t in range(nr_batches_test):
                    # construct randomly permuted minibatches
                    ran_from = t * batch_size
                    ran_to = (t + 1) * batch_size
                    begin_val_batch = time.time()

                    feed_dict = {input_pl: testx[ran_from:ran_to],
                                 is_training_pl: False}

                    scores_add,labels_add= sess.run([list_scores,alphas],
                                       feed_dict=feed_dict)
                    scores_add=scores_add.tolist()
                    scores+=scores_add

                    labels[t*batch_size:(t+1)*batch_size] = labels_add

                    inference_time.append(time.time() - begin_val_batch)

                logger.info('Testing : mean inference time is %.4f' % (
                    np.mean(inference_time)))

                ran_from = nr_batches_test * batch_size
                ran_to = (nr_batches_test + 1) * batch_size
                size = testx[ran_from:ran_to].shape[0]
                fill = np.ones([batch_size - size, 1,characters])

                batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
                feed_dict = {input_pl: batch,
                             is_training_pl: False}

                batch_score,batch_labels = sess.run([list_scores,alphas],
                                       feed_dict=feed_dict)
                batch_score=batch_score.tolist()

                scores += batch_score[:size]

                labels[nr_batches_test*batch_size:]=batch_labels[:size]

                per = np.percentile(scores, 24)   #前百分之？的数据当做正常的    the best:24

                y_pred = scores.copy()
                y_pred = np.array(y_pred)

                inds = (y_pred < per)
                inds_comp = (y_pred >= per)

                y_pred[inds] = 0.
                y_pred[inds_comp] = 1.


                precision, recall, f1, _ = precision_recall_fscore_support(testy,
                                                                           y_pred,
                                                                           average='binary')
                if(f1>f1_max):
                    precision_max = precision
                    recall_max = recall
                    f1_max = f1
                    max_epoch=epoch

                    max_testx= testx
                    max_test_label=test_label
                    max_test_ip=test_ip
                    max_testy=testy
                    max_y_pred=y_pred
                    max_labels=labels



                """
                TP = tf.count_nonzero(y_pred * testy)
                TN = tf.count_nonzero((y_pred - 1) * (testy - 1))
                FP = tf.count_nonzero(y_pred * (testy - 1))
                FN = tf.count_nonzero((y_pred - 1) * testy)

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                """
                print(
                    "Testing : Precision = %.6f | Recall = %.6f | F1 = %.6f "
                    % (precision, recall, f1))

                # f=open("bigan_result",'a')
                # for i in scores:
                #     f.write(str(i)+" ")
                # f.write("\n")

            epoch += 1
        print("***********************************************************")
        print("max epoch: " + str(max_epoch))
        print(
            "max: = %.6f | Recall = %.6f | F1 = %.6f "
            % (precision_max, recall_max, f1_max))

        print("max labels: ")
        print(max_labels)
        np.savetxt("max_labels",max_labels, fmt='%.4f')

        print("max_y_pred: ")
        print(max_y_pred)
        np.savetxt("max_y_pred", max_y_pred,fmt='%d')

        print("max_testy: ")
        print(max_testy)
        np.savetxt("max_testy", max_testy,fmt='%d')

        print("max_test_ip: ")
        print(max_test_ip)
        np.savetxt("max_test_ip", max_test_ip,fmt ='%s')

        np.savetxt("max_test_label",max_test_label,fmt='%.4f')

        #np.savetxt("max_testx",max_testx.astype(float))




def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed)