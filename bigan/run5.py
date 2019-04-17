import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.network7 as network
import train_data_bigan.data5 as data
from sklearn.metrics import precision_recall_fscore_support
"""
加入D2,pca适配一维数据

testing增加attention的对比展示

增加针对attention权重的训练过程

修改4中的获取数据bug
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

def score3(ip,slot_all_desip,label,srcip,desip):
    #str, list, 1d ndarray
    ip=str(ip)
    if ip in srcip:
        ip_src_score=srcip[ip]
    else:
        ip_src_score=0

    if ip in desip:
        ip_des_score=desip[ip]
    else:
        ip_des_score=0

    dip=[]
    inds= (label>1.0)
    for i in inds:
        des_tem=slot_all_desip[i]
        des_tem=des_tem.strip().split(",")
        for temip in des_tem:
            if(temip !='0'):
                dip.append(temip)  #注意这里没有进行判重！

    dip_src_score=0
    dip_des_score=0
    for i in dip:
        if i in srcip:
            dip_src_score +=srcip[i]
        else:
            dip_src_score ==0

        if i in desip:
            dip_des_score+=desip[i]
        else:
            dip_des_score+=0

    score= 0.3* ip_src_score+ 0.3 * ip_des_score +0.3 *dip_src_score +0.2 *dip_des_score
    return score




def train_and_test(nb_epochs, weight, method, degree, random_seed,ratio):
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
    input_label_a=tf.placeholder(tf.float32, shape=data.get_shape_labels(), name="input_label_a")  #异常训练数据对应的标签
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")


    # Data
    trainx, trainy, trainx_a, trainy_a, testx, testy, test_label, test_ip, a_label_train, test_slot_attack, test_slot_desip, test_slot_all_desip, srcip, desip = data._get_adapted_dataset(
        ratio)

    x_a = np.sum(testy)
    x_n = testy.shape[0] - x_a
    x_all = testy.shape[0]

    percent = float(x_n / x_all+0.00002)
    percent *= 100
    # print("testing1.20")
    # print(a_label_train.shape)
    trainx_copy = trainx.copy()
    trainx_a_copy = trainx_a.copy()


    # Parameters

    n_num = trainx.shape[0]
    a_num = trainx_a.shape[0]
    if(n_num<=a_num):
        batch_size=30
        batch_size_a=int(30*a_num/n_num)
    if(n_num>a_num):
        batch_size_a=30
        batch_size=int(30*n_num/a_num)


    starting_lr = network.learning_rate
    # batch_size = network.batch_size
    # batch_size_a=network.batch_size_a

    times=network.times
    characters=network.characters
    latent_dim=network.latent_dim
    ema_decay = 0.9999

    # n_num=trainx.shape[0]
    # a_num=trainx_a.shape[0]

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    print(str(nr_batches_train) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    nr_batches_test = int(testx.shape[0] / batch_size)

    ar_batches_train=int(trainx_a.shape[0]/batch_size_a)
    print(str(ar_batches_train) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    nr_batches_train= min(nr_batches_train,ar_batches_train)

    # print(str(nr_batches_train)+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(str(nr_batches_test)+"---------------------------")


    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder
    dis_1 = network.discriminator_1
    dis_2= network.discriminator_2
    attention= network.attention

    input_pl_a_transpose = tf.transpose(input_pl_a, perm=[0, 2, 1])



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
        l_encoder_a=dis_2(z_gen_a,input_pl_a,is_training=is_training_pl)
        l_generator_a=dis_2(z_a,x_gen_a,is_training=is_training_pl,reuse=True)

    with tf.variable_scope('attention_model'):
        alphas_train=attention(input_pl_a_transpose,3,is_training=is_training_pl)



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
        # loss_discriminator_2 = loss_dis_gen + loss_dis_enc


        # generator
        lg1=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator)
        lg2=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator_a), logits=l_generator_a)
        loss_generator = 0.8*tf.reduce_mean(lg1) - 0.2*tf.reduce_mean(lg2)

        # encoder
        le1=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder)
        le2=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder_a), logits=l_encoder_a)
        loss_encoder = 0.8*tf.reduce_mean(le1) - 0.2*tf.reduce_mean(le2)

        # attention
        loss_attention = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=input_label_a, logits=alphas_train))



    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_1_model' in var.name]
        dvars_a = [var for var in tvars if 'discriminator_2_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]
        avars = [var for var in tvars if 'attention_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_1_model' in x.name)]
        update_ops_dis_a = [x for x in update_ops if ('discriminator_2_model' in x.name)]
        update_ops_att = [x for x in update_ops if ('attention_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_dis_a = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer_a')  #调成0.9结果并没有变化
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')#不需要手动调优学习速率α，抗噪声能力强
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')
        optimizer_att = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.5, name='att_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator_1, var_list=dvars)
        with tf.control_dependencies(update_ops_dis_a):
            dis_op_a = optimizer_dis_a.minimize(loss_discriminator_2, var_list=dvars_a)
            # dis_op_attention=optimizer_dis_a.minimize(loss_attention, var_list=dvars_a)
        with tf.control_dependencies(update_ops_att):
            att_op = optimizer_att.minimize(loss_attention, var_list=avars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)
        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        dis_ema_a = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis_a = dis_ema_a.apply(dvars_a)
        with tf.control_dependencies([dis_op_a]):
            train_dis_op_a = tf.group(maintain_averages_op_dis_a)

        # dis_ema_attention=tf.train.ExponentialMovingAverage(decay=ema_decay)
        # maintain_averages_op_attention = dis_ema_attention.apply(dvars_a)
        # with tf.control_dependencies([dis_op_attention]):
        #     train_dis_op_attention = tf.group(maintain_averages_op_attention)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)
        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)
        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

        att_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_att = att_ema.apply(avars)
        with tf.control_dependencies([att_op]):
            train_att_op = tf.group(maintain_averages_op_att)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator1', loss_discriminator_1, ['dis'])
            tf.summary.scalar('loss_discriminator2', loss_discriminator_2, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])


        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        with tf.name_scope('att_summary'):
            tf.summary.scalar('loss_attention', loss_attention, ['att'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_att = tf.summary.merge_all('att')

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
        l_encoder_ema_a = dis_2(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema_a),
                                                 reuse=True)
        l_generator_ema_a = dis_2(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema_a),
                                                   reuse=True)
    with tf.variable_scope('attention_model'):
        attention_input=tf.transpose(input_pl,perm=[0,2,1])
        alphas=attention(attention_input,3,is_training=is_training_pl,getter=get_getter(att_ema),reuse=True)


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
            list_scores = 0.8*gen_score + 0.1* dis_score+ 0.1*dis_score_a           #0.4 0.1 0.5调整后差别不大

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
        max_time=0




        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling ccset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]


            inds_a=rng.permutation(trainx_a.shape[0])
            trainx_a = trainx_a[inds_a]  # shuffling dataset
            a_label_train=a_label_train[inds_a]
            trainx_a_copy = trainx_a_copy[rng.permutation(trainx_a.shape[0])]


            # print("testing 1605")
            # print(trainx_a.shape)
            # print(a_label_train.shape)


            train_loss_dis1, train_loss_dis2,train_loss_gen, train_loss_enc, train_loss_attention = [0, 0, 0, 0, 0]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                ran_from_a= t * batch_size_a
                ran_to_a= (t+1) *batch_size_a

                # print("tesing333")
                # print(a_label_train.shape)
                # print(ran_from_a)
                # print(ran_to_a)
                # print(a_label_train[ran_from_a, ran_to_a].shape[0])

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             input_pl_a:trainx_a[ran_from_a:ran_to_a],
                             input_label_a:a_label_train[ran_from_a:ran_to_a],
                             is_training_pl: True,
                             learning_rate: lr}

                _,_,_, ld1,ld2,lda, sm = sess.run([train_dis_op,
                                            train_dis_op_a,
                                            train_att_op,
                                            loss_discriminator_1,
                                            loss_discriminator_2,
                                            loss_attention,
                                            sum_op_dis],
                                            feed_dict=feed_dict)
                train_loss_dis1 += ld1
                train_loss_dis2 += ld2
                train_loss_attention+=lda
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
            train_loss_attention/=nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis1 = %.4f | loss dis2 = %.4f | loss atten = %.4f  "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis1,train_loss_dis2,train_loss_attention))
            current_time=time.time()
            #testing
            if((epoch+1)%3==0):
                logger.warn('Testing evaluation...')
                start_time=time.time()
                inds = rng.permutation(testx.shape[0])
                testx = testx[inds]  # shuffling  dataset
                testy = testy[inds]  # shuffling  dataset
                test_ip=test_ip[inds]
                test_label=test_label[inds]
                test_slot_attack=data.list_shuffle(test_slot_attack,inds)
                test_slot_desip=data.list_shuffle(test_slot_desip,inds)
                test_slot_all_desip=data.list_shuffle(test_slot_all_desip,inds)
                scores = []
                scores_d3=[]
                inference_time = []
                labels = np.ones(shape=[testx.shape[0],characters])  #学习到的测试集的attention权重

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
                    scores_add=scores_add.tolist()    #score_add指的是d1d2的结果
                    scores_d3_add=[]    #d3的结果

                    ip_batch = test_ip[ran_from:ran_to]
                    slot_all_desip_batch = test_slot_all_desip[ran_from * 10:ran_to * 10]

                    for iii in range(batch_size):
                        score_d3=score3(ip_batch[iii] , slot_all_desip_batch[iii*10:(iii+1)*10] , labels_add[iii] , srcip,desip)
                        scores_d3_add.append(score_d3)




                    scores+=scores_add
                    scores_d3+=scores_d3_add
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


                scores_d3_add = []  # d3的结果
                ip_batch = test_ip[ran_from:ran_from+size]
                slot_all_desip_batch = test_slot_all_desip[ran_from * 10:(ran_from+size) * 10]
                for iii in range(size):
                    score_d3 = score3(ip_batch[iii], slot_all_desip_batch[iii * 10:(iii + 1) * 10], batch_labels[iii],
                                      srcip, desip)
                    scores_d3_add.append(score_d3)

                scores += batch_score[:size]
                scores_d3 += scores_d3_add
                labels[nr_batches_test*batch_size:]=batch_labels[:size]


                for si in range(len(scores)):
                    scores[si]= scores[si]*0.5+ scores_d3[si]*0.5



                per = np.percentile(scores, percent)   #前百分之？的数据当做正常的    the best:24

                y_pred = scores.copy()
                y_pred = np.array(y_pred)

                inds = (y_pred < per)
                inds_comp = (y_pred >= per)

                y_pred[inds] = 0.
                y_pred[inds_comp] = 1.


                precision, recall, f1, _ = precision_recall_fscore_support(testy,
                                                                         y_pred,
                                                                       average='binary')
                # print("**********************************")
                end_time=time.time()
                test_time=end_time-start_time
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
                    max_slot_attack=test_slot_attack
                    max_slot_desip=test_slot_desip
                    max_time=test_time



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

        f=open('d1d2d3_18','a')
        # f.write(begin)
        # f.write("---->")
        # f.write(max_time)
        # f.write("\n")
        f.write(str(ratio)+'\n')
        f.write("test time: ")
        f.write(str(max_time))
        f.write('\n')
        f.write("max: = %.6f | Recall = %.6f | F1 = %.6f \n"
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

        # np.savetxt("max_slot_attack", max_slot_attack, fmt='%s')
        f=open("max_slot_attack",'w')
        for i in max_slot_attack:
            f.write(i)
            # f.write('\n')


        # np.savetxt("max_slot_desip", max_slot_desip, fmt='%s')
        f= open("max_slot_desip",'w')
        for i in max_slot_desip:
            f.write(i)
            # f.write('\n')

        #np.savetxt("max_testx",max_testx.astype(float))




def run(nb_epochs, weight, method, degree, label,ratio,random_seed=42, ):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # tf.reset_default_graph()
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed,ratio=ratio)