import os
import numpy as np
import tensorflow as tf
from PIL import Image
import random
# import theano.tensor as T
from lifelines.utils import concordance_index
import pandas as pd
import datetime
# import keras.backend as K
from tensorflow.contrib.layers import l2_regularizer


# load data
def load_data(path,cluster_num):

    train_path = path + cluster_num+'_train.csv'
    test_path = path + cluster_num+'_test.csv'

    train_list = pd.read_csv(train_path,header=None)
    train_list = np.array(train_list)
    test_list = pd.read_csv(test_path,header=None)
    test_list = np.array(test_list)

    train_data_list = train_list[:,0]
    train_time_list = train_list[:,1]
    train_event_list = train_list[:,2]

    test_data_list = test_list[:, 0]
    test_time_list = test_list[:, 1]
    test_event_list = test_list[:, 2]

    return train_data_list, train_time_list, train_event_list, test_data_list, test_time_list, test_event_list


# obtain batch
def next_batch(xs, es, batch_size, start, size):
    start = start % round(size / batch_size)
    start = start * batch_size
    end = start + batch_size
    batch_xs_list = xs[start:end]
    batch_es_list = es[start:end]
    batch_xs = []
    for i in range(len(batch_xs_list)):
        image = Image.open(batch_xs_list[i])
        image_narray = np.asarray(image, dtype='float32')
        image_narray = image_narray * 1.0 / 255
        batch_xs.append(image_narray)
    batch_xs = np.stack(batch_xs)
    batch_es = np.reshape(np.array(batch_es_list),(batch_size,1))
    return batch_xs, batch_es



# obtain batch which is sorted by survival time from smallest to largest
def next_batch_order(xs, ts ,es, batch_size, start, size):

    start = start % round(size / batch_size)
    start = start * batch_size
    end = start + batch_size
    batch_xs_list = xs[start:end]
    batch_ts_list = ts[start:end]
    batch_es_list = es[start:end]

    batch_xs = []
    for i in range(len(batch_xs_list)):
        path = batch_xs_list[i]
        image = Image.open(path)
        image_array = np.array(image, dtype='float32')
        image_array = image_array * 1.0 / 255
        batch_xs.append(image_array)

    # batch_xs = np.stack(batch_xs)
    # batch_es = np.reshape(np.array(batch_es_list), (batch_size, 1))

    # sort
    batch_xs_or = np.stack(batch_xs)
    batch_ts_or = np.reshape(np.array(batch_ts_list), (batch_size, 1))
    batch_es_or = np.reshape(np.array(batch_es_list),(batch_size,1))

    index = np.argsort(batch_ts_or,axis=0)
    index = np.reshape(index,(1,batch_size))

    batch_xs = batch_xs_or[index]
    batch_xs = np.reshape(batch_xs,(batch_size,512,512,3))
    # batch_ts = batch_ts_or[index]
    # batch_ts = np.reshape(batch_ts, (batch_size,1))
    batch_es = batch_es_or[index]
    batch_es = np.reshape(batch_es, (batch_size, 1))
    # print(batch_ts)

    return batch_xs, batch_es



def cbam(inputs, reduction_ratio, name=''):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        hidden_num = inputs.get_shape().as_list()[3]


        # channel attention
        maxpool_channel = tf.reduce_max(inputs,axis=[1,2], keepdims=True)
        avgpool_channel = tf.reduce_mean(inputs,axis=[1,2], keepdims=True)

        mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units= hidden_num //reduction_ratio, name="mlp_1",
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), activation=tf.nn.relu, reuse=None)
        mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    reuse=None)

        mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units= hidden_num //reduction_ratio, name="mlp_1",
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  activation=tf.nn.relu, reuse=True)
        mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2",
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), reuse=True)

        channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
        channel_refined_feature = inputs * channel_attention



        #spatial attention
        maxpool_spatial = tf.reduce_max(channel_refined_feature, axis=3, keepdims=True)
        avgpool_spatial = tf.reduce_mean(channel_refined_feature, axis=3, keepdims=True)
        max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
        conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), activation=None,name='spatial_conv')
        spatial_attention = tf.nn.sigmoid(conv_layer)

        refined_feature = channel_refined_feature * spatial_attention

        return refined_feature


# DCAS
def DCAS(input,drop,training):

    # bn0 = tf.layers.batch_normalization(input, axis=3, name='bn0', training=training)

    conv1_1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=3, strides=4, kernel_initializer=tf.variance_scaling_initializer(), name='conv1_1', padding='same')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, kernel_size=3, strides=1, kernel_initializer=tf.variance_scaling_initializer(), name='conv1_2', padding='same')
    conv1_3 = tf.layers.conv2d(inputs=conv1_2, filters=32, kernel_size=3, strides=1, kernel_initializer=tf.variance_scaling_initializer(), name='conv1_3', padding='same')
    cbam_1 = cbam(conv1_3, redu_ratio ,'cbam_1')
    bn1 = tf.layers.batch_normalization(cbam_1, axis=3, name='bn1', training=training)
    ac1 = tf.nn.relu(bn1, name='ac1')

    pool1 = tf.layers.max_pooling2d(inputs=ac1, pool_size=(2,2), strides=2, name='pool1')


    # conv1_1_1 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=3, strides=2, kernel_initializer=tf.variance_scaling_initializer(), name='conv1_1_1')
    # conv1_2_1 = tf.layers.conv2d(inputs=conv1_1_1, filters=32, kernel_size=3, strides=1, kernel_initializer=tf.variance_scaling_initializer(), name='conv1_2_1')
    # bn1_1 = tf.layers.batch_normalization(conv1_2_1, axis=3, name='bn1_1', training=training)
    # ac1_1 = tf.nn.relu(bn1_1, name='ac1_1')
    #
    # pool1_1 = tf.layers.max_pooling2d(inputs=ac1_1, pool_size=(2, 2), strides=2, name='pool1')

    conv2_1 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=3, strides=2, kernel_initializer=tf.variance_scaling_initializer(), name='conv2_1', padding='same')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=32, kernel_size=3, strides=1, kernel_initializer=tf.variance_scaling_initializer(), name='conv2_2', padding='same')
    cbam_2 = cbam(conv2_2, redu_ratio, 'cbam_2')
    # bn2 = tf.layers.batch_normalization(conv2_2, axis=3, name='bn2', training=training)
    ac2 = tf.nn.relu(cbam_2, name='ac2')


    conv3 = tf.layers.conv2d(inputs=ac2, filters=32, kernel_size=3, strides=2, kernel_initializer=tf.variance_scaling_initializer(), name='conv3', padding='same')
    cbam_3 = cbam(conv3, redu_ratio, 'cbam_3')
    bn3 = tf.layers.batch_normalization(cbam_3, axis=3, name='bn3', training=training)
    ac3 = tf.nn.relu(bn3, name='ac3')

    pool2 = tf.layers.max_pooling2d(inputs=ac3, pool_size=(2,2), strides=2, name='pool2')
    # print(pool2.get_shape())

    flatten = tf.layers.flatten(pool2, name='flatten')
    mlp_flatten = tf.layers.dense(inputs=flatten, units=flatten.get_shape().as_list()[1]/32, kernel_initializer=tf.variance_scaling_initializer(), name='mlp_flatten')
    new_flatten = tf.layers.dense(inputs=mlp_flatten, units=flatten.get_shape().as_list()[1], kernel_initializer=tf.variance_scaling_initializer(), name='new_flatten')
    att = tf.nn.sigmoid(new_flatten)
    # print(att.get_shape())
    att_f = flatten*att
    # print(att_f.get_shape())


    # drop = tf.layers.dropout(inputs=flatten, rate=drop, name='drop')
    fc1 = tf.layers.dense(inputs=att_f, units=32, kernel_initializer=tf.variance_scaling_initializer(), name='fc1')
    prediction = tf.layers.dense(inputs=fc1, units=1, name='out')

    return fc1, prediction



def calc_cost(cost_value, pred, event, batch_size):

    for i in range(batch_size):
        cost_value += event[i,0]*(pred[i,0]-tf.log(tf.reduce_sum(tf.exp(pred[i:,0]))))

    return -cost_value/batch_size



# train
def train(train_data_list, train_time_list, train_event_list, cluster_num, batch_size, drop_value, calc_ci, num_i):

    now_time = str(datetime.datetime.now())
    date = now_time.split(' ')[0]
    clork = now_time.split(' ')[-1]
    hour = clork.split(':')[0]
    min = clork.split(':')[1]

    model_path_last = './model_saving/' + str(cluster_num) + '/' + date + '_' + hour + '.' + min

    if not os.path.exists(model_path_last):
        os.makedirs(model_path_last)
    model_path = model_path_last + '/weights/ov_surv'
    log_path = model_path_last+'/log/'

    xs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    es = tf.placeholder(tf.float32, [None, 1])
    drop = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    global_ = tf.placeholder(tf.int64)

    fc_out, prediction = DCAS(xs, drop, training)

    loss = -tf.reduce_mean((prediction - tf.log(tf.reverse(tf.cumsum(tf.exp(tf.reverse(prediction,[0]))),[0]))) * es)
    tf.summary.scalar('loss',loss)
    learning_rate = tf.train.exponential_decay(learning_rate=lr_value, global_step = global_, decay_steps=20, decay_rate=0.5, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver(max_to_keep=epochs)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        for it in range(epochs):

            c = list(zip(train_data_list, train_time_list, train_event_list))
            random.Random(random.randint(0, 10000)).shuffle(c)
            train_data_list, train_time_list, train_event_list = zip(*c)

            for i in range(int(train_size/ batch_size)):
                batch_xs, batch_es = next_batch_order(train_data_list, train_time_list, train_event_list, batch_size, i, train_size)
                train_step.run(
                    feed_dict={global_:it, xs: batch_xs, es:batch_es, drop: drop_value, training: True})

                if int(train_size/ batch_size)-i == 1:
                    loss_value = sess.run(loss, feed_dict={global_:it, xs: batch_xs, es:batch_es, drop:0.0, training: False})
                    merged_result = sess.run(merged,feed_dict={global_:it, xs: batch_xs, es:batch_es, drop:0.0, training: False})
                    writer.add_summary(merged_result,it)
                    now_time = str(datetime.datetime.now())
                    print(now_time, ' epoch_%s' % (it + 1),
                          '  step %d, training loss %g' % (i + 1, loss_value))

            saver.save(sess, model_path, global_step=it+1)
            if ((it+1)%10)==0:
                if calc_ci:
                    predictions = []
                    if train_size % train_batch_size == 0:
                        for j in range(int(train_size / train_batch_size)):
                            # train_batch_xs, train_batch_es = next_batch_order(train_data_list, train_time_list, train_event_list, train_batch_size, j,train_size)
                            train_batch_xs, train_batch_es = next_batch(train_data_list, train_event_list, train_batch_size, j,train_size)
                            pred = sess.run(prediction,
                                            feed_dict={xs: train_batch_xs, es: train_batch_es, drop:0.0, training:False})
                            predictions.append(np.exp(pred))
                        preds = np.array(predictions)
                        preds = preds.reshape((train_size, 1))
                        train_time = np.array(train_time_list)
                        train_event = np.array(train_event_list)
                        ci = concordance_index(train_time, -preds, train_event)
                        print('epoch',it+1,'ci:',ci)
                    else:
                        print('Wrong train batch size input!')


#evaluate
def evaluate(restore_model_path,train_data_list, train_time_list, train_event_list, test_data_list, test_time_list, test_event_list, istrain):
    train_preds = []
    test_preds = []

    # restore_model_path = './model_saving/Vgg_softmax/2018-08-26_10.36/osteoporosis_classifier'
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 512, 512, 3])
    y_ = tf.placeholder(tf.float32, [None, 1])

    fc1, prediction = DCAS(x,0.0,False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Start to restore model')
        saver.restore(sess,restore_model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path
        if istrain:

            print('Start to evaluate train data')
            if train_size % train_batch_size == 0:
                for i in range(int(train_size / train_batch_size)):
                    train_batch_xs, train_batch_es = next_batch(train_data_list, train_event_list, train_batch_size, i, train_size)
                    pred = sess.run(prediction,
                                         feed_dict={x: train_batch_xs, y_: train_batch_es})
                    train_preds.append(np.exp(pred))
            else:
                print('wrong train batch size num')
            train_preds = np.array(train_preds)
            train_preds = train_preds.reshape((train_size,1))
            train_time = np.array(train_time_list)
            train_event = np.array(train_event_list)
            train_ci = concordance_index(train_time,-train_preds,train_event)
            print('train_ci',train_ci)

        print('Start to evaluate test data')
        if test_size % test_batch_size == 0:
            for i in range(int(test_size / test_batch_size)):
                test_batch_xs, test_batch_es = next_batch(test_data_list, test_event_list, test_batch_size, i, test_size)
                pred = sess.run(prediction,
                                     feed_dict={x: test_batch_xs, y_: test_batch_es})
                test_preds.append(np.exp(pred))
        else:
            print('wrong train batch size num')
        test_preds = np.array(test_preds)
        test_preds = test_preds.reshape((test_size,1))
        test_time = np.array(test_time_list)
        test_event = np.array(test_event_list)
        test_ci = concordance_index(test_time,-test_preds,test_event)
        print('test_ci',test_ci)
    sess.close()


def load_all_data(path):

    data = pd.read_csv(path,header=None)
    data = np.array(data)
    all_data = data[:,0]
    all_time = data[:,1]
    all_event = data[:,2]
    all_name = []
    for i in range(len(all_data)):
        name = ((all_data[i]).split('/')[-1]).split('_')[0]
        all_name.append(name)
    all_name = np.array(all_name)
    all_name = np.reshape(all_name,(len(all_name),1))

    return all_data, all_time, all_event, all_name


def calc_batch_size(num):
    out = 0

    list = []
    for i in range(1, num):
        if num % i == 0:
            list.append(i)
    number = []
    b = 150

    while len(number) == 0:
        number = [x for x in list if x in range(b, 200)]
        if len(number) == 0:
            b = int(b / 2)
        else:
            out = number[0]
            break
    return out


def extract_features(cluster_nums):

    for cluster_num in cluster_nums:

        csv_path = './csv/label/'+str(cluster_num)+'.csv'
        all_data, all_time, all_event, all_name = load_all_data(csv_path)
        all_size = len(all_data)
        all_batch_size = calc_batch_size(all_size)
        print('Cluster '+str(cluster_num)+',  all_size ',all_size,'  all_batch_size ',all_batch_size)
        model_path = './model_saving/'+str(cluster_num)+'/'+str([time_path for time_path in os.listdir('./model_saving/'+str(cluster_num))][-1])+'/weights/ov_surv-200'

        all_preds = []
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 512, 512, 3])
        y_ = tf.placeholder(tf.float32, [None, 1])

        fc1, prediction = cnn(x, 0.0, False)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Start to restore model cluster',str(cluster_num)+':')
            saver.restore(sess, model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

            print('    extract features.........')
            if all_size % all_batch_size == 0:
                for i in range(int(all_size / all_batch_size)):
                    all_batch_xs, all_batch_es = next_batch(all_data, all_event, all_batch_size, i, all_size)
                    pred = sess.run(fc1, feed_dict={x: all_batch_xs, y_: all_batch_es})
                    all_preds.append(np.exp(pred))
                all_preds = np.array(all_preds)
                all_preds = np.reshape(all_preds,(all_size,32))
                all = np.concatenate((all_name,all_preds),axis=1)
                all = pd.DataFrame(all)
                all.to_csv('./features/'+str(cluster_num)+'_feature.csv',header=None,index=False)
                print('    Finish!')

            else:
                print('wrong train batch size num')
            sess.close()


def calc_show_i(train_size,batch_size):
    i_all= int(train_size/batch_size)
    b = 100
    while i_all<b:
        b = int(b/2)
    return b



#main
def main(_):

    global image_size
    global train_size
    global test_size
    global lr_value
    global epochs
    global train_batch_size
    global test_batch_size
    global l2_value
    global batch_size
    global redu_ratio


    image_size = 512
    cluster_num = 9
    epochs= 150
    batch_size= 256
    drop_value = 0.0
    lr_value= 0.1
    l2_value = 1.0
    redu_ratio = 4

    train_data_list, train_time_list, train_event_list, test_data_list, test_time_list,test_event_list=load_data('./csv/split_sort_label/'+str(cluster_num)+'/',str(cluster_num))

    train_size = len(train_data_list)
    print('train_size:', train_size)
    train_batch_size = calc_batch_size(train_size)
    print('train_batch_size:', train_batch_size)
    num_i = calc_show_i(train_size, batch_size)
    print('num_i:', num_i)
    test_size = len(test_data_list)
    print('test_size:', test_size)
    test_batch_size = calc_batch_size(test_size)
    print('test_batch_size:', test_batch_size)


    # train(train_data_list, train_time_list, train_event_list, cluster_num, batch_size, drop_value, True, num_i)
    evaluate('./model_saving/' + str(cluster_num) + '/'+'2019-10-25_00.20/weights/ov_surv-150', train_data_list, train_time_list, train_event_list, test_data_list, test_time_list, test_event_list, False)
    # extract_features([0,1,2,3,7])
if __name__ == '__main__':
    tf.app.run(main=main)
