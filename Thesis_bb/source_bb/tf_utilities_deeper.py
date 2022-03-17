import tensorflow as tf
from source_bb.pc_utilities import load_pointcloud_with_bboxes_info, draw_box, compute_bboxes, calib_bboxes, draw_vectors, ransac
from source_bb.utilities import parse_file, read_calib_file, extract_translations, extract_dimensions, extract_rotations
import numpy as np
import os
import random
import sys
import matplotlib as mlp
from point_net import get_pointnet_model
mlp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

TRAINING_DATA = BASE_DIR + '/training/'
EVAL_DATA = BASE_DIR + '/eval/'
LABEL_DATA = BASE_DIR + '/labels/'
CALIB_DATA = BASE_DIR + '/calib/training/calib/'
OVERFIT = BASE_DIR + '/overfit/'

LOG_DIR = BASE_DIR + '/log_deep/'
IMG_DIR = LOG_DIR + 'img/'
LOG_L1 = LOG_DIR + 'L1/'
LOG_L2 = LOG_DIR + 'L2/'

EVAL_DIR = LOG_DIR + 'eval/'
EVAL_L1 = EVAL_DIR + 'L1/'
EVAL_L2 = EVAL_DIR + 'L2/'

CKPT = BASE_DIR + '/ckpt_deep/'
CKPT_L1 = CKPT + 'L1/'
CKPT_L2 = CKPT + 'L2/'

''' Overfit stuff '''
# def create_dataset_one_file(file_num):
#     dataset = []
#
#     for _ in range(99):
#         dataset.append('%06d.bin' % file_num)
#
#     return np.array(dataset)
#
#
# def create_eval_one_file(file_num):
#     dataset = []
#
#     for _ in range(3):
#         dataset.append('%06d.bin' % file_num)
#
#     return np.array(dataset)
#
#
# OVERFIT_TRAINING_FILES = create_dataset_one_file(46)
# OVERFIT_EVAL_FILES = create_eval_one_file(46)

try:
    TRAINING_FILES = os.listdir(TRAINING_DATA)
    try:
        index = TRAINING_FILES.index('.DS_Store')
        del TRAINING_FILES[index]
    except ValueError:
        pass

    EVAL_FILES = os.listdir(EVAL_DATA)
    try:
        index = EVAL_FILES.index('.DS_Store')
        del EVAL_FILES[index]
    except ValueError:
        pass
except OSError:
    print('No directory training files')


def get_model(pointcloud, batch_size, is_training, dropout):
    """

    :param pointcloud: batched point cloud
    :return: prediction
    """
    weights = {
        'W_conv1': tf.get_variable('W_conv1', shape=[5, 5, 1, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv2': tf.get_variable('W_conv2', shape=[5, 5, 64, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv3': tf.get_variable('W_conv3', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv4': tf.get_variable('W_conv4', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv5': tf.get_variable('W_conv5', shape=[1, 1, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv6': tf.get_variable('W_conv6', shape=[1, 1, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conv7': tf.get_variable('W_conv7', shape=[1, 1, 64, 25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),

        'W_dconv1': tf.get_variable('W_dconv1', shape=[1, 1, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_dconv2': tf.get_variable('W_dconv2', shape=[1, 1, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_dconv3': tf.get_variable('W_dconv3', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_dconv4': tf.get_variable('W_dconv4', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_dconv5': tf.get_variable('W_dconv5', shape=[5, 5, 64, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_out': tf.get_variable('W_out', shape=[1, 3, 25, 25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),

        'W_conc1': tf.get_variable('W_conc1', shape=[3, 3, 512, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conc2': tf.get_variable('W_conc2', shape=[3, 3, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conc3': tf.get_variable('W_conc3', shape=[3, 3, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conc4': tf.get_variable('W_conc4', shape=[3, 3, 128, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'W_conc5': tf.get_variable('W_conc5', shape=[3, 3, 128, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
    }

    biases = {
        'b_conv1': tf.get_variable('b_conv1', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv2': tf.get_variable('b_conv2', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv3': tf.get_variable('b_conv3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv4': tf.get_variable('b_conv4', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv5': tf.get_variable('b_conv5', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv6': tf.get_variable('b_conv6', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conv7': tf.get_variable('b_conv7', shape=[25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),

        'b_dconv1': tf.get_variable('b_dconv1', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_dconv2': tf.get_variable('b_dconv2', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_dconv3': tf.get_variable('b_dconv3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_dconv4': tf.get_variable('b_dconv4', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_dconv5': tf.get_variable('b_dconv5', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),

        'b_conc1': tf.get_variable('b_conc1', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conc2': tf.get_variable('b_conc2', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conc3': tf.get_variable('b_conc3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conc4': tf.get_variable('b_conc4', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
        'b_conc5': tf.get_variable('b_conc5', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
    }

    # [8, 13545, 3]
    pointcloud = tf.expand_dims(pointcloud, -1)
    # [8, 13545, 3, 1]

    conv1 = tf.nn.conv2d(pointcloud, filter=weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv1']
    # conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    # conv1 = tf.contrib.layers.group_norm(conv1, groups=8)
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.layers.dropout(conv1, dropout, training=is_training)
    # [8, 13545, 3, 64]

    conv2 = tf.nn.conv2d(conv1, filter=weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv2']
    # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    # conv2 = tf.contrib.layers.group_norm(conv2, groups=8)
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.layers.dropout(conv2, dropout, training=is_training)
    # [8, 13545, 3, 64]

    conv3 = tf.nn.conv2d(conv2, filter=weights['W_conv3'], strides=[1, 2, 2, 1], padding='VALID') + biases['b_conv3']
    # conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    # conv3 = tf.contrib.layers.group_norm(conv3, groups=8)
    conv3 = tf.nn.relu(conv3)
    # conv3 = tf.layers.dropout(conv3, dropout, training=is_training)
    # [8, 6772, 1, 128]

    conv4 = tf.nn.conv2d(conv3, filter=weights['W_conv4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv4']
    # conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    # conv4 = tf.contrib.layers.group_norm(conv4, groups=16)
    conv4 = tf.nn.relu(conv4)
    # conv4 = tf.layers.dropout(conv4, dropout, training=is_training)
    # [8, 6772, 1, 128]

    conv5 = tf.nn.conv2d(conv4, filter=weights['W_conv5'], strides=[1, 2, 1, 1], padding='VALID') + biases['b_conv5']
    # conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    # conv5 = tf.contrib.layers.group_norm(conv5, groups=16)
    conv5 = tf.nn.relu(conv5)
    # conv5 = tf.layers.dropout(conv5, dropout, training=is_training)
    # [8, 3386, 1, 256]

    conv6 = tf.nn.conv2d(conv5, filter=weights['W_conv6'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv6']
    # conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    # conv6 = tf.contrib.layers.group_norm(conv6, groups=16)
    conv6 = tf.nn.relu(conv6)
    # conv6 = tf.layers.dropout(conv6, dropout, training=is_training)
    # [8, 3386, 1, 256]

    # base_shape = conv5.get_shape().as_list()
    # output_shape = [batch_size, base_shape[1], base_shape[2], 512]
    # dconv1 = tf.nn.conv2d_transpose(conv6, filter=weights['W_dconv1'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='VALID') + biases['b_dconv1']
    # dconv1 = tf.layers.batch_normalization(dconv1, training=True)
    # # dconv1 = tf.contrib.layers.group_norm(dconv1, groups=16)
    # dconv1 = tf.nn.relu(dconv1)
    # # [3, 3386, 1, 512]
    #
    # conc1 = tf.concat([dconv1, conv5], -1)
    # # [3, 3386, 1, 1024]
    # conc1 = tf.nn.conv2d(conc1, filter=weights['W_conc1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc1']
    # # [3, 3386, 1, 512]
    #
    # base_shape = conv4.get_shape().as_list()
    # output_shape = [batch_size, base_shape[1], base_shape[2], 256]
    # dconv2 = tf.nn.conv2d_transpose(conc1, filter=weights['W_dconv2'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv2']
    # dconv2 = tf.layers.batch_normalization(dconv2, training=True)
    # # dconv2 = tf.contrib.layers.group_norm(dconv2, groups=16)
    # dconv2 = tf.nn.relu(dconv2)
    # # [3, 6772, 1, 256]

    base_shape = conv4.get_shape().as_list()
    output_shape = [batch_size, base_shape[1], base_shape[2], 128]
    dconv1 = tf.nn.conv2d_transpose(conv6, filter=weights['W_dconv1'], output_shape=output_shape, strides=[1, 2, 1, 1], padding='VALID') + biases['b_dconv1']
    # dconv1 = tf.layers.batch_normalization(dconv1, training=is_training)
    dconv1 = tf.nn.relu(dconv1)
    # dconv1 = tf.layers.dropout(dconv1, dropout, training=is_training)
    # [8, 6772, 1, 128]

    conc2 = tf.concat([dconv1, conv4], -1)  # dcv2
    # [8, 6772, 1, 256]
    conc2 = tf.nn.conv2d(conc2, filter=weights['W_conc2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc2']
    conc2 = tf.nn.relu(conc2)
    # [8, 6772, 1, 128]

    base_shape = conv3.get_shape().as_list()
    output_shape = [batch_size, base_shape[1], base_shape[2], 128]
    dconv3 = tf.nn.conv2d_transpose(conc2, filter=weights['W_dconv3'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv3']
    # dconv3 = tf.layers.batch_normalization(dconv3, training=is_training)
    # dconv3 = tf.contrib.layers.group_norm(dconv3, groups=8)
    dconv3 = tf.nn.relu(dconv3)
    # dconv3 = tf.layers.dropout(dconv3, dropout, training=is_training)
    # [8, 6772, 1, 128]

    conc3 = tf.concat([dconv3, conv3], -1)
    # [8, 6772, 1, 256]
    conc3 = tf.nn.conv2d(conc3, filter=weights['W_conc3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc3']
    conc3 = tf.nn.relu(conc3)
    # [8, 6772, 1, 128]

    base_shape = conv2.get_shape().as_list()
    output_shape = [batch_size, base_shape[1], base_shape[2], 64]
    dconv4 = tf.nn.conv2d_transpose(conc3, filter=weights['W_dconv4'], output_shape=output_shape, strides=[1, 2, 2, 1], padding='VALID') + biases['b_dconv4']
    # dconv4 = tf.layers.batch_normalization(dconv4, training=is_training)
    # dconv4 = tf.contrib.layers.group_norm(dconv4, groups=8)
    dconv4 = tf.nn.relu(dconv4)
    # dconv4 = tf.layers.dropout(dconv4, dropout, training=is_training)
    # [8, 13545, 3, 64]

    conc4 = tf.concat([dconv4, conv2], -1)
    # [8, 13545, 3, 128]
    conc4 = tf.nn.conv2d(conc4, filter=weights['W_conc4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc4']
    conc4 = tf.nn.relu(conc4)
    # [8, 13545, 3, 64]

    base_shape = conv1.get_shape().as_list()
    output_shape = [batch_size, base_shape[1], base_shape[2], 64]
    dconv5 = tf.nn.conv2d_transpose(conc4, filter=weights['W_dconv5'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv5']
    # dconv5 = tf.layers.batch_normalization(dconv5, training=is_training)
    # dconv5 = tf.contrib.layers.group_norm(dconv5, groups=8)
    dconv5 = tf.nn.relu(dconv5)
    # dconv5 = tf.layers.dropout(dconv5, dropout, training=is_training)
    # [8, 13545, 3, 64]

    conc5 = tf.concat([dconv5, conv1], -1)
    # [8, 13545, 3, 128]
    conc5 = tf.nn.conv2d(conc5, filter=weights['W_conc5'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc5']
    conc5 = tf.nn.relu(conc5)
    conc5 = tf.layers.dropout(conc5, dropout, training=is_training)
    # [8, 13545, 3, 64]

    conv7 = tf.nn.conv2d(conc5, filter=weights['W_conv7'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv7']
    # [8, 13545, 3, 25]

    output = tf.nn.conv2d(conv7, filter=weights['W_out'], strides=[1, 1, 1, 1], padding='VALID')
    # [8, 13545, 1, 25]

    output = tf.reshape(output, [batch_size, 13545, 25])  # a
    # [8, 13545, 25]

    return output


def cls_loss(predict, y, loss_l1):
    """

    :param predict: output of the network
    :param y: labels
    :return: computed loss
    """
    y_clip = tf.clip_by_value(y, 0, 1)
    tot_positive = tf.clip_by_value(tf.tile(tf.expand_dims(tf.reduce_sum(y_clip, axis=1), axis=1), tf.TensorShape([1, predict.get_shape()[1], 1])), 1, np.inf)
    tot_negative = tf.clip_by_value(tf.tile(tf.expand_dims(tf.reduce_sum(1 - y_clip, axis=1), axis=1), tf.TensorShape([1, predict.get_shape()[1], 1])), 1, np.inf)

    # Cross Entropy
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_clip, logits=predict)

    # if loss_l1:
    #     # L1 Loss
    #     loss = tf.abs(tf.subtract(predict, y))
    #     # Smooth L1 Loss
    #     loss = tf.where(loss <= 1, tf.square(loss), loss)
    # else:
    #     # L2 Loss
    #     loss = tf.square(tf.subtract(predict, y))

    loss_positive = tf.multiply(loss, y_clip)
    loss_negative = tf.multiply(loss, 1 - y_clip)
    loss_positive = tf.reduce_sum(loss_positive, axis=1)
    loss_negative = tf.reduce_sum(loss_negative, axis=1)
    tot_positive = tf.reduce_max(tot_positive, axis=1)
    tot_negative = tf.reduce_max(tot_negative, axis=1)
    loss_positive = tf.divide(loss_positive, tot_positive)
    loss_negative = tf.divide(loss_negative, tot_negative)

    return tf.add(loss_positive, loss_negative)

    # loss = tf.reduce_mean(loss, axis=1)
    #
    # return loss


def bb_loss(predict, y, loss_l1):
    cls_slice = predict[:, :, :1]
    y_cls_slice = y[:, :, :1]

    loss_cls = cls_loss(cls_slice, y_cls_slice, loss_l1)

    bb_slice = predict[:, :, 1:]
    # bb_slice = predict
    y_bb_slice = y[:, :, 1:]

    indices = tf.clip_by_value(tf.tile(y_cls_slice, tf.constant([1, 1, 24])), 0, 1)
    tot_positive = tf.clip_by_value(tf.tile(tf.expand_dims(tf.reduce_sum(indices, axis=1), axis=1), tf.TensorShape([1, predict.get_shape()[1], 1])), 1, np.inf)
    tot_negative = tf.clip_by_value(tf.tile(tf.expand_dims(tf.reduce_sum(1 - indices, axis=1), axis=1), tf.TensorShape([1, predict.get_shape()[1], 1])), 1, np.inf)
    if loss_l1:
        # L1 Loss
        loss = tf.abs(tf.subtract(bb_slice, y_bb_slice))
        # Smooth L1 Loss
        loss = tf.where(loss <= 1, tf.square(loss), loss)
    else:
        # L2 Loss
        loss = tf.square(tf.subtract(bb_slice, y_bb_slice))

    loss_positive = tf.multiply(loss, indices)
    loss_negative = tf.multiply(loss, 1 - indices)
    loss_positive = tf.reduce_sum(loss_positive, axis=1)
    loss_negative = tf.reduce_sum(loss_negative, axis=1)
    tot_positive = tf.reduce_max(tot_positive, axis=1)
    tot_negative = tf.reduce_max(tot_negative, axis=1)
    loss_positive = tf.divide(loss_positive, tot_positive)
    loss_negative = tf.divide(loss_negative, tot_negative)
    loss_bb = tf.add(loss_positive, loss_negative)
    loss_bb = tf.expand_dims(tf.reduce_mean(loss_bb, axis=1), -1)

    return tf.reduce_mean(tf.add(loss_cls, loss_bb))


def create_optimizer(loss, lr):
    """

    :param loss: loss function
    :param lr: learning rate
    :return: optimizer
    """
    # opt = tf.train.AdamOptimizer(lr)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    gradients, variables = zip(*opt.compute_gradients(loss))
    capped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = opt.apply_gradients(zip(capped_gradients, variables))
    return optimizer


def batch_from_files(batch_size, batch_num, num_point, is_eval=False):
    pointcloud = []
    labels = []
    names = []
    dt = []

    if not is_eval:
        batch_start = batch_num * batch_size
        batch_end = (batch_num + 1) * batch_size
        batch = TRAINING_FILES[batch_start:batch_end]
        # batch = OVERFIT_TRAINING_FILES[batch_start:batch_end]
    else:
        batch_start = batch_num * batch_size
        batch_end = (batch_num + 1) * batch_size
        batch = EVAL_FILES[batch_start:batch_end]
        # batch = OVERFIT_EVAL_FILES

    for file_name in batch:
        try:
            if not is_eval:
                dt = load_pointcloud_with_bboxes_info(TRAINING_DATA + file_name)
            else:
                dt = load_pointcloud_with_bboxes_info(EVAL_DATA + file_name)
        except ValueError:
            print('File: {}'.format(file_name))

        # tot_objects = len(np.unique(dt[:, 3]))
        # dataset = np.array(random.sample(list(dt), num_point))
        # # safe subsample
        # while len(np.unique(dataset[:, 3])) != tot_objects:
        #     dataset = np.array(random.sample(list(dt), num_point))

        num_to_remove = len(dt) - num_point
        indices = np.where(dt[:, 3] == 0)
        to_remove = random.sample(list(indices[0]), num_to_remove)
        dataset = np.delete(dt, to_remove, axis=0)

        pc = dataset[:, :3]
        lbs = dataset[:, 3:28]

        pointcloud.append(pc)
        labels.append(lbs)
        names.append(os.path.splitext(file_name)[0])

    return np.array(pointcloud, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(names)


def train_gpu(batch_size, num_point, num_epochs, restore=False, epoch_to_restore=0, loss_l1=False):
    """

    :param batch_size: batch size
    :param num_point: 13545 total point in a point cloud
    :param num_epochs: num epochs to train
    :param restore: use trained network
    :param epoch_to_restore: select the epoch
    :param loss_l1: True:use Smooth L1 Loss, False:use L2 Loss
    :return:
    """
    with tf.Session() as sess:
        pointcloud_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size, num_point, 25))
        is_training = tf.placeholder(tf.bool)
        dropout = tf.placeholder(tf.float32)
        model = get_model(pointcloud_pl, batch_size, is_training, dropout)
        # is_training = tf.placeholder(tf.bool, shape=())
        # model = get_pointnet_model(pointcloud_pl, is_training)
        loss = bb_loss(model, y, loss_l1)
        optimizer = create_optimizer(loss, 0.001)
        num_batches = int(len(TRAINING_FILES) / batch_size)
        num_batches_eval = int(len(EVAL_FILES) / batch_size)
        saver = tf.train.Saver(max_to_keep=100)

        with tf.device('/gpu:0'):
            if restore:
                if loss_l1:
                    saver.restore(sess, CKPT_L1 + 'model.ckpt-{}'.format(epoch_to_restore))
                else:
                    saver.restore(sess, CKPT_L2 + 'model.ckpt-{}'.format(epoch_to_restore))
                print('\nModel restored')
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs - epoch_to_restore):
                random.shuffle(TRAINING_FILES)
                num_batch = 0
                # if epoch == 5:
                #     lr = 0.01
                # if epoch == 245:
                #     lr = 0.001
                print('***** Epoch {} out of {} *****'.format(epoch + epoch_to_restore + 1, num_epochs))
                for n in range(num_batches):
                    batch_x, batch_y, _ = batch_from_files(batch_size, n, num_point)
                    _, batch_loss_norm = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: True, dropout: 0.4})
                    # _, batch_loss_norm = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y:batch_y, is_training: True})
                    # augmented batch adding noise
                    batch_x = batch_x + np.random.uniform(-0.05, 0.05, batch_x.shape)
                    _, batch_loss_noise = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: True, dropout: 0.4})
                    # _, batch_loss_noise = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: True})

                    # augmented batch rotating by +/-10 degrees on z axis
                    # angle = random.randint(-10, 10)
                    # rot_mat = np.array(
                    #     [[np.cos(angle * (np.pi / 180)), -np.sin(angle * (np.pi / 180)), 0],
                    #      [np.sin(angle * (np.pi / 180)), np.cos(angle * (np.pi / 180)), 0],
                    #      [0, 0, 1]
                    #      ]
                    # )
                    # batch_x = np.matmul(batch_x, rot_mat)
                    # _, batch_loss_angle = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, learning_rate: lr})

                    # augmented batch flipping the point cloud on y axis
                    batch_x[:, :, 1] *= -1
                    batch_y[:, :, 2] *= -1
                    batch_y[:, :, 2] += 0
                    _, batch_loss_flip = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: True, dropout: 0.4})
                    # _, batch_loss_flip = sess.run([optimizer, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: True})

                    if loss_l1:
                        filepath = LOG_L1 + 'error_normal.txt'
                        filepath2 = LOG_L1 + 'error_flip.txt'
                        filepath3 = LOG_L1 + 'error_noise.txt'
                    else:
                        filepath = LOG_L2 + 'error_normal.txt'
                        filepath2 = LOG_L2 + 'error_flip.txt'
                        filepath3 = LOG_L2 + 'error_noise.txt'
                    if not os.path.exists(os.path.dirname(filepath)):
                        try:
                            os.makedirs(os.path.dirname(filepath))
                        except:
                            print('Error')
                    with open(filepath, 'a') as f:
                        f.write('{}\n'.format(batch_loss_norm))
                    if not os.path.exists(os.path.dirname(filepath2)):
                        try:
                            os.makedirs(os.path.dirname(filepath2))
                        except:
                            print('Error')
                    with open(filepath2, 'a') as f:
                        f.write('{}\n'.format(batch_loss_flip))
                    if not os.path.exists(os.path.dirname(filepath3)):
                        try:
                            os.makedirs(os.path.dirname(filepath3))
                        except:
                            print('Error')
                    with open(filepath3, 'a') as f:
                        f.write('{}\n'.format(batch_loss_noise))

                    print('Batch {} completed out of {}'.format(num_batch + 1, int(len(TRAINING_FILES) / batch_size)))
                    num_batch += 1
                print('\n')

                tot_error_eval = []

                print('\nEvaluation')

                # random.shuffle(EVAL_FILES)
                for n_eval in range(num_batches_eval):
                    batch_x, batch_y, batch_names = batch_from_files(batch_size, n_eval, num_point, True)

                    predict, loss_eval = sess.run([model, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: False, dropout: 0})
                    # predict, loss_eval = sess.run([model, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: False})
                    # predict = np.array(predict[0])  # no loss

                    tot_error_eval.append(loss_eval)

                    # objectness
                    # cls_predict = np.round(predict[:, :, :1])
                    cls_predict = np.round(tf.sigmoid(predict[:, :, :1]).eval())
                    # objectness label
                    cls_y = batch_y[:, :, :1]
                    cls_temp = np.equal(cls_predict, np.clip(cls_y, 0, 1))
                    cls_temp = cls_temp.astype(np.float32)
                    cls_res = np.sum(cls_temp) / (batch_size * num_point) * 100

                    if loss_l1:
                        filepath = LOG_L1 + 'cls.txt'
                    else:
                        filepath = LOG_L2 + 'cls.txt'
                    if not os.path.exists(os.path.dirname(filepath)):
                        try:
                            os.makedirs(os.path.dirname(filepath))
                        except:
                            print('Error')
                    with open(filepath, 'a') as f:
                        f.write('{0:.3f}\n'.format(cls_res))

                    cls_temp_y = np.equal(np.clip(cls_y, 0, 1), 1)
                    cls_points = np.sum(cls_temp_y.astype(np.float32))
                    cls_temp_predict = np.equal(cls_predict, 1)
                    cls_temp_predict_recall = cls_temp_predict.astype(np.float32)
                    cls_temp_1 = np.logical_and(cls_temp_y, cls_temp_predict)
                    cls_temp_1 = cls_temp_1.astype(np.float32)
                    cls_res_1 = np.sum(cls_temp_1) / cls_points * 100
                    tot = np.sum(cls_temp_predict_recall)
                    if tot != 0:
                        recall = np.sum(cls_temp_1) / tot * 100
                    else:
                        recall = -1

                    if loss_l1:
                        filepath = LOG_L1 + 'cls_1.txt'
                    else:
                        filepath = LOG_L2 + 'cls_1.txt'
                    if not os.path.exists(os.path.dirname(filepath)):
                        try:
                            os.makedirs(os.path.dirname(filepath))
                        except:
                            print('Error')
                    with open(filepath, 'a') as f:
                        f.write('{0:.3f}\n'.format(cls_res_1))

                    if loss_l1:
                        filepath = LOG_L1 + 'rec_1.txt'
                    else:
                        filepath = LOG_L2 + 'rec_1.txt'
                    if not os.path.exists(os.path.dirname(filepath)):
                        try:
                            os.makedirs(os.path.dirname(filepath))
                        except:
                            print('Error')
                    with open(filepath, 'a') as f:
                        f.write('{0:.3f}\n'.format(recall))

                    if ((epoch + epoch_to_restore + 1) % 20 == 0) or ((epoch + epoch_to_restore + 1) == num_epochs):

                        # bounding boxes
                        bb_predict = predict[:, :, 1:]
                        # bb_predict = predict

                        for i_batch in range(batch_size):

                            # one pointcloud cls labels
                            batch = cls_y[i_batch]

                            # one pointcloud cls predicted
                            batch_predict = cls_predict[i_batch]

                            # same input pointcloud
                            batch_points = batch_x[i_batch]

                            # number of the file in input
                            file_num = int(batch_names[i_batch])

                            # max number of object in a pointcloud
                            max_batch = int(np.max(batch))

                            # list of indices of the objects in the pointcloud
                            objects_indices = np.unique(batch)

                            calib = read_calib_file(CALIB_DATA + '%06d.txt' % file_num)
                            labels = parse_file(LABEL_DATA + '%06d.txt' % file_num, ' ')
                            bboxes = compute_bboxes(extract_translations(labels), extract_rotations(labels), extract_dimensions(labels))
                            calib_boxes = calib_bboxes(bboxes, calib)

                            for i_object in range(max_batch):
                                if (i_object + 1) in objects_indices:

                                    # bounding box of the car
                                    bbox_gt = calib_boxes[i_object]

                                    # all the points belonging to the object
                                    mask = tf.cast(tf.equal(batch, (i_object + 1)), tf.int32).eval()

                                    # mask repeated 24 times to be multipled with the vectors
                                    mask_bb = np.tile(mask, [1, 1, 24])

                                    # mask repeated 3 times to be multipled with the points
                                    mask_points = np.tile(mask, [1, 1, 3])

                                    # all the points belonging to the object (predicted)
                                    mask_cls = tf.cast(tf.equal(batch_predict, 1), tf.int32).eval()

                                    # mask repeated 24 times to be multipled with the vectors (predicted)
                                    mask_bb_from_cls = np.tile(mask_cls, [1, 1, 24])

                                    # mask repeated 3 times to be multipled with the points (predicted)
                                    mask_points_cls = np.tile(mask_cls, [1, 1, 3])

                                    # car points of the specific object
                                    car_points = np.around(np.multiply(batch_points, mask_points), 3)

                                    # car points of the specific object (predicted)
                                    car_points_from_cls = np.around(np.multiply(car_points, mask_points_cls), 3)

                                    # points repeated 8 times to be summed with the corners
                                    temp_batch_points = np.repeat(batch_points, 8, axis=1)

                                    # all the bounding boxes predicted by the network
                                    temp_bb_predict = np.around(np.multiply(bb_predict[i_batch], mask_bb), 3)

                                    # all the points to be summed with the vectors to get the corners
                                    temp_batch_points = np.around(np.multiply(temp_batch_points, mask_bb), 3)

                                    # all the bounding boxes predicted by the network corresponding to the points of the cls
                                    temp_bb_predict_from_cls = np.around(np.multiply(temp_bb_predict, mask_bb_from_cls), 3)

                                    # all the points to be summed with the vectors to get the corners from the cls prediction
                                    temp_batch_points_from_cls = np.around(np.multiply(temp_batch_points, mask_bb_from_cls), 3)

                                    # remove all the null elements
                                    vectors_bb_predict = [i for i in temp_bb_predict[0] if any(i + 0)]
                                    vectors_bb_predict_from_cls = [i for i in temp_bb_predict_from_cls[0] if any(i + 0)]
                                    points_bb = [i for i in temp_batch_points[0] if any(i + 0)]
                                    points_bb_from_cls = [i for i in temp_batch_points_from_cls[0] if any(i + 0)]
                                    car_points_filtered = [i for i in car_points[0] if any(i + 0)]
                                    car_points_from_cls_filtered = [i for i in car_points_from_cls[0] if any(i + 0)]

                                    vectors_bb_predict = np.around(vectors_bb_predict, 3)
                                    if len(vectors_bb_predict_from_cls) > 0:
                                        vectors_bb_predict_from_cls = np.around(vectors_bb_predict_from_cls, 3)

                                    points_bb = np.around(points_bb, 3)
                                    if len(points_bb_from_cls) > 0:
                                        points_bb_from_cls = np.around(points_bb_from_cls, 3)

                                    car_points_filtered = np.around(car_points_filtered, 3)
                                    if len(car_points_from_cls_filtered) > 0:
                                        car_points_from_cls_filtered = np.around(car_points_from_cls_filtered, 3)

                                    # compute the box adding to the points the distance vectors from the corners
                                    bbox_predict = np.add(points_bb, vectors_bb_predict)

                                    if len(vectors_bb_predict_from_cls) > 0 and len(points_bb_from_cls) > 0:
                                        bbox_predict_from_cls = np.add(points_bb_from_cls, vectors_bb_predict_from_cls)

                                    # for start in range(8):
                                    #     to_plot = bbox_predict[:, start::8]
                                    #     if loss_l1:
                                    #         filepath = LOG_L1 + '{}/{}/{}_{}_vec-{}'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #         filepath2 = LOG_L1 + '{}/{}/{}_{}_vec-{}_points'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #         filepath3 = LOG_L1 + '{}/{}/{}_{}_vec-{}_box'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #     else:
                                    #         filepath = LOG_L2 + '{}/{}/{}_{}_vec-{}'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #         filepath2 = LOG_L2 + '{}/{}/{}_{}_vec-{}_points'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #         filepath3 = LOG_L2 + '{}/{}/{}_{}_vec-{}_box'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object, start)
                                    #     if not os.path.exists(os.path.dirname(filepath)):
                                    #         try:
                                    #             os.makedirs(os.path.dirname(filepath))
                                    #         except:
                                    #             print('Error')
                                    #     np.save(filepath, to_plot)
                                    #     if not os.path.exists(os.path.dirname(filepath2)):
                                    #         try:
                                    #             os.makedirs(os.path.dirname(filepath2))
                                    #         except:
                                    #             print('Error')
                                    #     np.save(filepath2, car_points_filtered)
                                    #     if not os.path.exists(os.path.dirname(filepath3)):
                                    #         try:
                                    #             os.makedirs(os.path.dirname(filepath3))
                                    #         except:
                                    #             print('Error')
                                    #     np.save(filepath3, bbox_gt)

                                    bbox_predict = np.expand_dims(np.median(bbox_predict, axis=0), -1)
                                    bbox_predict = np.reshape(bbox_predict, [3, 8])

                                    final_bbox = []
                                    if len(vectors_bb_predict_from_cls) > 0 and len(points_bb_from_cls) > 0:
                                        # bbox_predict_from_cls = np.expand_dims(np.median(bbox_predict_from_cls, axis=0), -1)
                                        # bbox_predict_from_cls = np.reshape(bbox_predict_from_cls, [3, 8])
                                        ransac_res = []
                                        for start in range(8):
                                            ransac_res.append(ransac(bbox_predict_from_cls[:, start::8], 10, 0.75))

                                        for index in range(8):
                                            final_bbox.append(np.median(ransac_res[index], axis=0))
                                        final_bbox = np.stack(final_bbox)
                                        final_bbox = np.transpose(final_bbox)

                                    # centroid_predict = KMeans(n_clusters=1).fit(np.transpose(bbox_predict)).cluster_centers_[0]
                                    # centroid_y = KMeans(n_clusters=1).fit(np.transpose(bbox_gt)).cluster_centers_[0]
                                    #
                                    # distance = np.sqrt(np.square(centroid_y[0] - centroid_predict[0]) + np.square(centroid_y[1] - centroid_predict[1]) - np.square(centroid_y[2] - centroid_predict[2]))

                                    # if loss_l1:
                                    #     filepath = LOG_L1 + '{}/{}/{}_{}_bb.txt'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    # else:
                                    #     filepath = LOG_L2 + '{}/{}/{}_{}_bb.txt'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    # if not os.path.exists(os.path.dirname(filepath)):
                                    #     try:
                                    #         os.makedirs(os.path.dirname(filepath))
                                    #     except:
                                    #         print('Error')
                                    # with open(filepath, 'a') as f:
                                    #     for i in range(3):
                                    #         for j in range(8):
                                    #             f.write('{} '.format(bbox_predict[i, j]))
                                    #         f.write('\n')
                                    #     for i in range(3):
                                    #         for j in range(8):
                                    #             f.write('{} '.format(bbox_gt[i, j]))
                                    #         f.write('\n')

                                    f = plt.figure(figsize=(15, 8))
                                    ax = f.add_subplot(111, projection='3d')
                                    ax.scatter(*np.transpose(car_points_filtered[:, [0, 1, 2]]), s=3, c='#00FF00', cmap='gray')
                                    if len(car_points_from_cls_filtered) > 0:
                                        ax.scatter(*np.transpose(car_points_from_cls_filtered[:, [0, 1, 2]]), s=3, c='#FF0000', cmap='gray')
                                    draw_box(ax, bbox_gt, 'green')
                                    draw_box(ax, bbox_predict, 'blue')
                                    # if len(vectors_bb_predict_from_cls) > 0:
                                    #     draw_box(ax, bbox_predict_from_cls, 'red')
                                    if len(final_bbox) > 0:
                                        draw_box(ax, final_bbox, 'red')

                                    if loss_l1:
                                        filepath = LOG_L1 + '{}/{}/{}_{}.png'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    else:
                                        filepath = LOG_L2 + '{}/{}/{}_{}.png'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    if not os.path.exists(os.path.dirname(filepath)):
                                        try:
                                            os.makedirs(os.path.dirname(filepath))
                                        except:
                                            print('Error')
                                    plt.savefig(filepath)
                                    plt.close()

                                    # filepath = LOG_L1 + '{}/{}/{}_{}_points_gt'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    # try:
                                    #     os.makedirs(os.path.dirname(filepath))
                                    # except:
                                    #     print('Error')
                                    # np.save(filepath, car_points_filtered)
                                    #
                                    # if len(car_points_from_cls_filtered) > 0:
                                    #     filepath = LOG_L1 + '{}/{}/{}_{}_points'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    #     np.save(filepath, car_points_from_cls_filtered)
                                    #
                                    # filepath = LOG_L1 + '{}/{}/{}_{}_gt'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    # np.save(filepath, bbox_gt)
                                    #
                                    # filepath = LOG_L1 + '{}/{}/{}_{}_predict'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    # np.save(filepath, bbox_predict)
                                    #
                                    # if len(final_bbox) > 0:
                                    #     filepath = LOG_L1 + '{}/{}/{}_{}_final'.format(epoch_to_restore + epoch + 1, n_eval, i_batch, i_object)
                                    #     np.save(filepath, final_bbox)

                        if loss_l1:
                            filepath = CKPT_L1 + 'model.ckpt'
                        else:
                            filepath = CKPT_L2 + 'model.ckpt'
                        if not os.path.exists(os.path.dirname(filepath)):
                            try:
                                os.makedirs(os.path.dirname(filepath))
                            except:
                                print('Error')
                        saver.save(sess, filepath, global_step=epoch_to_restore+epoch+1)

                if loss_l1:
                    filepath = LOG_L1 + 'error_eval.txt'
                else:
                    filepath = LOG_L2 + 'error_eval.txt'
                if not os.path.exists(os.path.dirname(filepath)):
                    try:
                        os.makedirs(os.path.dirname(filepath))
                    except:
                        print('Error')
                with open(filepath, 'a') as f:
                    f.write('{}\n'.format(np.median(np.array(tot_error_eval))))

            print('Training finished!')


# def eval_gpu(batch_size, num_point, epoch_to_restore=0, loss_l1=False):
#     with tf.Session() as sess:
#         pointcloud_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
#         is_training = tf.placeholder(tf.bool)
#         model = get_model(pointcloud_pl, batch_size, is_training)
#         y = tf.placeholder(tf.float32, shape=(batch_size, num_point, 25))
#         loss = bb_loss(model, y, loss_l1)
#         num_batches_eval = int(len(EVAL_FILES) / batch_size)
#         saver = tf.train.Saver()
#         with tf.device('/gpu:0'):
#             if loss_l1:
#                 saver.restore(sess, CKPT_L1 + 'model.ckpt-{}'.format(epoch_to_restore))
#             else:
#                 # saver = tf.train.import_meta_graph('model-1000.meta')
#                 saver.restore(sess, CKPT_L2 + 'model.ckpt-{}'.format(epoch_to_restore))
#                 # model = tf.get_default_graph().get_tensor_by_name('reshape:0')
#             print('\nModel restored')
#
#             print('\nEvaluation')
#
#             tot_error_eval = []
#
#             # random.shuffle(EVAL_FILES)
#             for n_eval in range(num_batches_eval):
#
#                 batch_x, batch_y, batch_names = batch_from_files(batch_size, n_eval, num_point, True)
#
#                 predict, loss_eval = sess.run([model, loss], feed_dict={pointcloud_pl: batch_x, y: batch_y, is_training: False})
#                 # predict = np.array(predict[0])  # no loss
#
#                 tot_error_eval.append(loss_eval)
#
#                 # objectness
#                 cls_predict = np.round(tf.sigmoid(predict[:, :, :1]).eval())
#                 # objectness label
#                 cls_y = batch_y[:, :, :1]
#                 # cls_temp = np.equal(cls_predict, np.clip(cls_y, 0, 1))
#                 # cls_temp = cls_temp.astype(np.float32)
#                 # cls_res = np.sum(cls_temp) / (batch_size * num_point) * 100
#                 #
#                 # if loss_l1:
#                 #     filepath = EVAL_L1 + '{}/cls.txt'.format(epoch_to_restore)
#                 # else:
#                 #     filepath = EVAL_L2 + '{}/cls.txt'.format(epoch_to_restore)
#                 # if not os.path.exists(os.path.dirname(filepath)):
#                 #     try:
#                 #         os.makedirs(os.path.dirname(filepath))
#                 #     except:
#                 #         print('Error')
#                 # with open(filepath, 'a') as f:
#                 #     f.write('{0:.3f}\n'.format(cls_res))
#                 #
#                 # cls_temp_y = np.equal(np.clip(cls_y, 0, 1), 1)
#                 # cls_points = np.sum(cls_temp_y.astype(np.float32))
#                 # cls_temp_predict = np.equal(cls_predict, 1)
#                 # cls_temp_1 = np.logical_and(cls_temp_y, cls_temp_predict)
#                 # cls_temp_1 = cls_temp_1.astype(np.float32)
#                 # cls_res_1 = np.sum(cls_temp_1) / cls_points * 100
#                 #
#                 # if loss_l1:
#                 #     filepath = EVAL_L1 + '{}/cls_1.txt'.format(epoch_to_restore)
#                 # else:
#                 #     filepath = EVAL_L2 + '{}/cls_1.txt'.format(epoch_to_restore)
#                 # if not os.path.exists(os.path.dirname(filepath)):
#                 #     try:
#                 #         os.makedirs(os.path.dirname(filepath))
#                 #     except:
#                 #         print('Error')
#                 # with open(filepath, 'a') as f:
#                 #     f.write('{0:.3f}\n'.format(cls_res_1))
#
#                 bb_predict = predict[:, :, 1:]
#
#                 for i_batch in range(batch_size):
#                     # one pointcloud cls labels
#                     batch = cls_y[i_batch]
#
#                     # one pointcloud cls predicted
#                     batch_predict = cls_predict[i_batch]
#
#                     # same input pointcloud
#                     batch_points = batch_x[i_batch]
#
#                     # number of the file in input
#                     file_num = int(batch_names[i_batch])
#
#                     # max number of object in a pointcloud
#                     max_batch = int(np.max(batch))
#
#                     # list of indices of the objects in the pointcloud
#                     objects_indices = np.unique(batch)
#
#                     calib = read_calib_file(CALIB_DATA + '%06d.txt' % file_num)
#                     labels = parse_file(LABEL_DATA + '%06d.txt' % file_num, ' ')
#                     bboxes = compute_bboxes(extract_translations(labels), extract_rotations(labels), extract_dimensions(labels))
#                     calib_boxes = calib_bboxes(bboxes, calib)
#
#                     for i_object in range(max_batch):
#                         if (i_object + 1) in objects_indices:
#
#                             # bounding box of the car
#                             bbox_gt = calib_boxes[i_object]
#
#                             # all the points belonging to the object
#                             mask = tf.cast(tf.equal(batch, (i_object + 1)), tf.int32).eval()
#
#                             # mask repeated 24 times to be multipled with the vectors
#                             mask_bb = np.tile(mask, [1, 1, 24])
#
#                             # mask repeated 3 times to be multipled with the points
#                             mask_points = np.tile(mask, [1, 1, 3])
#
#                             # all the points belonging to the object (predicted)
#                             mask_cls = tf.cast(tf.equal(batch_predict, 1), tf.int32).eval()
#
#                             # mask repeated 24 times to be multipled with the vectors (predicted)
#                             mask_bb_from_cls = np.tile(mask_cls, [1, 1, 24])
#
#                             # mask repeated 3 times to be multipled with the points (predicted)
#                             mask_points_cls = np.tile(mask_cls, [1, 1, 3])
#
#                             # car points of the specific object
#                             car_points = np.around(np.multiply(batch_points, mask_points), 3)
#
#                             # car points of the specific object (predicted)
#                             car_points_from_cls = np.around(np.multiply(car_points, mask_points_cls), 3)
#
#                             # points repeated 8 times to be summed with the corners
#                             temp_batch_points = np.repeat(batch_points, 8, axis=1)
#
#                             # all the bounding boxes predicted by the network
#                             temp_bb_predict = np.around(np.multiply(bb_predict[i_batch], mask_bb), 3)
#
#                             # all the points to be summed with the vectors to get the corners
#                             temp_batch_points = np.around(np.multiply(temp_batch_points, mask_bb), 3)
#
#                             # all the bounding boxes predicted by the network corresponding to the points of the cls
#                             temp_bb_predict_from_cls = np.around(np.multiply(temp_bb_predict, mask_bb_from_cls), 3)
#
#                             # all the points to be summed with the vectors to get the corners from the cls prediction
#                             temp_batch_points_from_cls = np.around(np.multiply(temp_batch_points, mask_bb_from_cls), 3)
#
#                             # remove all the null elements
#                             vectors_bb_predict = [i for i in temp_bb_predict[0] if any(i + 0)]
#                             vectors_bb_predict_from_cls = [i for i in temp_bb_predict_from_cls[0] if any(i + 0)]
#                             points_bb = [i for i in temp_batch_points[0] if any(i + 0)]
#                             points_bb_from_cls = [i for i in temp_batch_points_from_cls[0] if any(i + 0)]
#                             car_points_filtered = [i for i in car_points[0] if any(i + 0)]
#                             car_points_from_cls_filtered = [i for i in car_points_from_cls[0] if any(i + 0)]
#
#                             vectors_bb_predict = np.around(vectors_bb_predict, 3)
#                             if len(vectors_bb_predict_from_cls) > 0:
#                                 vectors_bb_predict_from_cls = np.around(vectors_bb_predict_from_cls, 3)
#
#                             points_bb = np.around(points_bb, 3)
#                             if len(points_bb_from_cls) > 0:
#                                 points_bb_from_cls = np.around(points_bb_from_cls, 3)
#
#                             car_points_filtered = np.around(car_points_filtered, 3)
#                             if len(car_points_from_cls_filtered) > 0:
#                                 car_points_from_cls_filtered = np.around(car_points_from_cls_filtered, 3)
#
#                             # compute the box adding to the points the distance vectors from the corners
#                             bbox_predict = np.add(points_bb, vectors_bb_predict)
#
#                             if len(vectors_bb_predict_from_cls) > 0 and len(points_bb_from_cls) > 0:
#                                 bbox_predict_from_cls = np.add(points_bb_from_cls, vectors_bb_predict_from_cls)
#
#                             # for start in range(8):
#                             #     to_plot = bbox_predict_from_cls[:, start::8]
#                             #     if loss_l1:
#                             #         filepath = EVAL_L1 + '{}/{}/{}_{}_vec-{}'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #         filepath2 = EVAL_L1 + '{}/{}/{}_{}_vec-{}_points'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #         filepath3 = EVAL_L1 + '{}/{}/{}_{}_vec-{}_box'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #     else:
#                             #         filepath = EVAL_L2 + '{}/{}/{}_{}_vec-{}'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #         filepath2 = EVAL_L2 + '{}/{}/{}_{}_vec-{}_points'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #         filepath3 = EVAL_L2 + '{}/{}/{}_{}_vec-{}_box'.format(epoch_to_restore, n_eval, i_batch, i_object, start)
#                             #     if not os.path.exists(os.path.dirname(filepath)):
#                             #         try:
#                             #             os.makedirs(os.path.dirname(filepath))
#                             #         except:
#                             #             print('Error')
#                             #     np.save(filepath, to_plot)
#                             #     if not os.path.exists(os.path.dirname(filepath2)):
#                             #         try:
#                             #             os.makedirs(os.path.dirname(filepath2))
#                             #         except:
#                             #             print('Error')
#                             #     np.save(filepath2, car_points_from_cls_filtered)
#                             #     if not os.path.exists(os.path.dirname(filepath3)):
#                             #         try:
#                             #             os.makedirs(os.path.dirname(filepath3))
#                             #         except:
#                             #             print('Error')
#                             #     np.save(filepath3, bbox_gt)
#
#                             bbox_predict = np.expand_dims(np.median(bbox_predict, axis=0), -1)
#                             bbox_predict = np.reshape(bbox_predict, [3, 8])
#
#                             final_bbox = []
#                             if len(vectors_bb_predict_from_cls) > 0 and len(points_bb_from_cls) > 0:
#                                 # bbox_predict_from_cls = np.expand_dims(np.median(bbox_predict_from_cls, axis=0), -1)
#                                 # bbox_predict_from_cls = np.reshape(bbox_predict_from_cls, [3, 8])
#                                 ransac_res = []
#                                 for start in range(8):
#                                     ransac_res.append(ransac(bbox_predict_from_cls[:, start::8], 10, 0.75))
#
#                                 # print('****************** {}\n'.format(np.array(ransac_res).shape))
#
#                                 for index in range(8):
#                                     # print('***** {} {}\n'.format(ransac_res[index].shape, ransac_res[index]))
#                                     final_bbox.append(np.median(ransac_res[index], axis=0))
#                                 final_bbox = np.stack(final_bbox)
#                                 final_bbox = np.transpose(final_bbox)
#
#                             f = plt.figure(figsize=(15, 8))
#                             ax = f.add_subplot(111, projection='3d')
#                             ax.scatter(*np.transpose(car_points_filtered[:, [0, 1, 2]]), s=3, c='#00FF00', cmap='gray')
#                             if len(car_points_from_cls_filtered) > 0:
#                                 ax.scatter(*np.transpose(car_points_from_cls_filtered[:, [0, 1, 2]]), s=3, c='#FF0000', cmap='gray')
#                             draw_box(ax, bbox_gt, 'green')
#                             draw_box(ax, bbox_predict, 'blue')
#                             # if len(vectors_bb_predict_from_cls) > 0:
#                             #     draw_box(ax, bbox_predict_from_cls, 'red')
#                             if len(final_bbox) > 0:
#                                 draw_box(ax, final_bbox, 'red')
#
#                             if loss_l1:
#                                 filepath = EVAL_L1 + '{}/{}/{}_{}.png'.format(epoch_to_restore, n_eval + 1, i_batch, i_object)
#                             else:
#                                 filepath = EVAL_L2 + '{}/{}/{}_{}.png'.format(epoch_to_restore, n_eval + 1, i_batch, i_object)
#                             if not os.path.exists(os.path.dirname(filepath)):
#                                 try:
#                                     os.makedirs(os.path.dirname(filepath))
#                                 except:
#                                     print('Error')
#                             plt.savefig(filepath)
#                             plt.close()
#
#             # if loss_l1:
#             #     filepath = EVAL_L1 + '{}/error_eval.txt'.format(epoch_to_restore)
#             # else:
#             #     filepath = EVAL_L2 + '{}/error_eval.txt'.format(epoch_to_restore)
#             # if not os.path.exists(os.path.dirname(filepath)):
#             #     try:
#             #         os.makedirs(os.path.dirname(filepath))
#             #     except:
#             #         print('Error')
#             # with open(filepath, 'a') as f:
#             #     f.write('{}\n'.format(np.median(np.array(tot_error_eval))))


# def get_model(pointcloud, batch_size):
#     """
#
#     :param pointcloud: batched point cloud
#     :return: prediction
#     """
#     weights = {
#         'W_conv1': tf.get_variable('W_conv1', shape=[5, 5, 1, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv2': tf.get_variable('W_conv2', shape=[5, 5, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv3': tf.get_variable('W_conv3', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv4': tf.get_variable('W_conv4', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv5': tf.get_variable('W_conv5', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv6': tf.get_variable('W_conv6', shape=[1, 1, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conv7': tf.get_variable('W_conv7', shape=[1, 1, 128, 25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#
#         'W_dconv1': tf.get_variable('W_dconv1', shape=[1, 1, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_dconv2': tf.get_variable('W_dconv2', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_dconv3': tf.get_variable('W_dconv3', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_dconv4': tf.get_variable('W_dconv4', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_dconv5': tf.get_variable('W_dconv5', shape=[5, 5, 128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_out': tf.get_variable('W_out', shape=[1, 3, 25, 25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#
#         'W_conc1': tf.get_variable('W_conc1', shape=[3, 3, 512, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conc2': tf.get_variable('W_conc2', shape=[3, 3, 512, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conc3': tf.get_variable('W_conc3', shape=[3, 3, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conc4': tf.get_variable('W_conc4', shape=[3, 3, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'W_conc5': tf.get_variable('W_conc5', shape=[3, 3, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
#     }
#
#     biases = {
#         'b_conv1': tf.get_variable('b_conv1', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv2': tf.get_variable('b_conv2', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv3': tf.get_variable('b_conv3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv4': tf.get_variable('b_conv4', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv5': tf.get_variable('b_conv5', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv6': tf.get_variable('b_conv6', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conv7': tf.get_variable('b_conv7', shape=[25], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#
#         'b_dconv1': tf.get_variable('b_dconv1', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_dconv2': tf.get_variable('b_dconv2', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_dconv3': tf.get_variable('b_dconv3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_dconv4': tf.get_variable('b_dconv4', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_dconv5': tf.get_variable('b_dconv5', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#
#         'b_conc1': tf.get_variable('b_conc1', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conc2': tf.get_variable('b_conc2', shape=[256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conc3': tf.get_variable('b_conc3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conc4': tf.get_variable('b_conc4', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01)),
#         'b_conc5': tf.get_variable('b_conc5', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
#     }
#
#     # [3, 13545, 3]
#     pointcloud = tf.expand_dims(pointcloud, -1)
#     # [3, 13545, 3, 1]
#
#     conv1 = tf.nn.conv2d(pointcloud, filter=weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv1']
#     conv1 = tf.layers.batch_normalization(conv1, training=True)
#     # conv1 = tf.contrib.layers.group_norm(conv1, groups=8)
#     conv1 = tf.nn.relu(conv1)
#     # [3, 13545, 3, 128]
#
#     conv2 = tf.nn.conv2d(conv1, filter=weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv2']
#     conv2 = tf.layers.batch_normalization(conv2, training=True)
#     # conv2 = tf.contrib.layers.group_norm(conv2, groups=8)
#     conv2 = tf.nn.relu(conv2)
#     # [3, 13545, 3, 128]
#
#     conv3 = tf.nn.conv2d(conv2, filter=weights['W_conv3'], strides=[1, 2, 2, 1], padding='VALID') + biases['b_conv3']
#     conv3 = tf.layers.batch_normalization(conv3, training=True)
#     # conv3 = tf.contrib.layers.group_norm(conv3, groups=8)
#     conv3 = tf.nn.relu(conv3)
#     # [3, 6772, 1, 128]
#
#     conv4 = tf.nn.conv2d(conv3, filter=weights['W_conv4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv4']
#     conv4 = tf.layers.batch_normalization(conv4, training=True)
#     # conv4 = tf.contrib.layers.group_norm(conv4, groups=16)
#     conv4 = tf.nn.relu(conv4)
#     # [3, 6772, 1, 256]
#
#     conv5 = tf.nn.conv2d(conv4, filter=weights['W_conv5'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv5']
#     conv5 = tf.layers.batch_normalization(conv5, training=True)
#     # conv5 = tf.contrib.layers.group_norm(conv5, groups=16)
#     conv5 = tf.nn.relu(conv5)
#     # [3, 6772, 1, 256]
#
#     conv6 = tf.nn.conv2d(conv5, filter=weights['W_conv6'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv6']
#     conv6 = tf.layers.batch_normalization(conv6, training=True)
#     # conv6 = tf.contrib.layers.group_norm(conv6, groups=16)
#     conv6 = tf.nn.relu(conv6)
#     # [3, 6772, 1, 256]
#
#     base_shape = conv5.get_shape().as_list()
#     output_shape = [batch_size, base_shape[1], base_shape[2], 256]
#     dconv1 = tf.nn.conv2d_transpose(conv6, filter=weights['W_dconv1'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv1']
#     dconv1 = tf.layers.batch_normalization(dconv1, training=True)
#     # dconv1 = tf.contrib.layers.group_norm(dconv1, groups=16)
#     dconv1 = tf.nn.relu(dconv1)
#     # [3, 6772, 1, 256]
#
#     conc1 = tf.concat([dconv1, conv5], -1)
#     # [3, 6772, 1, 512]
#     conc1 = tf.nn.conv2d(conc1, filter=weights['W_conc1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc1']
#     # [3, 6772, 1, 256]
#
#     base_shape = conv4.get_shape().as_list()
#     output_shape = [batch_size, base_shape[1], base_shape[2], 256]
#     dconv2 = tf.nn.conv2d_transpose(conc1, filter=weights['W_dconv2'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv2']
#     dconv2 = tf.layers.batch_normalization(dconv2, training=True)
#     # dconv2 = tf.contrib.layers.group_norm(dconv2, groups=16)
#     dconv2 = tf.nn.relu(dconv2)
#     # [3, 6772, 1, 256]
#
#     conc2 = tf.concat([dconv2, conv4], -1)
#     # [3, 6772, 1, 512]
#     conc2 = tf.nn.conv2d(conc2, filter=weights['W_conc2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc2']
#     # [3, 6772, 1, 256]
#
#     base_shape = conv3.get_shape().as_list()
#     output_shape = [batch_size, base_shape[1], base_shape[2], 128]
#     dconv3 = tf.nn.conv2d_transpose(conc2, filter=weights['W_dconv3'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv3']
#     dconv3 = tf.layers.batch_normalization(dconv3, training=True)
#     # dconv3 = tf.contrib.layers.group_norm(dconv3, groups=8)
#     dconv3 = tf.nn.relu(dconv3)
#     # [3, 6772, 1, 128]
#
#     conc3 = tf.concat([dconv3, conv3], -1)
#     # [3, 6772, 1, 256]
#     conc3 = tf.nn.conv2d(conc3, filter=weights['W_conc3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc3']
#     # [3, 6772, 1, 128]
#
#     base_shape = conv2.get_shape().as_list()
#     output_shape = [batch_size, base_shape[1], base_shape[2], 128]
#     dconv4 = tf.nn.conv2d_transpose(conc3, filter=weights['W_dconv4'], output_shape=output_shape, strides=[1, 2, 2, 1], padding='VALID') + biases['b_dconv4']
#     dconv4 = tf.layers.batch_normalization(dconv4, training=True)
#     # dconv4 = tf.contrib.layers.group_norm(dconv4, groups=8)
#     dconv4 = tf.nn.relu(dconv4)
#     # [3, 13545, 3, 128]
#
#     conc4 = tf.concat([dconv4, conv2], -1)
#     # [3, 13545, 3, 256]
#     conc4 = tf.nn.conv2d(conc4, filter=weights['W_conc4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc4']
#     # [3, 13545, 3, 128]
#
#     base_shape = conv1.get_shape().as_list()
#     output_shape = [batch_size, base_shape[1], base_shape[2], 128]
#     dconv5 = tf.nn.conv2d_transpose(conc4, filter=weights['W_dconv5'], output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + biases['b_dconv5']
#     dconv5 = tf.layers.batch_normalization(dconv5, training=True)
#     # dconv5 = tf.contrib.layers.group_norm(dconv5, groups=8)
#     dconv5 = tf.nn.relu(dconv5)
#     # [3, 13545, 3, 128]
#
#     conc5 = tf.concat([dconv5, conv1], -1)
#     # [3, 13545, 3, 256]
#     conc5 = tf.nn.conv2d(conc5, filter=weights['W_conc5'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conc5']
#     # [3, 13545, 3, 128]
#
#     conv7 = tf.nn.conv2d(conc5, filter=weights['W_conv7'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv7']
#     # [3, 13545, 3, 25]
#
#     output = tf.nn.conv2d(conv7, filter=weights['W_out'], strides=[1, 1, 1, 1], padding='VALID')
#     # [3, 13545, 1, 25]
#
#     output = tf.reshape(output, [batch_size, 13545, 25])
#     # [3, 13545, 25]
#
#     return output
