import tensorflow as tf
from source_bb.tf_utilities_deeper import get_model
from source_bb.pc_utilities import ransac, draw_box, draw_vectors
import numpy as np
from sklearn.cluster import DBSCAN
from moviepy.editor import ImageSequenceClip
import os
import sys
import random
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d


def load_pointcloud_video(filename):
    return np.fromfile(filename, dtype=np.float32).reshape((-1, 28))


def isclose(a, b, tol=0.1):
    return abs(a - b) <= tol


BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

VIDEO_DATA = BASE_DIR + '/video_files/'
CKPT = BASE_DIR + '/ckpt/'
VIDEO_IMAGES = BASE_DIR + '/res_images/'

VIDEO_FILES = os.listdir(VIDEO_DATA)
try:
    index = VIDEO_FILES.index('.DS_Store')
    del VIDEO_FILES[index]
except ValueError:
    pass
VIDEO_FILES.sort()

epoch_to_restore = 220
frames = []

with tf.Session() as sess:
    pointcloud_pl = tf.placeholder(tf.float32, shape=(1, 13545, 3))
    dropout = tf.placeholder(tf.float32)
    model = get_model(pointcloud_pl, 1, False, dropout)

    saver = tf.train.Saver()
    saver.restore(sess, CKPT + 'model.ckpt-{}'.format(epoch_to_restore))
    print('\nModel restored')

    for file_name in VIDEO_FILES:
        point_cloud = load_pointcloud_video(VIDEO_DATA + file_name)
        num_to_remove = len(point_cloud) - 13545
        indices = np.where(point_cloud[:, 3] == 0)
        to_remove = random.sample(list(indices[0]), num_to_remove)
        dataset = np.delete(point_cloud, to_remove, axis=0)

        to_feed = dataset[:, :3]
        y = dataset[:, 3:]

        predict = sess.run([model], feed_dict={pointcloud_pl: np.expand_dims(to_feed, 0), dropout: 0})
        predict = np.array(predict[0])

        # cls_ = np.round(tf.sigmoid(predict[:, :, :1]).eval())
        cls_ = tf.sigmoid(predict[:, :, :1]).eval()
        cls_ = np.array(cls_[0])
        cls_[cls_ > 0.8] = 1
        cls_[cls_ <= 0.8] = 0
        cls_y = y[:, :1]
        box_p = predict[:, :, 1:]
        box_y = y[:, 1:]

        # cls_ = cls_[0]
        box_p = box_p[0]

        ''' No labels '''
        mask_point = np.where(cls_ == 1)

        point_predicted = to_feed[mask_point[0]]
        box_predicted = box_p[mask_point[0]]

        clusters = DBSCAN(eps=0.7, min_samples=10).fit(point_predicted)
        labels = set(clusters.labels_)
        core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
        core_samples_mask[clusters.core_sample_indices_] = True
        ''' No labels '''

        axes_limits = [
            [-10, 30],  # X axis range
            [-20, 20],  # Y axis range
            [-3, 10]  # Z axis range
        ]

        f = plt.figure(figsize=(15, 8))
        ax = Axes3D(f)
        ax.view_init(elev=90, azim=180)
        ax.set_xlim3d(*axes_limits[0])
        ax.set_ylim3d(*axes_limits[1])
        ax.set_zlim3d(*axes_limits[2])
        ax.set_xlim3d((-10, 30))
        plt.axis('off')

        ax.scatter(*np.transpose(to_feed[:, [0, 1, 2]]), s=0.02, cmap='gray')

        ''' No labels'''
        for k in labels:
            if k != -1:
                class_member_mask = (clusters.labels_ == k)
                car_points = point_predicted[class_member_mask & core_samples_mask]
                car_box = box_predicted[class_member_mask & core_samples_mask]
                if len(car_points) > 50:
                    # if not isclose(np.median(car_box), 0.0):
                    final = np.add(np.repeat(car_points, 8, axis=1), car_box)
                    final_bbox = []
                    ransac_res = []
                    for start in range(8):
                        ransac_res.append(ransac(final[:, start::8], 10, 0.75))

                    for index in range(8):
                        final_bbox.append(np.median(ransac_res[index], axis=0))
                    final_bbox = np.stack(final_bbox)
                    final_bbox = np.transpose(final_bbox)
                    draw_box(ax, final_bbox, 'red')
        ''' No labels '''

        ''' Labels '''
        # max_batch = int(np.max(cls_y))
        # for i_object in range(max_batch):
        #     box_gt = box_y[np.where(cls_y == i_object + 1), :]
        #     box_gt = box_gt[0]
        #     box_gt = np.median(box_gt, axis=0)
        #     box_gt = np.reshape(box_gt, [3, 8])
        #
        #     mask = tf.cast(tf.equal(cls_y, i_object + 1), tf.int32).eval()
        #     mask_box = np.tile(mask, [1, 24])
        #     mask_points = np.tile(mask, [1, 3])
        #     box = np.round(np.multiply(box_p, mask_box), 3)
        #     box = np.round(np.multiply(box, cls_), 3)
        #     points = np.round(np.multiply(to_feed, mask_points), 3)
        #     points = np.round(np.multiply(points, cls_), 3)
        #
        #     vectors = [i for i in box if any(i + 0)]
        #     points_ = [i for i in points if any(i + 0)]
        #
        #     if len(vectors) > 0:
        #         vectors = np.round(vectors, 3)
        #     if len(points_) > 0:
        #         points_ = np.round(points_, 3)
        #
        #     box_ = []
        #     if len(vectors) > 0 and len(points_) > 0:
        #         points_ = np.repeat(points_, 8, axis=1)
        #         box_ = np.add(points_, vectors)
        #
        #     ''' Boxes '''
        #     # final_bbox = []
        #     # if len(vectors) > 0 and len(points_) > 0:
        #     #     ransac_res = []
        #     #     for start in range(8):
        #     #         ransac_res.append(ransac(box_[:, start::8], 10, 0.75))
        #     #
        #     #     for index in range(8):
        #     #         final_bbox.append(np.median(ransac_res[index], axis=0))
        #     #     final_bbox = np.stack(final_bbox)
        #     #     final_bbox = np.transpose(final_bbox)
        #
        #     ''' Vectors '''
        #     # draw_box(ax, box_gt, 'green')
        #     # # if len(final_bbox) > 0:
        #     # #     draw_box(ax, final_bbox, 'red')
        #     # if len(points_) > 0:
        #     #     draw_vectors(points_[:10, 0::8], box_[:10, 0::8], ax)
        ''' Labels '''

        filepath = VIDEO_IMAGES + '{}.png'.format(os.path.splitext(file_name)[0])
        if not os.path.exists(os.path.dirname(filepath)):
            try:
                os.makedirs(os.path.dirname(filepath))
            except:
                print('Error')
        plt.savefig(filepath)
        frames += [filepath]
        plt.close()

print 'Making GIF...'
clip = ImageSequenceClip(frames, fps=5)
clip.write_gif(VIDEO_IMAGES + 'video.gif', fps=5)
