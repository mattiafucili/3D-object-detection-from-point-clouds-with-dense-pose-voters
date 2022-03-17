import tensorflow as tf
from source_bb.tf_utilities_deeper import get_model, bb_loss
from source_bb.pc_utilities import ransac, draw_box, calib_matrices, expand_matrix, expand_box, expand_matrix2, draw_box_on_image,\
    load_pointcloud, get_view_point_cloud, inside_the_box
from source_bb.utilities import read_calib_file
import cv2 as cv
import time
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


def angle_bev(p):
    car = np.asarray([[0, 0, 1], [0, 0, 0]])
    car = (np.matmul(p[:3, :3], car.T) + p[:3, 3:4]).T[:, ::2]
    car_dir = car[0] - car[1]

    angle = np.arccos(np.dot(car_dir, (1, 0)))

    if car_dir[1] > 0:
        angle *= -1

    alpha = angle - np.arctan2(car[0, 0], car[0, 1])
    return angle, alpha


def check_rect(box):
    def ang(v_1, v_2, v_3, alpha):
        vA = [v_2[0] - v_1[0], v_2[1] - v_1[1], v_2[2] - v_1[2]]
        vB = [v_3[0] - v_1[0], v_3[1] - v_2[1], v_3[2] - v_1[2]]
        vec_prod = vA[0] * vB[0] + vA[1] * vB[1] + vA[2] * vA[2]
        normA = np.sqrt(vA[0] ** 2 + vA[1] ** 2 + vA[2] ** 2)
        normB = np.sqrt(vB[0] ** 2 + vB[1] ** 2 + vB[2] ** 2)
        teta = np.arccos(vec_prod / (normA * normB))
        if teta < alpha:
            return True
        else:
            return False

    v1 = box[:, 0]
    v2 = box[:, 1]
    v3 = box[:, 2]
    v4 = box[:, 3]
    v5 = box[:, 4]
    v6 = box[:, 5]
    v7 = box[:, 6]
    v8 = box[:, 7]

    return ang(v1, v2, v5, 60) & ang(v2, v1, v6, 60) & ang(v2, v3, v6, 60) & ang(v3, v2, v7, 60) & ang(v3, v4, v7, 60) & ang(v4, v3, v8, 60) & ang(v4, v1, v8, 60) &\
           ang(v1, v4, v5, 60) & ang(v1, v2, v4, 60) & ang(v2, v1, v3, 60) & ang(v3, v2, v4, 60) & ang(v4, v3, v1, 60) & ang(v5, v1, v6, 60) & ang(v6, v2, v7, 60) &\
           ang(v6, v2, v5, 60) & ang(v7, v6, v3, 60) & ang(v7, v3, v8, 60) & ang(v8, v7, v4, 60) & ang(v8, v4, v5, 60) & ang(v5, v1, v8, 60) & ang(v5, v6, v8, 60) &\
           ang(v6, v5, v7, 60) & ang(v7, v6, v8, 60) & ang(v8, v7, v5, 60)


BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
CKPT = BASE_DIR + '/ckpt/'
CALIB = BASE_DIR + '/calib/'

epoch_to_restore = 220
file_number = "000095"

with tf.Session() as sess:
    pointcloud_pl = tf.placeholder(tf.float32, shape=(1, 13545, 3))
    dropout = tf.placeholder(tf.float32)
    model = get_model(pointcloud_pl, 1, False, dropout)

    saver = tf.train.Saver()
    saver.restore(sess, CKPT + 'model.ckpt-{}'.format(epoch_to_restore))
    print('\nModel restored')

    # point_cloud = load_pointcloud_video(BASE_DIR + '/{}.bin'.format(file_number))[:, :3]
    point_cloud = load_pointcloud(BASE_DIR + '/{}.bin'.format(file_number))
    calib = read_calib_file(BASE_DIR + '/{}.txt'.format(file_number))  # CALIB
    img = cv.imread(BASE_DIR + '/{}.png'.format(file_number))
    img_height, img_width, _ = img.shape
    point_cloud = get_view_point_cloud(point_cloud, img_width, img_height, calib)
    point_cloud = point_cloud[:, :3]

    num_to_remove = len(point_cloud) - 13545
    indices = set()
    while len(indices) < num_to_remove:
        indices.add(random.randint(0, len(point_cloud - 1)))
    to_remove = random.sample(list(indices), num_to_remove)
    dataset = np.delete(point_cloud, to_remove, axis=0)
    to_feed = dataset[:13545, :3]

    ''' Funziona Top '''
    # bottom = point_cloud[point_cloud[:, 2] < -1.6]
    # top = point_cloud[point_cloud[:, 2] > -1.6]
    # num_missing = 13545 - top.shape[0]
    # print num_missing
    # if num_missing > 0:
    #     to_feed = np.concatenate([top, np.array(random.sample(list(bottom), num_missing))], axis=0)
    # else:
    #     to_feed = np.array(random.sample(list(top), 13545))
    ''' Funziona '''

    ''' Bottom '''
    # bottom = point_cloud[point_cloud[:, 2] < 0.1]
    # top = point_cloud[point_cloud[:, 2] > 0.1]
    # num_missing = 13545 - bottom.shape[0]
    # print num_missing
    # if num_missing > 0:
    #     to_feed = np.concatenate([bottom, np.array(random.sample(list(top), num_missing))], axis=0)
    # else:
    #     to_feed = np.array(random.sample(list(bottom), 13545))
    ''' Bottom '''

    ''' Slice '''
    # bottom = point_cloud[point_cloud[:, 2] < 0]
    # sl = bottom[bottom[:, 2] > -1.6]
    # to_choose = bottom[bottom[:, 2] < -1.6]
    # num_missing = 13545 - sl.shape[0]
    # print num_missing
    # if num_missing > 0:
    #     to_add = np.array(random.sample(list(to_choose), num_missing))
    #     to_feed = np.concatenate([sl, to_add], axis=0)
    # else:
    #     to_feed = np.array(random.sample(list(sl), 13545))
    ''' Slice '''

    start_time = time.clock()
    predict = sess.run([model], feed_dict={pointcloud_pl: np.expand_dims(to_feed, 0), dropout: 0})
    predict = np.array(predict[0])

    confs = tf.sigmoid(predict[:, :, :1]).eval()
    cls_ = np.round(confs)
    box_p = predict[:, :, 1:]

    cls_ = cls_[0]
    confs = confs[0]
    box_p = box_p[0]

    mask_point = np.where(cls_ == 1)

    point_predicted = to_feed[mask_point[0]]
    confs_predicted = confs[mask_point[0]]
    box_predicted = box_p[mask_point[0]]

    clusters = DBSCAN(eps=0.7, min_samples=10).fit(point_predicted)
    labels = set(clusters.labels_)
    core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
    core_samples_mask[clusters.core_sample_indices_] = True

    img = cv.imread(BASE_DIR + '/{}.png'.format(file_number))

    axes_limits = [
        [-10, 30],  # X axis range
        [-20, 20],  # Y axis range
        [-3, 10]  # Z axis range
    ]

    f = plt.figure(figsize=(15, 8))
    ax = Axes3D(f)
    ax.view_init(elev=30, azim=180)
    ax.set_xlim3d(*axes_limits[0])
    ax.set_ylim3d(*axes_limits[1])
    ax.set_zlim3d(*axes_limits[2])
    ax.set_xlim3d((-10, 30))
    plt.axis('off')
    # ax.scatter(*np.transpose(sl[:, [0, 1, 2]]), s=0.02, c='red')
    # ax.scatter(*np.transpose(to_add[:, [0, 1, 2]]), s=0.02, c='blue')
    # ax.scatter(*np.transpose(bottom[:, [0, 1, 2]]), s=0.02, c='green')
    # ax.scatter(*np.transpose(to_feed[:, [0, 1, 2]]), s=0.02, c='green')

    calib = read_calib_file(BASE_DIR + '/{}.txt'.format(file_number))  # CALIB
    r0, v2c, _, cam, T_w2c = calib_matrices(calib)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    for k, col in zip(labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (clusters.labels_ == k)
        res = point_predicted[class_member_mask & core_samples_mask]
        if len(res) > 50:
            ax.scatter(*np.transpose(res[:, [0, 1, 2]]), s=0.1, color=tuple(col), cmap='gray')

    for k in labels:
        if k != -1:
            class_member_mask = (clusters.labels_ == k)
            car_points = point_predicted[class_member_mask & core_samples_mask]
            car_confs = confs_predicted[class_member_mask & core_samples_mask]
            car_box = box_predicted[class_member_mask & core_samples_mask]
            if len(car_points) > 50:
                final = np.add(np.repeat(car_points, 8, axis=1), car_box)
                final_bbox = []
                ransac_res = []
                for start in range(8):
                    ransac_res.append(ransac(final[:, start::8], 10, 0.75))

                for index in range(8):
                    final_bbox.append(np.median(ransac_res[index], axis=0))

                final_bbox_to_check = np.stack(final_bbox)
                final_bbox_to_check = np.transpose(final_bbox_to_check)

                # if not check_rect(final_bbox_to_check):
                # r0 = expand_matrix(r0)
                # v2c = expand_matrix2(v2c)
                final_bbox_V = np.stack(final_bbox)
                # final_bbox = np.transpose(final_bbox)
                # final_bbox = expand_box(final_bbox)

                # Box is already in the camera coordinate system
                final_bbox = np.matmul(v2c[:3, :3], final_bbox_V.T).T + v2c[:3, 3]
                final_bbox = np.matmul(T_w2c[:3, :3], final_bbox.T).T + T_w2c[:3, 3]

                # front (bottom) 1,2, (top) 5,6 back (bottom) 4,3 (top) 8, 7
                width = np.linalg.norm(np.mean([final_bbox[1], final_bbox[2], final_bbox[5], final_bbox[6]], axis=0) -
                                       np.mean([final_bbox[0], final_bbox[3], final_bbox[4], final_bbox[7]], axis=0))

                length = np.linalg.norm(np.mean([final_bbox[0], final_bbox[1], final_bbox[4], final_bbox[5]], axis=0) -
                                       np.mean([final_bbox[2], final_bbox[3], final_bbox[6], final_bbox[7]], axis=0))

                height = np.linalg.norm(np.mean([final_bbox[0], final_bbox[1], final_bbox[2], final_bbox[3]], axis=0) -
                                       np.mean([final_bbox[4], final_bbox[5], final_bbox[6], final_bbox[7]], axis=0))

                centroid = np.mean([final_bbox[0], final_bbox[1], final_bbox[2], final_bbox[3],
                                    final_bbox[4], final_bbox[5], final_bbox[6], final_bbox[7]], axis=0)

                bbox3d = np.asarray([[-width / 2, height / 2, length / 2],
                                    [width / 2, height / 2, length / 2],
                                    [width / 2, height / 2, -length / 2],
                                    [-width / 2, height / 2, -length / 2],
                                    [-width / 2, -height / 2, length / 2],
                                    [width / 2, -height / 2, length / 2],
                                    [width / 2, -height / 2, -length / 2],
                                    [-width / 2, -height / 2, -length / 2]])

                H = np.matmul(bbox3d.T, (final_bbox - centroid))

                U, S, Vt = np.linalg.svd(H)
                pose = np.eye(4)
                pose[:3, 3] = centroid
                pose[:3, :3] = np.matmul(Vt.T, U.T)

                # Check if reflection
                determinant = np.linalg.det(pose[:3, :3])
                if determinant < 0:
                    print('DET -1', centroid)
                    B = np.eye(3)
                    B[2, 2] = determinant
                    pose[:3, :3] = np.matmul(np.matmul(Vt.T, B), U.T)

                bbox3d_transformed = np.matmul(pose[:3, :3], bbox3d.T).T + pose[:3, 3]

                final_bbox_2d = np.matmul(cam, bbox3d_transformed.T).T
                final_bbox_2d = (final_bbox_2d[:, :2] / final_bbox_2d[:, 2:3])[:, :2]

                final_bbox_2d = np.int32(final_bbox_2d).T

                export_translation = np.mean([bbox3d_transformed[0], bbox3d_transformed[2], bbox3d_transformed[3], bbox3d_transformed[4]], axis=0)
                export_angle, export_alpha = angle_bev(pose)

                bbox3d_transformed = np.matmul(v2c[:3, :3].T, bbox3d_transformed.T).T - v2c[:3, 3]
                bbox3d_transformed = np.matmul(T_w2c[:3, :3].T, bbox3d_transformed.T).T - T_w2c[:3, 3]

                o = bbox3d_transformed[0]
                a = bbox3d_transformed[1]
                b = bbox3d_transformed[3]
                c = bbox3d_transformed[4]
                s = 0.0

                for point in car_points:
                    if inside_the_box(point, o, a, b, c):
                        s += 1.0

                if (s / len(car_points) * 100) > 10:
                    # conf = np.round(np.mean(car_confs), 2)
                    # xmin = np.round(np.clip(np.float32(np.min(final_bbox_2d[0])), 0, img_width), 2)
                    # xmax = np.round(np.clip(np.float32(np.max(final_bbox_2d[0])), 0, img_width), 2)
                    # ymin = np.round(np.clip(np.float32(np.min(final_bbox_2d[1])), 0, img_height), 2)
                    # ymax = np.round(np.clip(np.float32(np.max(final_bbox_2d[1])), 0, img_height), 2)

                    # filepath = BASE_DIR + '/59_{}.txt'.format(file_number)
                    # if not os.path.exists(os.path.dirname(filepath)):
                    #     try:
                    #         os.makedirs(os.path.dirname(filepath))
                    #     except:
                    #         print('Error')
                    # with open(filepath, 'a') as writer:
                    #     writer.write('Car -1 -1 {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(np.round(export_alpha, 2), xmin, ymin, xmax, ymax,
                    #                                                                              np.round(height, 2), np.round(width, 2), np.round(length, 2),
                    #                                                                              np.round(pose[0, 3], 2), np.round(pose[1, 3], 2),
                    #                                                                              np.round(pose[2, 3], 2), np.round(export_angle, 2), conf))
                    draw_box_on_image(img, final_bbox_2d)
                    draw_box(ax, bbox3d_transformed.T, 'red')

    print('seconds {}'.format(time.clock() - start_time))
    plt.savefig('3d_39_{}.jpeg'.format(file_number))
    cv.imwrite('2d_39_{}.jpeg'.format(file_number), img)
