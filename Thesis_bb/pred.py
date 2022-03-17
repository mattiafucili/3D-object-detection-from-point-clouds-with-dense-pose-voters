import os
import sys
import tensorflow as tf
import numpy as np
from source_bb.tf_utilities_deeper import get_model
from source_bb.pc_utilities import load_pointcloud_with_bboxes_info, calib_matrices, ransac, inside_the_box, draw_box_on_image
from source_bb.utilities import read_calib_file, parse_file
import random
from sklearn.cluster import DBSCAN
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
EVAL_DATA = '/Users/mattiafucili/Desktop/eval/'
CKPT = BASE_DIR + '/ckpt/'
IMG = '/Users/mattiafucili/PycharmProjects/point_data/images/training/image_2/'
RES = BASE_DIR + '/res_detection/'
RES_IMG = BASE_DIR + '/res_detection_img/'
CALIB = BASE_DIR + '/calib/'
LABELS = '/Users/mattiafucili/PycharmProjects/point_data/labels/'

EVAL_FILES = os.listdir(EVAL_DATA)
try:
    index = EVAL_FILES.index('.DS_Store')
    del EVAL_FILES[index]
except ValueError:
    pass


def angle_bev(p):
    car = np.asarray([[0, 0, 1], [0, 0, 0]])
    car = (np.matmul(p[:3, :3], car.T) + p[:3, 3:4]).T[:, ::2]
    car_dir = car[0] - car[1]

    angle = np.arccos(np.dot(car_dir, (1, 0)))

    if car_dir[1] > 0:
        angle *= -1

    alpha = angle - np.arctan2(car[0, 0], car[0, 1])
    return angle, alpha


epoch_to_restore = 220

with tf.Session() as sess:
    pointcloud_pl = tf.placeholder(tf.float32, shape=(1, 13545, 3))
    dropout = tf.placeholder(tf.float32)
    model = get_model(pointcloud_pl, 1, False, dropout)

    saver = tf.train.Saver()
    saver.restore(sess, CKPT + 'model.ckpt-{}'.format(epoch_to_restore))
    print('\nModel restored')

    for f in EVAL_FILES:
        file_number = os.path.splitext(f)[0]

        point_cloud = load_pointcloud_with_bboxes_info(EVAL_DATA + '{}.bin'.format(file_number))[:, :3]

        num_to_remove = len(point_cloud) - 13545
        indices = set()
        while len(indices) < num_to_remove:
            indices.add(random.randint(0, len(point_cloud - 1)))
        to_remove = random.sample(list(indices), num_to_remove)
        dataset = np.delete(point_cloud, to_remove, axis=0)
        to_feed = dataset[:13545, :3]

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

        calib = read_calib_file(CALIB + '{}.txt'.format(file_number))
        r0, v2c, _, cam, T_w2c = calib_matrices(calib)

        img = cv.imread(IMG + '{}.png'.format(file_number))

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
                    final_bbox_V = np.stack(final_bbox)

                    final_bbox = np.matmul(v2c[:3, :3], final_bbox_V.T).T + v2c[:3, 3]
                    final_bbox = np.matmul(T_w2c[:3, :3], final_bbox.T).T + T_w2c[:3, 3]

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
                        conf = np.round(np.mean(car_confs), 2)
                        img_height, img_width, _ = img.shape
                        xmin = np.round(np.clip(np.float32(np.min(final_bbox_2d[0])), 0, img_width), 2)
                        xmax = np.round(np.clip(np.float32(np.max(final_bbox_2d[0])), 0, img_width), 2)
                        ymin = np.round(np.clip(np.float32(np.min(final_bbox_2d[1])), 0, img_height), 2)
                        ymax = np.round(np.clip(np.float32(np.max(final_bbox_2d[1])), 0, img_height), 2)

                        filepath = RES + '{}.txt'.format(file_number)
                        if not os.path.exists(os.path.dirname(filepath)):
                            try:
                                os.makedirs(os.path.dirname(filepath))
                            except:
                                print('Error')
                        with open(filepath, 'a') as writer:
                            writer.write('Car -1 -1 {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(np.round(export_alpha, 2), xmin, ymin, xmax, ymax,
                                                                                                     np.round(height, 2), np.round(width, 2), np.round(length, 2),
                                                                                                     np.round(pose[0, 3], 2), np.round(pose[1, 3], 2),
                                                                                                     np.round(pose[2, 3], 2), np.round(export_angle, 2), conf))
                        cv.line(img, (xmin, ymin), (xmin, ymax), color=(0, 0, 255), thickness=2)
                        cv.line(img, (xmin, ymin), (xmax, ymin), color=(0, 0, 255), thickness=2)
                        cv.line(img, (xmax, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                        cv.line(img, (xmin, ymax), (xmax, ymax), color=(0, 0, 255), thickness=2)

        gt_labels = parse_file(LABELS + '{}.txt'.format(file_number), ' ')
        for gt_label in gt_labels:
            if gt_label[0] == 'Car' or gt_label[0] == 'Van':
                xmin = int(float(gt_label[4]))
                ymin = int(float(gt_label[5]))
                xmax = int(float(gt_label[6]))
                ymax = int(float(gt_label[7]))
                cv.line(img, (xmin, ymin), (xmin, ymax), color=(0, 255, 0), thickness=2)
                cv.line(img, (xmin, ymin), (xmax, ymin), color=(0, 255, 0), thickness=2)
                cv.line(img, (xmax, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                cv.line(img, (xmin, ymax), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv.imwrite(RES_IMG + '{}.jpeg'.format(file_number), img)
