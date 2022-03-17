from moviepy.editor import ImageSequenceClip
from source_bb.pc_utilities import load_pointcloud, calib_matrices, ransac, inside_the_box, draw_box_on_image, get_view_point_cloud
from source_bb.tf_utilities_deeper import get_model
from source_bb.utilities import read_calib_file
from sklearn.cluster import DBSCAN
import tensorflow as tf
import random
import numpy as np
import cv2 as cv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

VIDEO_IMM = '/Users/mattiafucili/Downloads/2011_09_26_0059/2011_09_26_drive_0059_sync/image_02/data/'
VIDEO_DATA = '/Users/mattiafucili/Downloads/2011_09_26_0059/2011_09_26_drive_0059_sync/velodyne_points/data/'
RES_IMM = BASE_DIR + '/video_pres/'
CKPT = BASE_DIR + '/ckpt/'
CLIP = []

VIDEO_FILES = os.listdir(VIDEO_DATA)
try:
    index = VIDEO_FILES.index('.DS_Store')
    del VIDEO_FILES[index]
except ValueError:
    pass
VIDEO_FILES.sort()


def check_rect(box):
    def ang(v_1, v_2, v_3, alpha):
        vA = [v_2[0] - v_1[0], v_2[1] - v_1[1], v_2[2] - v_1[2]]
        vB = [v_3[0] - v_1[0], v_3[1] - v_2[1], v_3[2] - v_1[2]]
        vec_prod = vA[0] * vB[0] + vA[1] * vB[1] + vA[2] * vB[2]
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

    return ang(v1, v2, v5, 50) & ang(v2, v1, v6, 50) & ang(v2, v3, v6, 50) & ang(v3, v2, v7, 50) & ang(v3, v4, v7, 50) & ang(v4, v3, v8, 50) & ang(v4, v1, v8, 50) & \
           ang(v1, v4, v5, 50) & ang(v1, v2, v4, 50) & ang(v2, v1, v3, 50) & ang(v3, v2, v4, 50) & ang(v4, v3, v1, 50) & ang(v5, v1, v6, 50) & ang(v6, v2, v7, 50) &\
           ang(v6, v2, v5, 50) & ang(v7, v6, v3, 50) & ang(v7, v3, v8, 50) & ang(v8, v7, v4, 50) & ang(v8, v4, v5, 50) & ang(v5, v1, v8, 50) & ang(v5, v6, v8, 50) &\
           ang(v6, v5, v7, 50) & ang(v7, v6, v8, 50) & ang(v8, v7, v5, 50)


epoch_to_restore = 220

with tf.Session() as sess:
    pointcloud_pl = tf.placeholder(tf.float32, shape=(1, 13545, 3))
    dropout = tf.placeholder(tf.float32)
    model = get_model(pointcloud_pl, 1, False, dropout)

    saver = tf.train.Saver()
    saver.restore(sess, CKPT + 'model.ckpt-{}'.format(epoch_to_restore))
    print('\nModel restored')

    for frame in VIDEO_FILES:
        file_number = os.path.splitext(frame)[0]

        point_cloud = load_pointcloud(VIDEO_DATA + '{}.bin'.format(file_number))[:, :3]
        calib = read_calib_file('calib_pres.txt')
        img = cv.imread(VIDEO_IMM + '{}.png'.format(file_number))
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

        predict = sess.run([model], feed_dict={pointcloud_pl: np.expand_dims(to_feed, 0), dropout: 0})
        predict = np.array(predict[0])

        cls_ = np.round(tf.sigmoid(predict[:, :, :1]).eval())
        box_p = predict[:, :, 1:]

        cls_ = cls_[0]
        box_p = box_p[0]

        mask_point = np.where(cls_ == 1)

        point_predicted = to_feed[mask_point[0]]
        box_predicted = box_p[mask_point[0]]

        clusters = DBSCAN(eps=0.7, min_samples=10).fit(point_predicted)
        labels = set(clusters.labels_)
        core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
        core_samples_mask[clusters.core_sample_indices_] = True

        r0, v2c, _, cam, T_w2c = calib_matrices(calib)

        for k in labels:
            if k != -1:
                class_member_mask = (clusters.labels_ == k)
                car_points = point_predicted[class_member_mask & core_samples_mask]
                car_box = box_predicted[class_member_mask & core_samples_mask]
                if len(car_points) > 50:
                    final = np.add(np.repeat(car_points, 8, axis=1), car_box)
                    final_bbox = []
                    ransac_res = []
                    for start in range(8):
                        ransac_res.append(ransac(final[:, start::8], 10, 0.75))

                    for index in range(8):
                        final_bbox.append(np.median(ransac_res[index], axis=0))

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

                    draw_box_on_image(img, final_bbox_2d)

        file_path = RES_IMM + '{}.jpeg'.format(file_number)
        CLIP += [file_path]
        cv.imwrite(file_path, img)

print 'Making GIF...'
clip = ImageSequenceClip(CLIP, fps=11)
clip.write_gif(RES_IMM + 'video_pres.gif', fps=11)
