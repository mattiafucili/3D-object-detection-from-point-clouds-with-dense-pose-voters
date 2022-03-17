import numpy as np
from source_bb.pc_utilities import draw_vectors, draw_box, ransac
from source_bb.tf_utilities_deeper import get_model
from sklearn.cluster import DBSCAN
import tensorflow as tf
import random
import os
import sys
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

''' Display points '''
# def load_pointcloud_video(filename):
#     return np.fromfile(filename, dtype=np.float32).reshape((-1, 28))
#
#
# VIDEO_DATA = BASE_DIR + '/video_files/'
# CKPT = BASE_DIR + '/ckpt/'
# VIDEO_IMAGES = BASE_DIR + '/res_images/'
#
# epoch_to_restore = 240
#
# with tf.Session() as sess:
#     pointcloud_pl = tf.placeholder(tf.float32, shape=(1, 13545, 3))
#     dropout = tf.placeholder(tf.float32)
#     model = get_model(pointcloud_pl, 1, False, dropout)
#
#     saver = tf.train.Saver()
#     saver.restore(sess, CKPT + 'model.ckpt-{}'.format(epoch_to_restore))
#     print('\nModel restored')
#
#     point_cloud = load_pointcloud_video(VIDEO_DATA + '000147.bin')
#     num_to_remove = len(point_cloud) - 13545
#     indices = set()
#     while len(indices) < num_to_remove:
#         indices.add(random.randint(0, len(point_cloud - 1)))
#     to_remove = random.sample(list(indices), num_to_remove)
#     dataset = np.delete(point_cloud, to_remove, axis=0)
#
#     to_feed = dataset[:13545, :3]
#     y = dataset[:, 3:]
#
#     predict = sess.run([model], feed_dict={pointcloud_pl: np.expand_dims(to_feed, 0), dropout: 0})
#     predict = np.array(predict[0])
#
#     cls_predict = np.round(tf.sigmoid(predict[:, :, :1]).eval())
#     cls_predict = cls_predict[0]
#
#     p = np.where(cls_predict == 1)
#
#     pred = to_feed[p[0]]
#
#     clusters = DBSCAN(eps=0.7, min_samples=10).fit(pred)
#     labels = set(clusters.labels_)
#     core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
#     core_samples_mask[clusters.core_sample_indices_] = True
#     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
#
#     axes_limits = [
#         [-10, 30],  # X axis range
#         [-20, 20],  # Y axis range
#         [-3, 10]  # Z axis range
#     ]
#
#     f = plt.figure(figsize=(15, 8))
#     ax = f.add_subplot(111, projection='3d')
#     ax.view_init(elev=60)
#     ax.set_xlim3d(*axes_limits[0])
#     ax.set_ylim3d(*axes_limits[1])
#     ax.set_zlim3d(*axes_limits[2])
#     ax.set_xlim3d((-10, 30))
#
#     # ax.scatter(*np.transpose(to_feed[:, [0, 1, 2]]), s=0.02, cmap='gray')
#     # ax.scatter(*np.transpose(pred[:, [0, 1, 2]]), s=0.1, color='red', cmap='gray')
#     for k, col in zip(labels, colors):
#         if k == -1:
#             col = [0, 0, 0, 1]
#
#         class_member_mask = (clusters.labels_ == k)
#         res = pred[class_member_mask & core_samples_mask]
#         if len(res) > 50:
#             ax.scatter(*np.transpose(res[:, [0, 1, 2]]), s=0.1, color=tuple(col), cmap='gray')
#     plt.savefig('clusters.png')

''' Display npy files '''
# FILES = os.listdir('npy_files')
# FILES.sort()
#
# for i in xrange(len(FILES) / 5):
#     to_plot = FILES[i * 5: (i + 1) * 5]
#     final = np.load('npy_files/{}'.format(to_plot[0]))
#     gt = np.load('npy_files/{}'.format(to_plot[1]))
#     points = np.load('npy_files/{}'.format(to_plot[2]))
#     points_gt = np.load('npy_files/{}'.format(to_plot[3]))
#     predict = np.load('npy_files/{}'.format(to_plot[4]))
#
#     prefix = "{}".format(os.path.splitext(to_plot[0]))
#     prefix = prefix.split('_')
#     _, _, prefix_1 = prefix[0]
#
#     f = plt.figure(figsize=(15, 8))
#     ax = f.add_subplot(111, projection='3d')
#     ax.scatter(*np.transpose(points), s=3, c='#FF0000', cmap='gray')
#     ax.scatter(*np.transpose(points_gt), s=3, c='#00FF00', cmap='gray')
#     draw_box(ax, gt, 'green')
#     draw_box(ax, predict, 'blue')
#     draw_box(ax, final, 'red')
#
#     filepath = '60/7/{}_{}.png'.format(prefix_1, prefix[1])
#     if not os.path.exists(os.path.dirname(filepath)):
#         try:
#             os.makedirs(os.path.dirname(filepath))
#         except:
#             print('Error')
#     plt.savefig(filepath)
#     plt.close()


''' CLS plot '''
# cls = np.fromfile('imm_cls_tesi.txt', sep='\n')
#
# plt.title('Classification accuracy {} epochs'.format(len(cls)))
# plt.plot(range(len(cls)), cls, '-r')
# plt.show()
#
# cls_schifo = np.fromfile('imm_cls_tesi_schifo.txt', sep='\n')
#
# plt.title('Classification accuracy {} epochs'.format(len(cls_schifo)))
# plt.plot(range(len(cls_schifo)), cls_schifo, '-r')
# plt.show()

''' Test ransac one vertex '''
# tot_point = []
# tot_vector = []
# for i in xrange(8):
#     tot_point.append(np.reshape(np.load('2_7_vec-{}_points.npy'.format(i)), [-1, 3]))
#     tot_vector.append(np.reshape(np.load('2_7_vec-{}.npy'.format(i)), [-1, 3]))
#
# box = np.reshape(np.load('2_7_vec-0_box.npy'), [3, 8])
# tot_point = np.array(tot_point)
# tot_vector = np.array(tot_vector)
#
# f = plt.figure(figsize=(15, 8))
# ax = f.add_subplot(111, projection='3d')
# ax.scatter(*np.transpose(tot_point), s=3, c='#00FF00', cmap='gray')
# ax.plot([box[0, 0]], [box[1, 0]], [box[2, 0]], 'bo')
# for i in xrange(8):
#     draw_vectors(tot_point, tot_vector[i], ax)
# draw_box(ax, box, 'green')
# plt.show()
#
# ransac_res = []
# final_bbox = []
# ransac_res.append(ransac(vectors, 10, 0.75))
#
# final_bbox.append(np.median(ransac_res[0], axis=0))

''' Error plots '''
# ERROR_DIR = BASE_DIR + '/error_files2/'
#
# error_normal = np.fromfile(ERROR_DIR + 'error_normal.txt', sep='\n')
# error_flip = np.fromfile(ERROR_DIR + 'error_flip.txt', sep='\n')
# error_noise = np.fromfile(ERROR_DIR + 'error_noise.txt', sep='\n')
# error_eval = np.fromfile(ERROR_DIR + 'error_eval.txt', sep='\n')
# cls = np.fromfile(ERROR_DIR + 'cls.txt', sep='\n')
# cls_1 = np.fromfile(ERROR_DIR + 'cls_1.txt', sep='\n')
# rec = np.fromfile(ERROR_DIR + 'rec_1.txt', sep='\n')
#
# num = len(error_normal) / 712
# error_normal = error_normal[:(num * 712)]
# error_flip = error_flip[:(num * 712)]
# error_noise = error_noise[:(num * 712)]
# cls = cls[:(num * 8)]
# cls_1 = cls_1[:(num * 8)]
# rec = rec[:(num * 8)]
#
# res_normal = []
# res_flip = []
# res_noise = []
# res_cls = []
# res_cls_1 = []
# res_rec = []
#
# for i in xrange(num):
#     sub = error_normal[(i * 712):((i + 1) * 712)]
#     res_normal.append(np.median(sub))
#
#     sub = error_flip[(i * 712):((i + 1) * 712)]
#     res_flip.append(np.median(sub))
#
#     sub = error_noise[(i * 712):((i + 1) * 712)]
#     res_noise.append(np.median(sub))
#
#     sub = cls[(i * 8):((i + 1) * 8)]
#     res_cls.append(np.median(sub))
#
#     sub = cls_1[(i * 8):((i + 1) * 8)]
#     res_cls_1.append(np.median(sub))
#
#     sub = rec[(i * 8):((i + 1) * 8)]
#     res_rec.append(np.median(sub))
#
# res_normal = np.array(res_normal)
# res_flip = np.array(res_flip)
# res_noise = np.array(res_noise)
# res_cls = np.array(res_cls)
# res_cls_1 = np.array(res_cls_1)
# res_rec = np.array(res_rec)
#
# plt.title('Error normal {} epochs'.format(num))
# plt.plot(range(num), res_normal, '-r')
# plt.savefig(ERROR_DIR + 'error_normal.png')
# plt.close()
#
# plt.title('Error flip {} epochs'.format(num))
# plt.plot(range(num), res_flip, '-r')
# plt.savefig(ERROR_DIR + 'error_flip.png')
# plt.close()
#
# plt.title('Error noise {} epochs'.format(num))
# plt.plot(range(num), res_noise, '-r')
# plt.savefig(ERROR_DIR + 'error_noise.png')
# plt.close()
#
# plt.title('Error eval {} epochs'.format(num))
# plt.plot(range(num), error_eval, '-r')
# plt.savefig(ERROR_DIR + 'error_eval.png')
# plt.close()
#
# plt.title('CLS {} epochs'.format(num))
# plt.plot(range(num), res_cls, '-r')
# plt.savefig(ERROR_DIR + 'cls.png')
# plt.close()
#
# plt.title('CLS 1 {} epochs'.format(num))
# plt.plot(range(num), res_cls_1, '-r')
# plt.savefig(ERROR_DIR + 'cls_1.png')
# plt.close()
#
# plt.title('PREC 1 {} epochs'.format(num))
# plt.plot(range(num), res_rec, '-r')
# plt.savefig(ERROR_DIR + 'rec_1.png')
# plt.close()

''' Vectors '''
# points = np.reshape(np.load('0_1_vec-0_points.npy'), [-1, 3])
# vectors = np.reshape(np.load('0_1_vec-0.npy'), [-1, 3])
# box = np.reshape(np.load('0_1_vec-0_box.npy'), [3, 8])
#
# f = plt.figure(figsize=(15, 8))
# ax = f.add_subplot(111, projection='3d')
# ax.scatter(*np.transpose(points[:, [0, 1, 2]]), s=3, c='#00FF00', cmap='gray')
# draw_vectors(points, vectors, ax)
# draw_box(ax, box, 'green')
# plt.show()
