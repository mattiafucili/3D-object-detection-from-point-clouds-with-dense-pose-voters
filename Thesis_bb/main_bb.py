# from source_bb.tf_utilities import train_gpu, eval_gpu
# # threw away 000000 004782
#
# train_gpu(3, 13545, 250, True, 310, False)
# eval_gpu(3, 13545, 250)

''' Test not in tuple '''
# import numpy as np
#
# a = []
# c = np.array([10, 10, 10])
# for i in range(20):
#     b = []
#     b.append(i)
#     b.append(i)
#     b.append(i)
#     a.append(b)
# f = []
# f.append(1)
# a.append(f)
# for index in range(len(a)):
#     k = a[index]
# p = [tuple(el) for el in a]
# if tuple(np.expand_dims(c, -1)) in p:
#     print 'a'


''' Test bbox ransac '''
# c = []
# a = []
#
#
# def aa(x):
#     res = []
#     for i in range(10):
#         res.append(x)
#     return res
#
#
# for i in range(10):
#     a.append(aa(i))
#
# a = np.stack(a)
# print len(a)
# print len(c)
# if len(c) < len(a):
#     c = a
# print len(c)

''' Test cls eval '''
# import numpy as np
# import math
#
#
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
#
#
# a = [0, 1, 0.1, 0.4, 0.51]
# for el in a:
#     print np.round(sigmoid(el))

''' Test cls eval 2 '''
# a = tf.constant([1, 0.1, -1.2, -0.002, 0.1])
# b = tf.constant([0.1, -0.44, 0.001, -0.102, 1])
# c = tf.add(a, b)
#
# with tf.Session() as sess:
#     res = sess.run([c])
#     k = tf.sigmoid(res[0]).eval()
#     print res
#     print k
#     print np.round(k)

''' Test cls '''
# import numpy as np
# a = [0, 1, 0, 0, 2, 1]
# b = [0, 0, 0, 0, 1, 0]
# c = np.equal(np.clip(a, 0, 1), 1)
# d = np.equal(b, 1)
# e = np.logical_and(c, d)
# e = e.astype(np.float32)
# f = np.sum(e) / 6 * 100
# print f

''' New safe subsample '''
# from source_bb.pc_utilities import load_pointcloud_with_bboxes_info
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from mpl_toolkits.mplot3d import axes3d, Axes3D
#
# a = load_pointcloud_with_bboxes_info('000046.bin')
# num_to_remove = len(a) - 13545
# objects = a[:, 3]
# # objects_list = np.unique(objects)
# # res_list = []
# # for obj in objects_list:
# #     num = len(objects[np.where(objects == obj)])
# #     res_list.append(num)
# res = np.where(objects == 0)
# to_remove = random.sample(list(res[0]), num_to_remove)
# final = np.delete(a, to_remove, axis=0)
# # objects2 = final[:, 3]
# # objects2_list = np.unique(objects2)
# # res_list2 = []
# # for obj in objects2_list:
# #     num = len(objects2[np.where(objects2 == obj)])
# #     res_list2.append(num)
#
# # b = np.array(random.sample(list(a), 4000))
# # c = a[:4000]
# f = plt.figure(figsize=(15, 8))
# ax = f.add_subplot(111, projection='3d')
# # ax.scatter(*np.transpose(a[:, [0, 1, 2]]), s=1, c='#FF0000', cmap='gray')
# ax.scatter(*np.transpose(final[:, [0, 1, 2]]), s=1, c='#FF0000', cmap='gray')
# plt.show()
