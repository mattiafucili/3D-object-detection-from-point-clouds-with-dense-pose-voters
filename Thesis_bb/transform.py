import os
import numpy as np
from source_bb.pc_utilities import load_pointcloud_with_bboxes_info


''' Remove cars with less than 50 points '''
# FILES = os.listdir('/Users/mattiafucili/Desktop/training')
# try:
#     index = FILES.index('.DS_Store')
#     del FILES[index]
# except ValueError:
#     pass
# FILES_DONE = os.listdir('/Users/mattiafucili/Desktop/training_res')
# try:
#     index = FILES_DONE.index('.DS_Store')
#     del FILES_DONE[index]
# except ValueError:
#     pass
# progress = 1
#
# for f in FILES:
#     print '{}/{}'.format(progress, len(FILES))
#     progress += 1
#     if f not in FILES_DONE:
#         res = []
#         pointcloud = load_pointcloud_with_bboxes_info('/Users/mattiafucili/Desktop/training/' + f)
#         objects = pointcloud[:, 3]
#         objects_list = np.unique(objects)
#         for obj in objects_list:
#             num = len(objects[np.where(objects == obj)])
#             if num < 50:
#                 res.append(obj)
#         if len(res) != 0:
#             with open('res_points.txt', 'a') as ff:
#                 ff.write('training/{}\n'.format(f))
#             with open('/Users/mattiafucili/Desktop/training_res/' + f, 'wb') as writer:
#                 for row in pointcloud:
#                     writer.write(np.float32(row[0]))
#                     writer.write(np.float32(row[1]))
#                     writer.write(np.float32(row[2]))
#                     if row[3] not in res:
#                         for index in range(3, 34):
#                             writer.write(np.float32(row[index]))
#                     else:
#                         for index in range(3, 34):
#                             writer.write(np.float32(0))

''' Remove file without objects '''
# FILES = os.listdir('/Users/mattiafucili/Desktop/training_res')
# try:
#     index = FILES.index('.DS_Store')
#     del FILES[index]
# except ValueError:
#     pass
#
# for f in FILES:
#     pointcloud = load_pointcloud_with_bboxes_info('/Users/mattiafucili/Desktop/training_res/' + f)
#     m = max(pointcloud[:, 3])
#     if m == 0:
#         with open('bin_points.txt', 'a') as writer:
#             writer.write('training_res/{}\n'.format(f))

''' Remove not interesting objects '''
# FILES = os.listdir('/Users/mattiafucili/Desktop/training')
#
# res_list = []
# bin_list = open('bin.txt').readlines()
# progress = 1
#
# for f in FILES:
#     print '{}/{}'.format(progress, len(FILES))
#     progress += 1
#     number = os.path.splitext(f)[0]
#     indices = []
#     if number not in bin_list:
#         if number == '000870':
#             print 'a'
#         try:
#             with open('/Users/mattiafucili/PycharmProjects/Thesis/labels/{}.txt'.format(number)) as r:
#                 cont = 1
#                 ok = 0
#                 lines = r.readlines()
#                 for line in lines:
#                     if 'Person' in line or 'Pedestrian' in line or 'Cyclist' in line or 'Misc' in line or 'Tram' in line or 'Truck' in line:
#                         indices.append(cont)
#                     if 'Car' in line or 'Van' in line:
#                         ok = 1
#                     if 'DontCare' in line:
#                         cont += 0
#                     else:
#                         cont += 1
#             if len(indices) != 0:
#                 if ok == 0:
#                     with open('bin.txt', 'a') as ff:
#                         ff.write('{}\n'.format(number))
#                 else:
#                     with open('res.txt', 'a') as ff:
#                         ff.write('training/{}.bin\n'.format(number))
#                     pointcloud = load_pointcloud_with_bboxes_info('/Users/mattiafucili/Desktop/training/{}.bin'.format(number))
#                     with open('/Users/mattiafucili/Desktop/training_res/{}.bin'.format(number), 'wb') as writer:
#                         for row in pointcloud:
#                             writer.write(np.float32(row[0]))
#                             writer.write(np.float32(row[1]))
#                             writer.write(np.float32(row[2]))
#                             if row[3] not in indices:
#                                 for index in range(3, 34):
#                                     writer.write(np.float32(row[index]))
#                             else:
#                                 for index in range(3, 34):
#                                     writer.write(np.float32(0))
#         except:
#             with open('error.txt', 'a') as ff:
#                 ff.write('{}\n'.format(number))
