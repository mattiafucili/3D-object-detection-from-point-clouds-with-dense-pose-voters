import numpy as np
import random
from source_bb.utilities import parse_file
import os
import sys
from sklearn.cluster import KMeans
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
LABEL_DATA = BASE_DIR + '/labels/'


def load_pointcloud(filename):
    """

    :param filename: filename
    :return: point cloud without reflectance (nx3)
    """
    return np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]


def load_pointcloud_with_bboxes_info(filename):
    """

    :param filename: filename
    :return: point cloud (nx34)
    """
    return np.fromfile(filename, dtype=np.float32).reshape((-1, 34))


def draw_box(pyplot_axis, vertices, color='black'):
    face_idx = np.array([[0, 1, 5, 4],  # front face
                         [1, 2, 6, 5],  # left face
                         [2, 3, 7, 6],  # back face
                         [3, 0, 4, 7]])  # right

    for i in range(4):
        pyplot_axis.plot([vertices[0][face_idx[0, i]], vertices[0][face_idx[0, (i + 1) % 4]]], [vertices[1][face_idx[0, i]], vertices[1][face_idx[0, (i + 1) % 4]]], [vertices[2][face_idx[0, i]], vertices[2][face_idx[0, (i + 1) % 4]]], color=color)
        pyplot_axis.plot([vertices[0][face_idx[1, i]], vertices[0][face_idx[1, (i + 1) % 4]]], [vertices[1][face_idx[1, i]], vertices[1][face_idx[1, (i + 1) % 4]]], [vertices[2][face_idx[1, i]], vertices[2][face_idx[1, (i + 1) % 4]]], color=color)
        pyplot_axis.plot([vertices[0][face_idx[3, i]], vertices[0][face_idx[3, (i + 1) % 4]]], [vertices[1][face_idx[3, i]], vertices[1][face_idx[3, (i + 1) % 4]]], [vertices[2][face_idx[3, i]], vertices[2][face_idx[3, (i + 1) % 4]]], color=color)
        pyplot_axis.plot([vertices[0][face_idx[2, i]], vertices[0][face_idx[2, (i + 1) % 4]]], [vertices[1][face_idx[2, i]], vertices[1][face_idx[2, (i + 1) % 4]]], [vertices[2][face_idx[2, i]], vertices[2][face_idx[2, (i + 1) % 4]]], color=color)


def draw_vectors(points, vertices, ax):
    for index in range(len(points)):
        point = points[index]
        vertex = vertices[index]
        ax.plot([point[0], vertex[0]], [point[1], vertex[1]], [point[2], vertex[2]], 'red')
        # ax.plot([points[index, 0]], [points[index, 1]], [points[index, 2]], 'bo')


def safe_subsample(pointcloud, frame, num_point):
    dataset = np.array(random.sample(list(pointcloud), num_point))
    labels = parse_file(LABEL_DATA + '%06d.txt' % frame, ' ')
    centroid_points = np.unique(dataset[:, 4])
    labels = list(x for x in labels if 'DontCare' not in x)
    a = np.count_nonzero(centroid_points)
    while a != len(labels):
        a = np.count_nonzero(centroid_points)
        dataset = np.array(random.sample(list(pointcloud), 13545))
        centroid_points = np.unique(dataset[:, 4])

    return dataset


def compute_bboxes(translations, yaws, dimensions):
    """

    :param translations: x, y, z
    :param yaws: rotation around z
    :param dimensions: h, w, l
    :return: 3D vertices coordinates
    """
    bbox = []
    for i in range(translations.shape[0]):

        h = dimensions[i][0]
        w = dimensions[i][1]
        l = dimensions[i][2]

        if h == -1:
            continue

        rot_mat = np.array(
            [[np.cos(yaws[i]), 0, np.sin(yaws[i])],
             [0, 1, 0],
             [-np.sin(yaws[i]), 0, np.cos(yaws[i])]]
        )

        x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

        vertices = np.matmul(rot_mat, np.array([x_corners, y_corners, z_corners]))
        vertices[0, :] = vertices[0, :] + translations[i][0]
        vertices[1, :] = vertices[1, :] + translations[i][1]
        vertices[2, :] = vertices[2, :] + translations[i][2]

        bbox.append(vertices)

    bbox = np.stack(bbox)

    return bbox


def calib_bboxes(bboxes, calib):
    """

    :param bboxes: bounding boxes
    :param calib: calibration to velodyne coordinate system (VDS)
    :return: bounding boxes in VDS
    """
    r0, _, c2v, _ = calib_matrices(calib)

    for i in range(bboxes.shape[0]):
        # 3x3 * 3x8 = 3x8
        bboxes[i] = np.dot(np.linalg.inv(r0), bboxes[i])
        hom = cart2hom(np.transpose(bboxes[i]))
        bboxes[i] = np.transpose(np.dot(hom, np.transpose(c2v)))

    return bboxes


def calib_matrices(calib):
    """

    :param calib: calibration file
    :return: useful parameters
    """
    p = calib['P2']
    p = np.reshape(p, [3, 4])
    cam, R, t, _, _, _, _ = cv.decomposeProjectionMatrix(p)
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = -t[:3, 0]
    v2c = np.eye(4)
    v2c[:3, :4] = np.reshape(calib['Tr_velo_to_cam'], [3, 4])

    c2v = inverse_rigid_trans(v2c)
    r0 = calib['R0_rect']
    r0 = np.reshape(r0, [3, 3])

    return r0, v2c, c2v, cam,  T_w2c


def inverse_rigid_trans(t):
    # 3x4
    inv_t = np.zeros_like(t)
    inv_t[0:3, 0:3] = np.transpose(t[0:3, 0:3])
    inv_t[0:3, 3] = np.dot(-np.transpose(t[0:3, 0:3]), t[0:3, 3])
    return inv_t


def cart2hom(cart):
    n = cart.shape[0]
    hom = np.hstack((cart, np.ones((n, 1))))
    return hom


def ransac(points, max_iteration, threshold):
    tot_points = len(points)
    res_points = []
    possible_res = []

    for _ in range(max_iteration):
        possible_res = random.sample(list(points), min(tot_points, 5))
        # print('{} {}\n'.format(points.shape, np.array(possible_res).shape))
        # centroid = KMeans(n_clusters=1).fit(np.array(possible_res)).cluster_centers_[0]
        centroid = np.mean(possible_res, axis=0)
        starter_points = [tuple(el) for el in possible_res]
        for point in points:
            if tuple(np.expand_dims(point, -1)) not in starter_points:
                distance = np.sqrt(np.square(point[0] - centroid[0]) + np.square(point[1] - centroid[1]) + np.square(point[2] - centroid[2]))
                if distance <= threshold:
                    possible_res.append(point)
        if len(res_points) < len(possible_res):
            res_points = possible_res

    return res_points


def get_view_point_cloud(pointcloud, img_width, img_height, calib):
    """

    :param pointcloud: initial point cloud
    :param img_width: width
    :param img_height: height
    :param calib: calibration file
    :return: point cloud of the projection of the image in the initial point cloud
    """
    r0, v2c, _, cam, T_w2c = calib_matrices(calib)

    # points = cart2hom(pointcloud)  # nx3 -> nx4
    points = np.matmul(v2c[:3,:3], pointcloud.T).T + v2c[:3, 3]
    points = np.matmul(T_w2c[:3,:3], points.T).T + T_w2c[:3, 3]

    # points = np.transpose(np.dot(r0, np.transpose(points)))  # (3x3 * 3xn = 3xn) -> nx3
    # points = cart2hom(points)  # nx3 -> nx4

    points_2d = np.matmul(cam, points.T).T # nx4 * 4x3 = nx3
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    points_2d = points_2d[:, :2]

    view_point = (points_2d[:, 0] < img_width) & (points_2d[:, 0] >= 0) & (points_2d[:, 1] < img_height) & (points_2d[:, 1] >= 0)
    view_point = view_point & (points[:, 2] > 2.0)

    return pointcloud[view_point, :]


def draw_box_on_image(image, vertices):
    # lower square
    cv.line(image, (vertices[0, 0], vertices[1, 0]), (vertices[0, 1], vertices[1, 1]), color=(0, 0, 255), thickness=2)
    cv.line(image, (vertices[0, 1], vertices[1, 1]), (vertices[0, 2], vertices[1, 2]), color=(0, 255, 0), thickness=2)
    cv.line(image, (vertices[0, 2], vertices[1, 2]), (vertices[0, 3], vertices[1, 3]), color=(0, 255, 0), thickness=2)
    cv.line(image, (vertices[0, 3], vertices[1, 3]), (vertices[0, 0], vertices[1, 0]), color=(0, 255, 0), thickness=2)

    # upper square
    cv.line(image, (vertices[0, 4], vertices[1, 4]), (vertices[0, 5], vertices[1, 5]), color=(0, 0, 255), thickness=2)
    cv.line(image, (vertices[0, 5], vertices[1, 5]), (vertices[0, 6], vertices[1, 6]), color=(0, 255, 0), thickness=2)
    cv.line(image, (vertices[0, 6], vertices[1, 6]), (vertices[0, 7], vertices[1, 7]), color=(0, 255, 0), thickness=2)
    cv.line(image, (vertices[0, 7], vertices[1, 7]), (vertices[0, 4], vertices[1, 4]), color=(0, 255, 0), thickness=2)

    # links
    cv.line(image, (vertices[0, 0], vertices[1, 0]), (vertices[0, 4], vertices[1, 4]), color=(0, 0, 255), thickness=2)
    cv.line(image, (vertices[0, 1], vertices[1, 1]), (vertices[0, 5], vertices[1, 5]), color=(0, 0, 255), thickness=2)
    cv.line(image, (vertices[0, 2], vertices[1, 2]), (vertices[0, 6], vertices[1, 6]), color=(0, 255, 0), thickness=2)
    cv.line(image, (vertices[0, 3], vertices[1, 3]), (vertices[0, 7], vertices[1, 7]), color=(0, 255, 0), thickness=2)


def inside_the_box(point, o, a, b, c):
    """

    :param point: one point of the point cloud frame
    :param o: one vertex of the 3D bounding box '[x, y, z]'
    :param a: one of the 3 vertices adjacent to o '[x, y, z]'
    :param b: one of the 3 vertices adjacent to o '[x, y, z]'
    :param c: one of the 3 vertices adjacent to o '[x, y, z]'

    a != b != c

    :return: True if the point is inside the box, False otherwise
    """
    op = [point[0] - o[0], point[1] - o[1], point[2] - o[2]]
    oa = [a[0] - o[0], a[1] - o[1], a[2] - o[2]]
    ob = [b[0] - o[0], b[1] - o[1], b[2] - o[2]]
    oc = [c[0] - o[0], c[1] - o[1], c[2] - o[2]]
    ap = [point[0] - a[0], point[1] - a[1], point[2] - a[2]]
    bp = [point[0] - b[0], point[1] - b[1], point[2] - b[2]]
    cp = [point[0] - c[0], point[1] - c[1], point[2] - c[2]]
    ao = [o[0] - a[0], o[1] - a[1], o[2] - a[2]]
    bo = [o[0] - b[0], o[1] - b[1], o[2] - b[2]]
    co = [o[0] - c[0], o[1] - c[1], o[2] - c[2]]

    return (np.dot(op, oa) >= 0) and (np.dot(op, ob) >= 0) and (np.dot(op, oc) >= 0) and (np.dot(ap, ao) >= 0) and (np.dot(bp, bo) >= 0) and (np.dot(cp, co) >= 0)


def expand_matrix(matrix):
    temp = np.zeros([4, 4], dtype=np.float32)
    temp[0:3, 0:3] = matrix
    temp[3, 3] = 1

    return temp


def expand_matrix2(matrix):
    temp = np.zeros([4, 4], dtype=np.float32)
    temp[0:3, 0:4] = matrix
    temp[3, 3] = 1

    return temp


def expand_box(box):
    temp = np.ones([4, 8])
    temp[0:3] = box

    return temp
