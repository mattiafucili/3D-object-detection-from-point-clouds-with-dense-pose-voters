import sys
import numpy as np
import math


def print_progress(iteration, total):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------

    iteration :
                Current iteration (Int)
    total     :
                Total iterations (Int)
    """
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '[]' * filled_length + '-' * (100 - filled_length)

    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def parse_file(filename, delim):
    """

    :param filename: name of the file to read
    :param delim: character between elements
    :return: list of elements
    """
    info = []
    with open(filename) as f:
        for line in f:
            info.append(line.split(delim))

    return info


def read_calib_file(filename):
    """

    :param filename: filename
    :return:
    """
    data = {}
    with open(filename, 'r') as f:
        for line in f.readlines():

            line = line.rstrip()
            if len(line) == 0:
                continue

            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def extract_types(labels):
    types = []
    for label in labels:
        obj_type = label[0]
        types.append(obj_type)

    return types


def extract_translations(labels):
    translations = []
    for label in labels:
        translation = [np.float32(label[11]), np.float32(label[12]), np.float32(label[13])]
        translations.append(translation)

    return np.stack(translations)


def extract_dimensions(labels):
    dimensions = []
    for label in labels:
        dimension = [np.float32(label[8]), np.float32(label[9]), np.float32(label[10])]
        dimensions.append(dimension)

    return np.stack(dimensions)


def extract_rotations(labels):
    rotations = []
    for label in labels:
        rotation = np.float32(label[14])
        rotations.append(rotation)

    return rotations
