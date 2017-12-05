# encoding: utf-8
import sys
import tensorflow as tf
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        if sys.version_info.major == 3:
            d = pickle.load(fo, encoding='bytes')
        else:
            d = pickle.load(fo)
    return d


def _handler_images_data(proto):
    # return np.vstack((proto[:, :1024], proto[:, 1024:1024*2], proto[:, 1024*2:1024*3])).T.reshape((-1, 32, 32, 3))
    # Permute the dimensions of an array.
    return np.transpose(proto.reshape((-1, 3, 1024)), axes=[0, 2, 1]).reshape((-1, 32, 32, 3)).astype('float32')


def produce_test_data(testfilename):
    # test data
    d_data = unpickle(testfilename)
    batch_labels = d_data[b'labels']
    batch_images = d_data[b'data']

    assert len(batch_labels) == len(batch_images)
    batch_labels = np.array(batch_labels)[:, np.newaxis]
    batch_images = _handler_images_data(batch_images)
    # TODO
    # dataset_test = tf.data.Dataset.from_tensor_slices((batch_images, batch_labels))
    # return dataset_test
    return batch_images, batch_labels

def produce_data(filelist):
    assert isinstance(filelist, list)
    assert len(filelist) > 0
    labels_list = []
    images_list = []
    for filename in filelist:
        d_data = unpickle(filename)
        batch_labels = d_data[b'labels']
        batch_images = d_data[b'data']

        assert len(batch_labels) == len(batch_images)
        batch_labels = np.array(batch_labels)[:, np.newaxis]
        labels_list.append(batch_labels)
        batch_images = _handler_images_data(batch_images)
        images_list.append(batch_images)

    images = np.vstack(images_list)
    labels = np.vstack(labels_list)
    # self.data_set = tf.contrib.data.Dataset.zip(dataset_list)
    # data_set = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
    return images, labels


def read_label_name(label_name_file):
    if not tf.gfile.Exists(label_name_file):
        print("can't find %s" % label_name_file)
    else:
        d_data = unpickle(label_name_file)
        label_names = d_data[b'label_names']
    return label_names

if __name__ == '__main__':
    data = produce_test_data('cifar-10-batches-py/test_batch')
    print(data)