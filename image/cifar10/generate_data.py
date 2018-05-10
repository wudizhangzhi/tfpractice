import numpy as np
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def handler_images_data(proto):
    # return np.vstack((proto[:, :1024], proto[:, 1024:1024*2], proto[:, 1024*2:1024*3])).T.reshape((-1, 32, 32, 3))
    return np.transpose(proto.reshape((-1, 3, 1024)), axes=[0, 2, 1]).reshape((-1, 32, 32, 3))

def produce_test_data(testfilename):
    # test data
    d_data = unpickle(testfilename)
    batch_labels = d_data[b'labels']
    batch_images = d_data[b'data']

    assert len(batch_labels) == len(batch_images)
    batch_labels = np.array(batch_labels)[:, np.newaxis]
    batch_images = handler_images_data(batch_images)
    dataset_test = tf.contrib.data.Dataset.from_tensor_slices((batch_images, batch_labels))
    return dataset_test

def produce_data(filelist, testfilename, label_name_file=None):
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
        batch_images = handler_images_data(batch_images)
        images_list.append(batch_images)

    images = np.vstack(images_list)
    labels = np.vstack(labels_list)
    # self.data_set = tf.contrib.data.Dataset.zip(dataset_list)
    data_set = tf.contrib.data.Dataset.from_tensor_slices((images, labels))

    # test data
    produce_test_data(testfilename)

    # 处理名称 label_names
    label_names = []
    if label_name_file:
        if not tf.gfile.Exists(label_name_file):
            print("can't find %s" % label_name_file)
        else:
            d_data = unpickle(label_name_file)
            label_names = d_data[b'label_names']

    return data_set, dataset_test, label_names



if __name__ == '__main__':
    batch = unpickle('data/data_batch_1')
    labels = unpickle('data/batches.meta')
    print(batch.keys())
    print(labels)
