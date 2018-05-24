import numpy as np

import input_data

"""
use np to trans dense model
"""

w_dict = {}
b_dict = {}


def dense(inputs, units, name, activation_fun=False):
    input_shape = inputs.shape
    w_name = name + '_w'
    b_name = name + '_b'
    if w_name not in w_dict:
        w = np.random.standard_normal((input_shape[1], units))
        w_dict[w_name] = w
    else:
        w = w_dict[w_name]
    if b_name not in w_dict:
        b = np.random.standard_normal((input_shape[0]))
        b_name[b_name] = b
    else:
        b = b_dict[b_name]

    output = np.add(np.dot(inputs, w), b)
    # activation function
    # relu(rectified linear unit)  TODO solfmax
    if activation_fun:
        output = np.where(output < 0, 0, output)
    return output


def inference(images):
    d1 = dense(images, 196, name='layer_1')
    d2 = dense(d1, 49, name='layer_2')
    d3 = dense(d2, 10, name='layer_3')
    # solfmax
    exp_sum = np.sum(np.exp(d2))
    d3 = d3 / exp_sum
    return d3


def train():
    # load data
    dataset = input_data.read_data_sets('.', one_hot=True)
    dataset_train = dataset.train
    dataset_validation = dataset.validation
    dataset_test = dataset.test

    # train
    loss = 0
    while dataset_train.epochs_completed < 5:
        train_images, train_labels = dataset_train.next_batch(32)
        output = inference(train_images)
        # compute loss

        # cross entropy loss
        loss += - np.sum(np.dot(train_labels, np.log(output)))


if __name__ == '__main__':
    train()
