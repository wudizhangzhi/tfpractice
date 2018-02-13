#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/13 下午7:58
# @Author  : wudizhangzhi

import tensorflow as tf
import iris_data

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('train_steps', 1000, 'training steps')

FLAGS = flags.FLAGS


def main(_):
    # build dataset
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # feature_column(describe how to use input)
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # estimator
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=my_feature_columns,
        n_classes=3,
    )

    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y, FLAGS.batch_size),
        steps=FLAGS.train_steps
    )

    evaluate_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, FLAGS.batch_size),
        steps=FLAGS.train_steps
    )
    # {'accuracy': 0.96666664, 'average_loss': 0.058699723, 'loss': 1.7609917, 'global_step': 1000}
    print('-' * 40)
    print('accuracy :{accuracy:0.3f}'.format(**evaluate_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    print('-' * 40)
    predict_result = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x, label=None, batch_size=FLAGS.batch_size),
    )
    template = '预测结果: {} 概率:{:0.2f}%  。期望: {}'
    for predict_dict, expec in zip(predict_result, expected):
        # predict_dict : {
        #                 'logits': array([ 13.478667 ,   7.7477455, -21.122244 ], dtype=float32),
        #                 'probabilities': array([9.9676645e-01, 3.2335958e-03, 9.3671849e-16], dtype=float32),
        #                 'class_ids': array([0]), 'classes': array([b'0'], dtype=object)}
        class_ids = predict_dict['class_ids'][0]
        _predict = iris_data.SPECIES[class_ids]
        probability = predict_dict['probabilities'][class_ids]
        print(template.format(_predict, probability * 100, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
