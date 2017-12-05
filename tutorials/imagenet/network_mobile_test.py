# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_base import mobilenet_v1, mobilenet_v1_base


class MobileNetTest(tf.test.TestCase):
    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = mobilenet_v1(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('MobilenetV1/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                             [batch_size, num_classes])

    def testEvalution(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))

        logits, end_points = mobilenet_v1(inputs, num_classes)

        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEqual(output.shape, (batch_size,))

    def testEvalWithReuse(self):
        batch_size = 5
        eval_batch_size = 2
        height, width = 224, 224
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        mobilenet_v1(inputs, num_classes)

        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = mobilenet_v1(eval_inputs, num_classes, reuse=True)
        predition = tf.argmax(logits, axis=1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predition)
            self.assertEqual(output.shape, (eval_batch_size,))

    def testModelHasExpectedNumberParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm):
            net, endpoints = mobilenet_v1_base(inputs)
            total_params, _ = slim.model_analyzer.analyze_vars(slim.get_model_variables())
            # TODO 怎么算的?
            self.assertEqual(3217920, total_params)

    def testBuildAndCheckAllEndPointsApproximateFaceNet(self):
        batch_size = 5
        height, width = 128, 128
        inputs = tf.random_uniform((batch_size, height, width, 3))

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm):
            net, end_points = mobilenet_v1_base(inputs, final_endpoint='Conv2d_13_pointwise', depth_multiplier=0.75)
            endpoints_shapes = {
                'Conv2d_0': [batch_size, 64, 64, 24],
                'Conv2d_1_depthwise': [batch_size, 64, 64, 24],
                'Conv2d_1_pointwise': [batch_size, 64, 64, 48],

                'Conv2d_2_depthwise': [batch_size, 32, 32, 48],
                'Conv2d_2_pointwise': [batch_size, 32, 32, 96],

                'Conv2d_3_depthwise': [batch_size, 32, 32, 96],
                'Conv2d_3_pointwise': [batch_size, 32, 32, 96],

                'Conv2d_4_depthwise': [batch_size, 16, 16, 96],
                'Conv2d_4_pointwise': [batch_size, 16, 16, 192],

                'Conv2d_5_depthwise': [batch_size, 16, 16, 192],
                'Conv2d_5_pointwise': [batch_size, 16, 16, 192],

                'Conv2d_6_depthwise': [batch_size, 8, 8, 192],
                'Conv2d_6_pointwise': [batch_size, 8, 8, 384],
                'Conv2d_7_depthwise': [batch_size, 8, 8, 384],
                'Conv2d_7_pointwise': [batch_size, 8, 8, 384],
                'Conv2d_8_depthwise': [batch_size, 8, 8, 384],
                'Conv2d_8_pointwise': [batch_size, 8, 8, 384],
                'Conv2d_9_depthwise': [batch_size, 8, 8, 384],
                'Conv2d_9_pointwise': [batch_size, 8, 8, 384],
                'Conv2d_10_depthwise': [batch_size, 8, 8, 384],
                'Conv2d_10_pointwise': [batch_size, 8, 8, 384],
                'Conv2d_11_depthwise': [batch_size, 8, 8, 384],
                'Conv2d_11_pointwise': [batch_size, 8, 8, 384],

                'Conv2d_12_depthwise': [batch_size, 4, 4, 384],
                'Conv2d_12_pointwise': [batch_size, 4, 4, 768],

                'Conv2d_13_depthwise': [batch_size, 4, 4, 768],
                'Conv2d_13_pointwise': [batch_size, 4, 4, 768],
            }
            for key, value in endpoints_shapes.items():
                self.assertIn(key, end_points)
                self.assertEqual(value, end_points[key].get_shape().as_list())

    def testOutputStride16BuildAndCheckAllEndPointsUptoConv2d_13(self):
        batch_size = 5
        height, width = 224, 224
        output_stride = 16

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm):
            net, endpoints = mobilenet_v1_base(inputs, output_stride=output_stride)

            endpoints_shapes = {
                'Conv2d_0': [batch_size, 112, 112, 32],
                'Conv2d_1_depthwise': [batch_size, 112, 112, 32],
                'Conv2d_1_pointwise': [batch_size, 112, 112, 64],

                'Conv2d_2_depthwise': [batch_size, 56, 56, 64],
                'Conv2d_2_pointwise': [batch_size, 56, 56, 128],

                'Conv2d_3_depthwise': [batch_size, 56, 56, 128],
                'Conv2d_3_pointwise': [batch_size, 56, 56, 128],

                'Conv2d_4_depthwise': [batch_size, 28, 28, 128],
                'Conv2d_4_pointwise': [batch_size, 28, 28, 256],

                'Conv2d_5_depthwise': [batch_size, 28, 28, 256],
                'Conv2d_5_pointwise': [batch_size, 28, 28, 256],

                'Conv2d_6_depthwise': [batch_size, 14, 14, 256],
                'Conv2d_6_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_7_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_7_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_8_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_8_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_9_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_9_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_10_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_10_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_11_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_11_pointwise': [batch_size, 14, 14, 512],

                'Conv2d_12_depthwise': [batch_size, 14, 14, 512],
                'Conv2d_12_pointwise': [batch_size, 14, 14, 1024],

                'Conv2d_13_depthwise': [batch_size, 14, 14, 1024],
                'Conv2d_13_pointwise': [batch_size, 14, 14, 1024],
            }

            self.assertEqual(endpoints.keys(), endpoints_shapes.keys())
            for key, value in endpoints_shapes.items():
                self.assertTrue(key in endpoints)
                self.assertEqual(endpoints[key].get_shape().as_list(), value)


if __name__ == '__main__':
    tf.test.main()
    pass
