import tensorflow as tf
import os

IMAGE_SIZE = 28
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
        )

    tf.summary.image('image', images)
    return images, tf.reshape(label_batch, [batch_size])


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_byte = 1
    result.width = 32
    result.height = 32
    result.depth = 3
    image_bytes = result.width * result.height * result.depth

    reader = tf.FixedLengthRecordReader(record_bytes=image_bytes + label_byte)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.int8)

    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_byte]), tf.int8)
    deph_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_byte], [label_byte + image_bytes]),
        [result.depth, result.height, result.width]
    )

    result.unit8image = tf.transpose(deph_major, [1, 2, 0])
    return result


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('找不到文件: ' + f)

        # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    with tf.name_scope('data_augmentation'):
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.unit8image, tf.float32)

        # 一些随机性的处理
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # crop
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        # flip
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # noise
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)


if __name__ == "__main__":
    distorted_inputs('cifar-10-batches-bin', 100)
