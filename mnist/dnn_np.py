import numpy as np
import matplotlib.pyplot as plt

"""
use np build a simple model to compute y = 3.5x + 2
"""
LR = 0.01


class FCLayer(object):
    """
    全连接
    """

    def __init__(self, input_units, output_units):
        self._input_units = input_units
        self._output_units = output_units
        self.weight = np.random.standard_normal(size=(input_units, output_units))
        self.biase = np.zeros((input_units,))  # TODO use none-zero value

    def feed_forward(self, inputs, use_activation_fuc=True):
        self.input = inputs
        self.out = np.matmul(inputs, self.weight)
        if use_activation_fuc:
            self.out = self.activation_fuc(self.out)
        return self.out

    def activation_fuc(self, x):
        """
        use sigmod
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def d_out_wrt_net(self):
        # ∂oᵢ/∂netᵢ
        return self.out * (1 - self.out)


def generate_data(params=1, batch_size=36):
    x = np.random.uniform(-1, 1, (batch_size, params))
    if params == 1:
        y = np.where(x > 0.5, 1, 0)
    else:
        y = np.where(np.sum(x ** 2, 1) > 0.5, 1, 0).reshape((batch_size, 1))
        # TODO
        one_hot_label = np.zeros((batch_size, 2))
        for i in range(one_hot_label.shape[0]):
            one_hot_label[i][y[i]] = 1
        return x, one_hot_label
    return x, y


def compute_loss(predictions, labels):
    """
    使用l2 loss
    """
    # return np.sum(np.power(np.argmax(labels, 1) - np.argmax(predictions, 1), 2) / 2)
    return np.sum(np.power(labels - predictions, 2) / 2)


def compute_accuracy(preditions, labels):
    return np.abs(np.sum(np.equal(np.argmax(preditions, 1), np.argmax(labels, 1))) / len(preditions) * 100)


def train(params=1, maxstep=1000, batch_size=32):
    print('开始训练')
    loss_list = []
    accuracy_list = []
    fc2_weight_list = []
    fc1_weight_list = []

    # 建立模型
    fc1 = FCLayer(params, 5)
    fc2 = FCLayer(5, 2)

    def predict(input):
        return fc2.feed_forward(fc1.feed_forward(input))

    step = 0
    while step < maxstep:
        # 生成训练数据
        inputs, labels = generate_data(params, batch_size)
        # forward propagation
        fc1_output = fc1.feed_forward(inputs)
        fc2_output = fc2.feed_forward(fc1_output)
        # 误差
        loss = compute_loss(fc2_output, labels)

        # 最后一层的delta
        delta_out = (fc2_output - labels) * fc2_output * (1 - fc2_output)  # [batch_size, out]
        # 隐藏层的delta
        # [batch_size, out] * [unit_1, out].T
        delta_hidden = np.dot(delta_out, fc2.weight.T) * fc1_output * (1 - fc1_output)  # [batch_size, unit_1]

        # 更新参数，weight
        # [unit_1, out] = [batch_size, unit_1].T * [batch_size, out]
        fc2.weight -= LR * np.dot(fc1.out.T, delta_out)
        # [input, unit_1] = [batch_size, input].T * [batch_size, unit_1]
        fc1.weight -= LR * np.dot(inputs.T, delta_hidden)

        if step % 100 == 0:
            accuracy = compute_accuracy(fc2_output, labels)
            print('step: %d , loss: %0.4f, accuracy: %0.2f' % (step, loss, accuracy))
            # print(fc2.weight)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            fc2_weight_list.append(np.average(fc2.weight))
            fc1_weight_list.append(np.average(fc1.weight))
        step += 1

    # test
    test_data, test_label = generate_data(params, 100)
    results = predict(test_data)
    print('正确率: %0.2f' % compute_accuracy(results, test_label))

    # plot 散点图
    if params > 1:
        positive_indexes = np.where(np.argmax(results, 1) == 0)
        negative_indexes = np.where(np.argmax(results, 1) == 1)
        print(positive_indexes, negative_indexes)
        plt.plot(test_data[positive_indexes, 0], test_data[positive_indexes, 1], 'ro')
        plt.plot(test_data[negative_indexes, 0], test_data[negative_indexes, 1], 'bx')
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.show()
    else:
        # plot weight
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('percent')
        ax1.set_title('accuracy')
        ax1.plot(np.arange(len(accuracy_list)), accuracy_list, color='blue', lw=2)

        ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
        ax2.plot(np.arange(len(loss_list)), loss_list, color='yellow', lw=2)
        plt.show()


def main():
    train(params=2, maxstep=1000, batch_size=10)


if __name__ == '__main__':
    main()
    # inputs, labels = generate_data(2, batch_size=100)
    # print(np.argmax(labels, 1))
    # print(inputs)
    # print(labels)
    # print(compute_accuracy(inputs, labels))
