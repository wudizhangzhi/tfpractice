import numpy as np
import matplotlib.pyplot as plt

"""
use np build a simple model to compute y = 3.5x + 2
"""
LR = 0.1


class Neuron:
    def __init__(self):
        pass


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

    def d_out_wrt_net(self):
        # ∂oᵢ/∂netᵢ
        return self.out * (1 - self.out)


def generate_data(batch_size=36):
    x = np.random.random_sample((batch_size, 1))
    # y = 3.5 * x + 2
    y = np.where(x > 0.5, 1, 0)
    return x, y


def compute_loss(predictions, labels):
    """
    使用l2 loss
    """
    return np.sum(np.power(labels - predictions, 2) / 2)


def compute_accuracy(preditions, labels):
    return np.abs(np.sum(np.equal(np.where(preditions > 0.5, 1, 0), labels)) / len(preditions) * 100)


def train(params=1, maxstep=1000, batch_size=32):
    print('开始训练')
    loss_list = []
    accuracy_list = []
    fc2_weight_list = []
    fc1_weight_list = []

    fc1 = FCLayer(params, 5)
    fc2 = FCLayer(5, 1)

    def predict(input):
        return fc2.feed_forward(fc1.feed_forward(input))

    step = 0
    while step < maxstep:
        inputs, labels = generate_data(batch_size)
        fc1_output = fc1.feed_forward(inputs)
        fc2_output = fc2.feed_forward(fc1_output)

        loss = compute_loss(fc2_output, labels)

        # 最后一层的delta
        delta_out = (fc2_output - labels) * fc2_output * (1 - fc2_output)  # [batch_size, out]
        # 隐藏层的delta
        # [batch_size, out] * [unit_1, out].T
        delta_hidden = np.dot(delta_out, fc2.weight.T) * fc2_output * (1 - fc2_output)  # [batch_size, unit_1]

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

    result = predict(np.array([[0.2], [0.8]]))
    print(result)

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
    train(maxstep=10000, batch_size=10)


if __name__ == '__main__':
    main()
    # inputs, labels = generate_data(2)
    # print(inputs)
    # print(labels)
    # print(compute_accuracy(inputs, labels))
