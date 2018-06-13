import numpy as np


class Neuron:
    """
    神经元
    """

    def __init__(self, bias):
        self.weight = []
        self.bias = bias if bias else np.random.random()

    def compute_total_net_input(self, inputs):
        total = 0
        for i in range(len(inputs)):
            total += self.weight[i] * inputs[i]
        return total + self.bias

    def compute_out(self, inputs):
        self.inputs = inputs
        self.out = self.activation_func(self.compute_total_net_input(inputs))
        return self.out

    def activation_func(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_error(self, target_output):
        return np.power(target_output - self.out, 2) / 2

    def d_error_wrt_out(self, target_output):
        # dE/dOⱼ
        return -(target_output - self.out)

    def d_out_wrt_net(self):
        # dOⱼ/dnetⱼ
        return self.out * (1 - self.out)

    def d_net_wrt_weight(self, index):
        # dnetⱼ/d wᵢⱼ
        return self.inputs[index]

    def compute_delta(self, target_output):
        return self.d_error_wrt_out(target_output) * self.d_out_wrt_net()


class NeuronLayer:
    def __init__(self, neuron_num, bias):
        self.bias = bias
        self.neurons = []
        for i in range(neuron_num):
            self.neurons.append(Neuron(bias=bias))

    def compute_sum_delta_matmul_weight(self, target_ouputs):
        total = 0
        for i in range(len(target_ouputs)):
            total += self.neurons[i].compute_delta(target_ouputs[i])
        return total
