import math
import random
import numpy
import mnist


class Neuron:

    def __init__(self, layer_number, node_number, bias):
        self.layer_number = layer_number
        self.node_number = node_number
        self.bias = bias
        self.value = 0
        self.inputs = {}

    # adds a neuron to the dictionary of inputs
    def establish_input_neuron(self, neuron_id, weight):
        # defaults input value to 0
        self.inputs[neuron_id] = [0, weight]

    # output inputs dictionary keys so the outputs of these neurons can be obtained from outside this object
    def get_input_nodes(self):
        return list(self.inputs.keys())

    # with the neuron ids outputted from the function above, their output values are obtained and pushed back
    # into this object with this function.
    def set_input_value(self, neuron_id, value):
        self.inputs[neuron_id][0] = value

    # once all inputs are collected they can be processed
    def process_inputs(self):
        # sigmoid function
        summation = self.bias
        for input_node in self.inputs:
            summation += self.inputs[input_node][0] * self.inputs[input_node][1]
        # 1 / (1 + e^-z)
        self.value = 1 / (1 + math.exp(-summation))

    def get_value(self):
        return self.value


# adds set_value method specifically for neurons in the first layer
class InputNeuron(Neuron):

    def set_value(self, value):
        self.value = value


class NeuralNet:

    def __init__(self, data):
        # create network structure

        self.dataset_size = len(data)
        self.data = data
        # default values
        # first layer is an input layer so number of neurons equals dataset size
        self.layers = (self.dataset_size, 4, self.dataset_size)
        self.default_bias = 1
        self.default_weighting = 1

        # express neural_net as 2D list
        self.network = []
        for layer in self.layers:
            self.network.append([])

        self.output_vector = []

        # iterate through each layer
        for layer_id in range(0, len(self.layers)):
            # iterate through each node in layer
            for node_id in range(0, self.layers[layer_id]):
                # if input layer, create InputNeurons that can have their value manually set
                if layer_id == 0:
                    node = InputNeuron(layer_id, node_id, numpy.random.rand())
                # otherwise set number of inputs equal to number of nodes in previous layer
                else:
                    node = Neuron(layer_id, node_id, numpy.random.rand())

                    # connect each node to every node in the previous layer
                    for next_layer_node_id in range(0, self.layers[layer_id-1]):
                        # default weight to 1
                        node.establish_input_neuron(next_layer_node_id, numpy.random.rand())

                # add node to neural net list
                self.network[layer_id].append(node)

    def pass_data(self):
        # pass data through network

        # feed data to first layer
        for i in range(0, len(self.data)):
            # set values for input layer
            self.network[0][i].set_value(self.data[i])

        # iterate through each layer after input layer
        for layer_id in range(1, len(self.network)):
            # iterate through each neuron in layer
            for neuron in self.network[layer_id]:
                # get data from previous layer
                input_nodes = neuron.get_input_nodes()
                for input_neuron in input_nodes:
                    # get the outputted value from node in previous layer
                    input_neuron_value = self.network[layer_id-1][input_neuron].get_value()
                    # set the input value for this node
                    neuron.set_input_value(input_neuron, input_neuron_value)

                # apply sigmoid to inputted data
                neuron.process_inputs()

                # if in output layer, add value to output_vector
                if layer_id == len(self.network)-1:
                    self.output_vector.append(neuron.get_value())


        def gradient_descent(self):
            pass
            # choose a small number, m, of randomly chosen training inputs - a mini batch
            # ideally the average gradient vector from this sample is roughly equal to the gradient vector from the
            #       overall training dataset
            # this gives rough pseudocode:
            # until all training inputs used:
            # choose small number m of randomly chosen training inputs:
            #   weight_sum = 0
            #   bias_sum = 0
            #   for j in range(0, m):
            #       weight_sum += gradient of cost against weight for sample x_j
            #       bias_sum += gradient of cost against bias for sample x_j
            #   avg_weight_grad = weight_sum * learning_rate / m
            #   avg_bias_grad = bias_sum * learning_rate / m
            #   updated_weights = weights - avg_weight_grad
            #   updated_biases = biases - avg_bias_grad



    def output_result(self):
        for x in self.output_vector:
            print(x)


number_samples = 1000
number_input_neurons = 8

# generate a list of samples and outputted data
samples = []
for i in range(0, number_samples):
    sample = []
    for j in range(0, number_input_neurons):
        sample.append(random.uniform(0, 1))
    outputs = []
    for y in sample:
        # for now, the expected outputs are half the inputs
        outputs.append(y/2)
    samples.append([sample, outputs])


# pass each dataset through the neural net and store the outputs
for sample in samples:
    test_net = NeuralNet(sample[0])
    test_net.pass_data()
    output = test_net.output_vector
    sample.append(output)


# use the cost function to evaluate outputs

# cost function
summation = 0
for sample in samples:
    # find vector difference between expected and real outcome
    vector_difference = []
    # for each output value
    for i in range(0, len(sample[1])):
        # find difference between expected and actual output value, put this in a difference vector
        vector_difference.append(sample[1][i] - sample[2][i])

    # find square magnitude by doing pythagoras and square rooting, then square again
    # last two steps cancel out so just find sum of squares
    square_sum = 0

    for x in vector_difference:
        square_sum += x**2

    summation += square_sum

cost = summation / (2 * number_samples)

print(cost)
