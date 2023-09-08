import numpy as np
import functions 



class Neuron:
  def __init__(self, weights, bias):
    self.weights = np.random.rand(2)
    self.bias = np.random.rand(2)
    #print("weights", self.weights)
  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    #print("total", total)
    return functions.sigmoid(total)

class NeuralNetwork:

  def __init__(self):
    # Weights
    self.weight = np.random.rand(3,2)
    # Biases
    self.bias = np.random.rand(3,2)
    print("NeuralNetwork weight", self.weight)
    self.hidden1 = Neuron(self.weight, self.bias)
    self.hidden2 = Neuron(self.weight, self.bias)
    self.output1 = Neuron(self.weight, self.bias)
    
  def feedforward(self, inputs):
    print("hidden1 weight", self.hidden1)
    out_h1 = self.hidden1.feedforward(inputs)
    print("out_h1 weight", out_h1)
    out_h2 = self.hidden2.feedforward(inputs)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.output1.feedforward(np.array([out_h1, out_h2]))
    print("out_h1 weight 2", out_o1)
    return out_o1

  def train(self, inputs, all_y_trues):
    pass

  def predict(inputs, all_y_trues):
    pass

network = NeuralNetwork()
x = np.array([3, 2])
print(network.feedforward(x)) 