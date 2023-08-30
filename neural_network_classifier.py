import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  #maps data to be between 0 and 1 (basically yes or no)
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  #y true = actual value (M or F aka 0 or 1), y pred = predicted value
  #shows how accurate algorithm is
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
        #aka a layer between the input and output layer
    - an output layer with 1 neuron (o1)
  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  #neural network is formed by matrix multiplication, basically
  #kind of similar to my image opacity layering format.
  #result = weight1 * x1 + weight2 * x2 + constant
  #tweaking weights and constant (the w's and b's in this case)
  #to get accurate result
  def __init__(self):
    #initializing weights and biases to random
    # Weights
    self.w1 = 0.5
    self.w2 = 0.5
    self.w3 = 0.5
    self.w4 = 0.5
    self.w5 = 0.5
    self.w6 = 0.5

    # Biases
    self.b1 = 0
    self.b2 = 0
    self.b3 = 0

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    #heres the multiplication i was talkign about earlier
    mult1 = np.array([[self.w1, self.w3],[self.w2, self.w4]])
    mult2 = np.array([[self.w5], [self.w6]])
    hmatrix = x.dot(mult1)
    hmatrix[0] += self.b1
    hmatrix[1] += self.b2
    h = sigmoid(hmatrix)
    o1 = sigmoid(h.dot(mult2) + self.b3)
    return o1[0] 
  '''
  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    #heres the multiplication i was talkign about earlier
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1
  '''
  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        '''
        mult1 = np.array([[self.w1, self.w2],[self.w3, self.w4]])
        mult2 = np.array([[self.w5], [self.w6]])

        hsum = x.dot(mult1)
        hsum[0] += self.b1
        hsum[1] += self.b2
        h = sigmoid(hsum)

        osum = h.dot(mult2) + self.b3
        o1 = sigmoid(osum)
        '''
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)
        
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        #calculate partial of L to w1 by using chain rule
        #this partial tells us how to tweak the data (make w1 smaller or bigger)
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 100 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        '''
        print(y_preds)
        new = np.empty(shape = (4,))
        new[0] = y_preds[0]
        new[1] = y_preds[1]
        new[2] = y_preds[2]
        new[3] = y_preds[3]
        '''
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))
        print("Coefficients:")
        print("w1: {}".format(self.w1))
        print("w2: {}".format(self.w2))
        print("w3: {}".format(self.w3))
        print("w4: {}".format(self.w4))
        print("w5: {}".format(self.w5))
        print("w6: {}".format(self.w6))
        print("b1: {}".format(self.b1))
        print("b2: {}".format(self.b2))
        print("b3: {}".format(self.b3))
        print()

def matrix_multiply(a, b):
    numrows = a.shape[0]
    numcols = b.shape[1]
    result = np.empty(shape = [numrows, numcols])
    for row in range(numrows):
      for col in range(numcols):
        result[row, col] = a[row].dot(b[:, col])
    return result        

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
tracy = np.array([-25, -1]) #110 pounds, 65 inches
print("Tracy: %.3f" % network.feedforward(tracy)) #0.964 - F (which is correct)
alice = np.array([-2, -1])
print("Alice: %.3f" % network.feedforward(alice))
bob = np.array([25, 6])
print("Bob: %.3f" % network.feedforward(bob))
charlie = np.array([17, 4])
print("Charlie: %.3f" % network.feedforward(charlie))
diana = np.array([-15, -6])
print("Diana: %.3f" % network.feedforward(diana))

'''
a = np.array([[1, 2]])
b = np.array([[2], [1]])
c = np.array([[1,2,3],[4,5,6]])
d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(matrix_multiply(c, d))
print(matrix_multiply(a, b))

c = np.array([[1,2,3],[4,5,6]])
d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(c.dot(d))
print(matrix_multiply(c, d))
'''
