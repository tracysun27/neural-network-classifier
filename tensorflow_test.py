import sys
sys.path.insert(0, "/Users/trac.k.y/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages")
import numpy as np
import tensorflow as tf

'''
ORIGINAL VALUES AT END OF TRAINING
layer1bias = tf.constant_initializer(np.array([[-0.3378255404279579],
                                               [-0.3378255404279579]]))
layer2bias = tf.constant_initializer(2.9134704326136895)
init1 = tf.constant_initializer(np.array([[1.1757534896912027, 0.83786076484401],
                  [1.1757534896912027, 0.83786076484401]]))
init2 = tf.constant_initializer(np.array([[-3.049036127174657],
                                         [-3.049036127174657]]))
'''
#initialize all weights to 0.5 and all biases to 0
init1 = tf.constant_initializer(np.array([[0.5, 0.5],
                                          [0.5, 0.5]]))
init2 = tf.constant_initializer(np.array([[0.5],
                                         [0.5]]))
layer1bias = tf.constant_initializer(np.array([[0],
                                               [0]]))
layer2bias = tf.constant_initializer(0)


inputlayer = tf.keras.Input(shape = 2)
layer1 = tf.keras.layers.Dense(units = 2,
                               activation = 'sigmoid',
                               kernel_initializer=init1,
                               bias_initializer=layer1bias)
layer2 = tf.keras.layers.Dense(1, activation = 'sigmoid',
                               kernel_initializer=init2,
                               bias_initializer=layer2bias)                               
model = tf.keras.models.Sequential()
model.add(inputlayer)
model.add(layer1)
model.add(layer2)

#model.summary()
data = np.array([[-2, -1],  # Alice
                  [25, 6],   # Bob
                  [17, 4],   # Charlie
                  [-15, -6], # Diana
                ])
all_y_trues = np.array([[1], # Alice
                          [0], # Bob
                          [0], # Charlie
                          [1], # Diana
                        ])
class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 4000
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.counter == self.SHOW_NUMBER or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
            if self.epoch > 1:
                self.counter = 0
        self.counter += 1

model.compile(optimizer = 'sgd',
  loss = 'mean_squared_error',
  metrics = ['accuracy'])

#model.fit(x=data, y=all_y_trues, epochs=10000, batch_size=1, verbose=0, callbacks=[Callback()])
model.fit(x=data, y=all_y_trues, epochs=10000, batch_size=1, verbose=0)
print(model.weights)

model.save('/Users/trac.k.y/Documents/python coding fun/neural networks')
loadmodel = tf.keras.models.load_model('/Users/trac.k.y/Documents/python coding fun/neural networks')
print(loadmodel.weights)
print(loadmodel.predict([[-25,-1]]))
