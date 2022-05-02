# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

# data set preparationgit 
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

# data check
print("Shape of the sample data is",x.shape)
print("Shape of the target data is", y.shape)

# Regression with third feature
x = x[:, 2]
print(x.shape)


# +
# class for Neuron

class Neuron:
    
    def __init__(self):
        self.w = 1
        self.b = 1
        
    def forpass(self, x):
        y_hat = self.w*x + self.b
        return y_hat
    
    def backprop(self, x, error):
        w_grad = -x*error
        b_grad = -1*error
        return w_grad, b_grad
    
    def fit(self, x, y, epochs):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                error = y_i - y_hat
                self.w = self.w - self.backprop(x_i, error)[0]
                self.b = self.b - self.backprop(x_i, error)[1]


# -

neuron = Neuron()

neuron.fit(x, y, 100)

neuron.w, neuron.b

plt.scatter(x, y)
plt.plot([-0.1, 0.18], [-0.1*neuron.w+neuron.b, 0.18*neuron.w+neuron.b ])

plt.boxplot(diabetes.data)
plt.show()
plt.scatter(diabetes.data[:, 6], diabetes.target)
plt.show()

# +
neuron2 = Neuron()
neuron2.fit(diabetes.data[:, 6], y, 100)

plt.scatter(diabetes.data[:, 6], y)
plt.plot([-0.1, 0.18], [(-0.1)*neuron2.w+neuron2.b, (0.18)*neuron2.w+neuron2.b])
