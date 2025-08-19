import numpy as np
from sklearn.metrics import accuracy_score
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def forward_propagation(X, weights1, bias1, weights2, bias2, weights3, bias3):
    z1 = np.dot(weights1, X) + bias1
    a1 = relu(z1)
    z2 = np.dot(weights2, a1) + bias2
    a2 = relu(z2)
    z3 = np.dot(weights3, a2) + bias3
    output = sigmoid(z3)
    return output

np.random.seed(23)
X = np.random.rand(3, 1)
weights1 = np.random.rand(3, 3)
bias1 = np.random.rand(3, 1)
weights2 = np.random.rand(3, 3) 
bias2 = np.random.rand(3, 1)  
weights3 = np.random.rand(1, 3) 
bias3 = np.random.rand(1, 1) 
output = forward_propagation(X, weights1, bias1, weights2, bias2, weights3, bias3)
print("Output (probability):", output)
y_true = np.array([1])  
y_pred = (output > 0.5).astype(int) 

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)