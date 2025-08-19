# Objective: 

Create a simple neural network in TensorFlow to classify breast cancer data from the Breast Cancer dataset. You’ll define the model, train it on labeled data, and evaluate its performance during the training.

Instructions:

Import tensorflow and load the Breast Cancer dataset from sklearn.datasets.load_breast_cancer. (Including two classes: malignant (0) and benign (1).)
Stratified 5-Fold Cross-Validation:
Use StratifiedKFold from sklearn.model_selection to perform 5-fold cross-validation.
Define the model architecture:
Input layer: flatten the input data to a 1D vector (since the dataset has 30 features).
Hidden layers: The number of layers and neurons in each layer is up to you. Consider the appropriate activation function. 
Output layer: The output layer should consist of a single neuron (Because it's a binary classification). Consider an appropriate activation function for binary classification. 
Compile the model:
Use Adam as the optimizer.
Use accuracy as the metric.
Train the model:
Train the model on the training data in each fold, using a validation set.
Model Evaluation:
Evaluate the model on the validation set for each fold.
Plot the training and validation accuracy over epochs for each fold.


<img width="2318" height="1596" alt="75b3a8db-ba60-48ae-973b-b8b4355ae50c" src="https://github.com/user-attachments/assets/5ff61af3-3180-41c7-b558-ca0760d08222" />
