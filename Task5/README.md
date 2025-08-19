# Objective: 

Create a simple neural network in TensorFlow to classify handwritten digits from the MNIST dataset. You’ll define the model, train it on labeled data, and evaluate its performance during the training.


Instructions:

Import tensorflow and load the MNIST dataset from tf.keras.datasets.
Define the model architecture:
For the input layer, flatten the 28x28 input images.
Compile the model:
Use Adam as the optimizer.
Use accuracy as the metric.
Train the model:
Use a validation_split=0.2 to monitor the model’s performance on validation data.
Plot training results:
Plot the training and validation accuracy over epochs to visualize model performance.
Note: You can access both training and validation accuracy values from the history object returned by model.fit(). To explore the structure of history and confirm the data you need, try running it in debugging mode.
Submission:

Submit your code and plots.


![Alt text](8e6d89d5-75c6-4d32-8e8b-82e456512fff.png)