import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


S = StratifiedKFold(n_splits=5)
all_histories = []

# and here
for fold_idx, (train_idx, test_idx) in enumerate(S.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20, batch_size=32, verbose=0 
    )
    
    all_histories.append(history.history['val_accuracy'])
    print(f"Fold {fold_idx + 1}: Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

plt.figure(figsize=(10, 6))
for i, val_acc in enumerate(all_histories):
    plt.plot(val_acc, label=f'Fold {i + 1}')
plt.title('Validation Accuracy across 5 Folds (StratifiedKFold)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()