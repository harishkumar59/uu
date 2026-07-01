import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),   # 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),   # 128 neurons
    tf.keras.layers.Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model parameters
EPOCHS = 10
BATCH_SIZE = 32

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"\nTest Loss: {val_loss:.4f}    Test Accuracy: {val_acc:.4f}")

# Save and reload the model
model.save('mnist.keras')
new_model = tf.keras.models.load_model('mnist.keras')

# Make and display predictions on the first 10 test samples
predictions = new_model.predict(x_test)

for i in range(10):
    pred_label = np.argmax(predictions[i])
    print(f"Sample {i} — Predicted: {pred_label}, True: {y_test[i]}")
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.show()

# Plot training & validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()