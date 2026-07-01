#Classification of images of clothing using Tensorflow (Fashion MNIST dataset)
#python -m pip install numpy matplotlib tensorflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class names
class_names = [
    "T-shirt/top",  
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
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
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Save and reload the model
model.save("fashion_mnist.keras")
new_model = tf.keras.models.load_model("fashion_mnist.keras")

# Predict
predictions = new_model.predict(x_test)

# -----------------------------
# Display Actual Labels
# -----------------------------
plt.figure(figsize=(15,3))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(class_names[y_test[i]], fontsize=8)
    plt.axis('off')

plt.suptitle("Actual Labels", fontsize=14)
plt.show()

# -----------------------------
# Display Predicted Labels
# -----------------------------
plt.figure(figsize=(15,3))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(class_names[np.argmax(predictions[i])], fontsize=8)
    plt.axis('off')

plt.suptitle("Predicted Labels", fontsize=14)
plt.show()

# -----------------------------
# Plot Accuracy Graph
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()