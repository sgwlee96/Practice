import tensorflow as tf
# from tensorflow.python.keras import layers, models
# from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
# from tensorflow.keras import datasets, layers, models
import cv2
import matplotlib.pyplot as plt

# print(tf.__version__)

# load datasets
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.cifar10.load_data()

# split train into train and validate
train_img, validate_img, train_label, validate_label = train_test_split(train_img, train_label, test_size=0.2, random_state=777)

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.grid(False)
#     plt.imshow(train_img[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_label[i][0]])

# plt.show()

# Normalize images
train_img, test_img = train_img/255.0, test_img/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)

])


model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_img, train_label, epochs=10, validation_data=(validate_img, validate_label))

test_loss, test_accuracy = model.evaluate(test_img, test_label)

print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")