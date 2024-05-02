import tensorflow as tf
from keras import datasets
from sklearn.model_selection import train_test_split


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def load_data():
    (train_img, train_label), (test_img, test_label) = datasets.cifar10.load_data()


    train_img, test_img = train_img / 255.0, test_img/255.0

    train_img, validate_img, train_label, validate_label = train_test_split(train_img, train_label, test_size=0.2, random_state=777)

    return train_img, train_label, validate_img, validate_label, test_img, test_label
