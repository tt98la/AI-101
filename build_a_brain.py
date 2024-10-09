# import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np

epochs = 5
model = keras.Sequential
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
number_of_classes = train_labels.max() + 1  # aka number of neurons / categories
labels = ["T-shirt/top", "Trouser", "Pullover3", "Dress4", "Coat5", "Sandal6", "Shirt7", "Sneaker8", "Bag9", "Ankle boot"]

def main():
    create_model()
    verify_model()
    train_model()
    
    for i in range(10):
        classify_image(np.random.randint(0, len(train_images)))

def create_model():
    #create the layers
    global model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(number_of_classes)
    ])

def verify_model():
    model.summary()

def train_model():
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])

    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        verbose=True,
        validation_data=(valid_images, valid_labels))

def classify_image(data_idx=42):
    plot_image(data_idx)
    pred = model.predict(train_images[data_idx:data_idx+1])

    x_values = range(number_of_classes)
    plt.figure()
    plt.bar(x_values, pred.flatten())
    plt.xticks(range(number_of_classes), labels=labels, rotation=20)
    plt.show()
    
    print("correct answer:", labels[pred.argmax()])

def plot_image(data_idx):
    plt.figure()
    plt.imshow(train_images[data_idx], cmap='gray')
    plt.colorbar()
    plt.grid(False)
    plt.show()

if (__name__ == "__main__"):
    main()
