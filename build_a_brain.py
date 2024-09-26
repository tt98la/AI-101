import tensorflow as tf
import matplotlib.pyplot as plt

model = None
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
number_of_classes = train_labels.max() + 1  # aka number of neurons / categories

def main():
    create_model()
    verify_model()
    train_model()
    evaluate_model()
    classify_image()

def create_model():
    #create the layers
    global model
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes)
    ])

def verify_model():
    model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True)

def train_model():
    model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

def evaluate_model(epochs=5):
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        verbose=True,
        validation_data=(valid_images, valid_labels)
    )

def classify_image(data_idx=42):
    plot_image(data_idx)

    x_values = range(number_of_classes)
    plt.figure()
    plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1]).flatten())
    plt.xticks(range(number_of_classes))
    plt.show()
    
    print("correct answer:", train_labels[data_idx])

def plot_image(data_idx):
    plt.figure()
    plt.imshow(train_images[data_idx], cmap='gray')
    plt.colorbar()
    plt.grid(False)
    plt.show()

if (__name__ == "__main__"):
    main()
