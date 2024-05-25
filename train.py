from tensorflow.keras import datasets, layers, models


def train():
    # Load CIFAR-10 dataset
    train_images, train_labels, test_images, test_labels = datasets.cifar10.load_data()

    # Normalize pixel values to range [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Create a convolutional neural network (CNN) model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    model.save("ann.h5")
