import os
# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def run_cifar10_classification():
    print("Downloading and loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print("Building Convolutional Neural Network (CNN)...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Training model (this will take a few moments)...")
    # Training for 10 epochs gives a solid baseline performance for assessment
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    
    print("\n" + "="*50)
    print("              CIFAR-10 MODEL RESULTS")
    print("="*50)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test Loss:     {test_loss:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_cifar10_classification()