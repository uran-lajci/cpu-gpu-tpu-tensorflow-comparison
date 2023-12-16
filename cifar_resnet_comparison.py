import time

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def create_model(input_shape, num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=input_shape))
    x = Flatten()(base_model.output)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_evaluate_model(hardware):
    model = create_model((32, 32, 3), 10)

    batch_size = 128
    epochs = 10
    start_time = time.time()
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    training_time = time.time() - start_time
    start_time = time.time()
    _, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    inference_time = (time.time() - start_time) / len(test_images) * 1000

    return training_time, inference_time, accuracy


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    cpu_metrics = train_evaluate_model('/cpu:0')
    print("CPU Metrics: Training Time (s), Inference Time (ms/sample), Accuracy (%)")
    print(cpu_metrics)

    if tf.config.list_physical_devices('GPU'):
        gpu_metrics = train_evaluate_model('/gpu:0')
        print("GPU Metrics: Training Time (s), Inference Time (ms/sample), Accuracy (%)")
        print(gpu_metrics)
    else:
        print("GPU not available")
