import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Detect and initialize the TPU
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', resolver.master())
except ValueError:
    print('TPU not found')

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the CNN model for TPU
def create_model():
    with tpu_strategy.scope():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate model
def train_evaluate_model(hardware):
    if hardware == 'TPU':
        strategy = tpu_strategy
    else:
        strategy = tf.device(hardware)

    with strategy.scope():
        model = create_model()
        batch_size = 128
        epochs = 5
        start_time = time.time()
        model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
        training_time = time.time() - start_time
        start_time = time.time()
        _, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        inference_time = (time.time() - start_time) / len(test_images) * 1000
    return training_time, inference_time, accuracy

# Train and evaluate on CPU, GPU (if available), and TPU
cpu_metrics = train_evaluate_model('/cpu:0')
print("CPU Metrics: Training Time (s), Inference Time (ms/sample), Accuracy (%)")
print(cpu_metrics)

if tf.config.list_physical_devices('GPU'):
    gpu_metrics = train_evaluate_model('/gpu:0')
    print("GPU Metrics: Training Time (s), Inference Time (ms/sample), Accuracy (%)")
    print(gpu_metrics)
else:
    print("GPU not available")

tpu_metrics = train_evaluate_model('TPU')
print("TPU Metrics: Training Time (s), Inference Time (ms/sample), Accuracy (%)")
print(tpu_metrics)
