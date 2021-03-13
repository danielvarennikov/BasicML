import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt

print(tf.__version__)


def get_dataset(url, filename, size):

    tf.keras.utils.get_file(filename, url,
                            untar=True, cache_dir='.',
                            cache_subdir='')
    batch_size = size

    return [
        tf.keras.preprocessing.text_dataset_from_directory('train', batch_size=batch_size, validation_split=0.2, subset='training', seed=42),
        tf.keras.preprocessing.text_dataset_from_directory('train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=42),
        tf.keras.preprocessing.text_dataset_from_directory('test', batch_size=batch_size)
    ]


def print_dataset(dataset, size):
    print("Languages: ")
    print(dataset.class_names)
    for text_batch, label_batch in dataset.take(1):
        for i in range(size):
            print("Set number: " + str(i))
            print(text_batch.numpy()[i])
            print(label_batch.numpy()[i])
            print()


def eval_model(current_model, dataset, size):
    loss, accuracy = current_model.evaluate(dataset)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)



set_size = 32
train_reruns = 2
our_dataset = get_dataset("http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz", "stack_overflow_16k.tar.gz", set_size)

raw_train_ds = our_dataset[0]
raw_val_ds = our_dataset[1]
raw_test_ds = our_dataset[2]

print_dataset(raw_train_ds, set_size)

# Prepare data for training
max_features = 5000
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Vectorise the data
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Configure dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the model
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(4)])

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=train_reruns)

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']


