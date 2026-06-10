import tensorflow as tf
import matplotlib.pyplot as plt

#  Tai va preprocess du lieu 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test.astype('float32'))

#  Build MobileNetV2 Transfer Learning Model
base = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
base.trainable = False

model = tf.keras.models.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

#  Train & Evaluate
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
print(f"\nTest Acc: {model.evaluate(x_test, y_test, verbose=0)[1] * 100:.2f}%")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
