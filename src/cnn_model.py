from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape images for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation
datagen = ImageDataGenerator(

    rotation_range=8,

    width_shift_range=0.08,

    height_shift_range=0.08,

    zoom_range=0.08,

)

datagen.fit(x_train)

# Early stopping
early_stop = EarlyStopping(

    monitor='val_loss',

    patience=3,

    restore_best_weights=True
)

# Build improved CNN model
model = Sequential([

    # First CNN block
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),
    
    BatchNormalization(),

    Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),
    
    BatchNormalization(),

    MaxPooling2D((2, 2)),

    Dropout(0.25),

    # Second CNN block
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    
    BatchNormalization(),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    
    BatchNormalization(),

    MaxPooling2D((2, 2)),

    Dropout(0.25),

    # Dense layers
    Flatten(),

    Dense(
        128,
        activation='relu'
    ),
    
    BatchNormalization(),

    Dropout(0.3),

    Dense(
        10,
        activation='softmax'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show architecture
model.summary()

# Train improved CNN model
history = model.fit(

    datagen.flow(x_train, y_train, batch_size=32),

    epochs=20,

    validation_data=(x_test, y_test),

    callbacks=[early_stop]
)

# Evaluate CNN model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Save CNN model
model.save("models/cnn_model.keras")

print("CNN model saved successfully.")