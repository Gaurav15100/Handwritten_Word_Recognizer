import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

model = load_model(
    "models/emnist_cnn_model_v3.keras"
)

label_map = {
    '0':0, '1':1, '2':2, '3':3, '4':4,
    '5':5, '6':6, '7':7, '8':8, '9':9,

    'A':10, 'B':11, 'C':12, 'D':13, 'E':14,
    'F':15, 'G':16, 'H':17, 'I':18, 'J':19,
    'K':20, 'L':21, 'M':22, 'N':23, 'O':24,
    'P':25, 'Q':26, 'R':27, 'S':28, 'T':29,
    'U':30, 'V':31, 'W':32, 'X':33, 'Y':34,
    'Z':35,

    'a':36,
    'b':37,
    'd':38,
    'e':39,
    'f':40,
    'g':41,
    'h':42,
    'n':43,
    'q':44,
    'r':45,
    't':46,
}

images = []
labels = []

dataset_path = "data/custom_letters"

for folder_name in os.listdir(dataset_path):

    folder_path = os.path.join(
        dataset_path,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue
    
    if folder_name.endswith("_lower"):

        character = folder_name[0]

    elif folder_name.endswith("_upper"):

        character = folder_name[0].upper()

    else:
        continue
    
    label = label_map[character]

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path,
            image_name
        )
        
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue
        
        image = image / 255.0
        
        image = image.reshape(
            28,
            28,
            1
        )
        
        images.append(image)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

y_train = to_categorical(
    y_train,
    num_classes=47
)

y_test = to_categorical(
    y_test,
    num_classes=47
)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print(
    f"\nBefore Fine-Tuning Accuracy: "
    f"{accuracy * 100:.2f}%"
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,

    validation_data=(
        x_test,
        y_test
    ),

    epochs=10,
    batch_size=16
)

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print(
    f"\nAfter Fine-Tuning Accuracy: "
    f"{accuracy * 100:.2f}%"
)

model.save(
    "models/emnist_personalized.keras"
)

print(
    "\nPersonalized model saved successfully."
)