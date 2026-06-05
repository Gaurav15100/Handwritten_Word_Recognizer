import numpy as np


from prediction.autocorrect import (
    get_best_match
)

from tensorflow.keras.models import load_model

from prediction.label_map import label_map

from preprocessing.preprocess import (
    preprocess_canvas,
    preprocess_character
)

from preprocessing.segmentation import (
    segment_characters
)

# ─────────────────────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────────────────────
model = load_model(
    "models/emnist_personalized.keras"
)

def predict_character(
    image,
    prediction_label
):
    
    processed, processed_image = preprocess_canvas(image)

    characters, boxes = segment_characters(np.array(image))

    if len(characters) == 0:

        prediction_label.config(
            text="Draw something first",
            fg="red"
        )

        return

    predicted_text = ""
    
    
    character_alternatives = []

    for char in characters:

        processed_char, processed_image = preprocess_character(char)

        if processed_char is None:
            continue

        prediction = model.predict(
            processed_char,
            verbose=0
        )

        prediction = prediction[0]
        
        top_indices = np.argsort(
            prediction
        )[-3:]
        
        top_indices = top_indices[::-1]

        top_predictions = []

        for index in top_indices:

            top_predictions.append(
                (
                    label_map[index],
                    prediction[index] * 100
                )
            )
        
        character_alternatives.append(
            top_predictions
        )
        
        confidence = np.max(prediction) * 100

        predicted_index = np.argmax(prediction)

        predicted_character = label_map[predicted_index]
        
        print(top_predictions)
                
        print(f"{predicted_character} : {confidence:.2f}%")

        predicted_text += predicted_character 
        
    corrected_text = get_best_match(
        predicted_text,
        character_alternatives
    )    
    
    prediction_label.config(
        text=(
            f"Predicted: {predicted_text}\n"
            f"Corrected: {corrected_text}"
        ),
        fg="green"
    )
    