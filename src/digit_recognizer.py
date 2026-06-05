# Improved Styled GUI (Canvas Position Preserved)
import tkinter as tk
from PIL import ImageGrab
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Create window
window = tk.Tk()

window.title("Digit Recognizer")

window.geometry("500x700")

window.configure(bg="#1e1e1e")

# Load model
model = load_model("models/cnn_model.keras")

# Title
heading = tk.Label(
    window,
    text="Digit Recognizer",
    font=("Arial", 24, "bold"),
    bg="#1e1e1e",
    fg="white"
)

heading.pack(pady=20)

# Subtitle
subtitle = tk.Label(
    window,
    text="Draw a digit from 0-9",
    font=("Arial", 14),
    bg="#1e1e1e",
    fg="#cfcfcf"
)

subtitle.pack(pady=5)

# Create canvas
canvas = tk.Canvas(
    window,
    width=300,
    height=300,
    bg="black",
    highlightthickness=2,
    highlightbackground="#444444"
)

canvas.pack(pady=25)

# Prediction label
prediction_label = tk.Label(
    window,
    text="Prediction: ",
    font=("Arial", 22, "bold"),
    bg="#1e1e1e",
    fg="#00ff99"
)

prediction_label.pack(pady=15)

# Status label
status_label = tk.Label(
    window,
    text="Ready",
    font=("Arial", 12),
    bg="#1e1e1e",
    fg="#bbbbbb"
)

status_label.pack(pady=5)

last_x = None
last_y = None

# Drawing function
def draw(event):

    global last_x, last_y

    if last_x is not None and last_y is not None:

        canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            fill="white",
            width=10,
            smooth=True,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND
        )

    last_x = event.x
    last_y = event.y
    
def reset_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None

# Save and predict function
def save_canvas():
    
    # Force UI update
    window.update()

    # Canvas coordinates
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()

    width = canvas.winfo_width()
    height = canvas.winfo_height()

    x1 = x + width
    y1 = y + height

    # Capture canvas
    image = ImageGrab.grab(
        bbox=(
            x + 153,
            y + 15,
            x1 + 230,
            y1 + 120
        )
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(
        1,
        28,
        28,
        1
    )

    # Predict
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # Show prediction
    prediction_label.config(
        text=f"Prediction: {predicted_digit}"
    )

    status_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

# Clear canvas
def clear_canvas():

    canvas.delete("all")

    prediction_label.config(
        text="Prediction: "
    )

    status_label.config(
        text="Canvas Cleared"
    )

# Predict button
predict_button = tk.Button(
    window,
    text="Predict Digit",
    command=save_canvas,
    font=("Arial", 16, "bold"),
    bg="#00aa66",
    fg="white",
    activebackground="#00cc77",
    activeforeground="white",
    padx=20,
    pady=10,
    relief="flat",
    cursor="hand2"
)

predict_button.pack(pady=15)

# Clear button
clear_button = tk.Button(
    window,
    text="Clear Canvas",
    command=clear_canvas,
    font=("Arial", 14),
    bg="#aa3333",
    fg="white",
    activebackground="#cc4444",
    activeforeground="white",
    padx=15,
    pady=8,
    relief="flat",
    cursor="hand2"
)

clear_button.pack(pady=10)

# Bind drawing
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_draw)

# Run app
window.mainloop()