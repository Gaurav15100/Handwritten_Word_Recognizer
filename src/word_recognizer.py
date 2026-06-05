import tkinter as tk

from PIL import Image, ImageDraw

import os

from prediction.predictor import (
    predict_character
)

# ─────────────────────────────────────────────────────────────
# GUI setup
# ─────────────────────────────────────────────────────────────
window = tk.Tk()

window.title(
    "Word Recognizer"
)

window.geometry("420x520")

window.configure(bg="white")

# ─────────────────────────────────────────────────────────────
# Canvas setup
# ─────────────────────────────────────────────────────────────
canvas_width = 800
canvas_height = 250

canvas = tk.Canvas(
    window,
    width=canvas_width,
    height=canvas_height,
    bg="black",
    cursor="cross"
)

canvas.pack(pady=20)

# ─────────────────────────────────────────────────────────────
# PIL image
# ─────────────────────────────────────────────────────────────
image = Image.new(
    "L",
    (canvas_width, canvas_height),
    "black"
)

draw = ImageDraw.Draw(image)

# Track previous mouse position
last_x = None
last_y = None

# ─────────────────────────────────────────────────────────────
# Drawing function
# ─────────────────────────────────────────────────────────────
def draw_lines(event):

    global last_x, last_y, draw

    x = event.x
    y = event.y

    radius = 6

    # Draw circle
    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ),
        fill="white"
    )

    # Connect strokes
    if last_x is not None:

        canvas.create_line(
            last_x,
            last_y,
            x,
            y,
            fill="white",
            width=radius * 2,
            capstyle=tk.ROUND
        )

        draw.line(
            [
                last_x,
                last_y,
                x,
                y
            ],
            fill="white",
            width=radius * 2
        )

    last_x = x
    last_y = y

def reset_last(event):

    global last_x, last_y

    last_x = None
    last_y = None

canvas.bind(
    "<B1-Motion>",
    draw_lines
)

canvas.bind(
    "<ButtonRelease-1>",
    reset_last
)

# ─────────────────────────────────────────────────────────────
# Clear function
# ─────────────────────────────────────────────────────────────
def clear_canvas():

    canvas.delete("all")

    global image
    global draw

    image = Image.new(
        "L",
        (canvas_width, canvas_height),
        "black"
    )

    draw = ImageDraw.Draw(image)

    prediction_label.config(
        text="Draw a word",
        fg="black"
    )
    processed_preview_label.config(image="")

    processed_preview_label.image = None

# Processed 28x28 preview
processed_preview_label = tk.Label(
    window,
    bg="white"
)

processed_preview_label.pack(pady=10)

# ─────────────────────────────────────────────────────────────
# Prediction label
# ─────────────────────────────────────────────────────────────
prediction_label = tk.Label(
    window,
    text="Draw a word",
    font=("Arial", 18),
    bg="white",
    fg="black"
)

prediction_label.pack(pady=10)

# ─────────────────────────────────────────────────────────────
# Buttons
# ─────────────────────────────────────────────────────────────
button_frame = tk.Frame(
    window,
    bg="white"
)

button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 14),
    command=lambda: predict_character(
    image,
    prediction_label
    ),
    width=10
)

predict_button.grid(
    row=0,
    column=0,
    padx=10
)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 14),
    command=clear_canvas,
    width=10
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)

window.bind(
    "<Return>",
    lambda event: predict_character(
        image,
        prediction_label
    )
)

window.bind(
    "<BackSpace>",
    lambda event: clear_canvas()
)

# ─────────────────────────────────────────────────────────────
# Run GUI
# ─────────────────────────────────────────────────────────────
window.mainloop()