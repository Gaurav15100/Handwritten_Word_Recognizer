import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess_canvas(image):

    # Convert PIL image to numpy
    img = np.array(image)

    # Remove faint noise
    img = np.where(
        img < 30,
        0,
        img
    ).astype(np.uint8)

    # Find character
    coords = cv2.findNonZero(img)

    if coords is None:
        return None, None

    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)

    # Crop character
    img = img[
        y:y+h,
        x:x+w
    ]

    # Preserve aspect ratio
    scale = 20.0 / max(h, w)

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    # Resize character
    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create 28x28 black canvas
    canvas_28 = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center image
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas_28[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = img

    # Saveable image
    processed_image = canvas_28.copy()
    
    # Normalize
    canvas_28 = canvas_28 / 255.0
    
    # Reshape for CNN
    canvas_28 = canvas_28.reshape(
        1,
        28,
        28,
        1
    )

    return canvas_28, processed_image


def preprocess_character(char_img):

    # Find character pixels
    coords = cv2.findNonZero(char_img)

    if coords is None:
        return None, None

    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)

    # Crop tightly
    char_img = char_img[
        y:y+h,
        x:x+w
    ]

    # Preserve aspect ratio
    scale = 20.0 / max(h, w)

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    # Resize
    char_img = cv2.resize(
        char_img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create centered 28x28 image
    canvas_28 = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas_28[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = char_img

    # Saveable image
    processed_image = canvas_28.copy()

    # Normalize
    canvas_28 = canvas_28 / 255.0

    # EMNIST correction
    
    # Reshape for CNN
    canvas_28 = canvas_28.reshape(
        1,
        28,
        28,
        1
    )
    
    return canvas_28, processed_image