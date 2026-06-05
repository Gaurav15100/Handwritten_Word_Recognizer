import cv2

def merge_dot_with_stem(boxes):

    merged = []
    used = set()

    for i, (x1, y1, w1, h1) in enumerate(boxes):

        if i in used:
            continue

        best_j = None
        best_dist = float('inf')

        for j, (x2, y2, w2, h2) in enumerate(boxes):

            if i == j or j in used:
                continue

            is_small = w2 < 25 and h2 < 25
            is_above = (y2 + h2) < (y1 + h1 * 0.4)
            gap = y1 - (y2 + h2)
            close = gap < max(h1 * 0.5, 12)
            
            # Use center alignment instead of just overlap
            center1 = x1 + w1 / 2
            center2 = x2 + w2 / 2
            center_dist = abs(center1 - center2)
            aligned = center_dist < max(w1, 15)

            if is_small and is_above and close and aligned:
                if center_dist < best_dist:
                    best_dist = center_dist
                    best_j = j

        if best_j is not None:
            x2, y2, w2, h2 = boxes[best_j]
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1+w1, x2+w2) - new_x
            new_h = max(y1+h1, y2+h2) - new_y
            x1, y1, w1, h1 = new_x, new_y, new_w, new_h
            used.add(best_j)

        merged.append((x1, y1, w1, h1))
        used.add(i)

    return sorted(merged, key=lambda b: b[0])

def segment_characters(image):

    # Binary threshold
    _, thresh = cv2.threshold(
        image,
        20,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        # Ignore only actual noise
        if area < 10:
            continue

        boxes.append((x, y, w, h))

    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Merge i/j dots with their stems
    boxes = merge_dot_with_stem(boxes)

    characters = []

    for (x, y, w, h) in boxes:

        char = thresh[y:y+h, x:x+w]

        characters.append(char)
    
    return characters, boxes
