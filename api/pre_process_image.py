import cv2
import mediapipe as mp
import math
import requests
import numpy as np
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def crop_and_rotate_image(image_path):
    response = requests.get(image_path)
    distance_threshold = 100
    extra_distance_percent = 20
    # Load the image
    # image = cv2.imread(image_path)
    # image = np.array(bytearray(response.content), dtype=np.uint8)
    image = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)


    if image is None:
        print("Failed to load image:", image_path)
        return None

    # Initialize Mediapipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Process the image with FaceMesh
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("No face landmarks found in the image.")
        return None

    # Get the landmarks of the first face
    face_landmarks = results.multi_face_landmarks[0]

    # Get the coordinates of nodes 10, 1, and 152
    x10 = face_landmarks.landmark[10].x * image.shape[1]
    y10 = face_landmarks.landmark[10].y * image.shape[0]
    x1 = face_landmarks.landmark[1].x * image.shape[1]
    y1 = face_landmarks.landmark[1].y * image.shape[0]
    x152 = face_landmarks.landmark[152].x * image.shape[1]
    y152 = face_landmarks.landmark[152].y * image.shape[0]

    # Calculate the distance between nodes 10 and 152
    distance_10_152 = calculate_distance(x10, y10, x152, y152)

    # Calculate the extra distance along the x-axis
    extra_distance = int(distance_10_152 * extra_distance_percent / 100)

    # Calculate the width and height based on the distance and extra distance
    width = int(distance_10_152 + (2 * extra_distance))
    height = int(distance_10_152 + (2 * extra_distance))

    # Calculate the center x-coordinate of the line connecting node 10 and node 152
    center_x = int((x10 + x152) / 2)

    # Calculate the center y-coordinate of the line connecting node 10 and node 152
    center_y = int((y10 + y152) / 2)

    # Calculate the angle between the line connecting node 10 and node 152 and the x-axis
    angle = math.degrees(math.atan2(y152 - y10, x152 - x10)) - 90

    # Rotate the image around the center point between node 10 and node 152
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Calculate the top-left and bottom-right coordinates for cropping
    crop_x = max(0, center_x - (width // 2))
    crop_y = max(0, center_y - (height // 2))
    crop_w = width
    crop_h = height

    # Perform cropping
    cropped_image = rotated_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # Check if the width is not equal to the height
    if cropped_image.shape[1] != cropped_image.shape[0]:
        # Determine the maximum dimension
        max_dim = max(cropped_image.shape[0], cropped_image.shape[1])
        # Create a new canvas with dimensions equal to the maximum dimension
        new_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        # Calculate the padding needed on both sides to make the image square
        pad_left = (max_dim - cropped_image.shape[1]) // 2
        pad_right = max_dim - cropped_image.shape[1] - pad_left
        pad_top = (max_dim - cropped_image.shape[0]) // 2
        pad_bottom = max_dim - cropped_image.shape[0] - pad_top
        # Paste the cropped image on the new canvas
        new_canvas[pad_top:pad_top+cropped_image.shape[0], pad_left:pad_left+cropped_image.shape[1]] = cropped_image
        # Update the cropped image
        cropped_image = new_canvas

    return cropped_image