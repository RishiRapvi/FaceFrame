import cv2
import mediapipe as mp
import math

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Define landmark indices for key facial metrics
LANDMARKS = {
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
    "jaw_left": 234,
    "jaw_right": 454,
    "mid_forehead": 151,
    "mid_chin": 199
}

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def classify_face_shape(width, height):
    ratio = height / width
    if ratio > 1.5:
        return "Oval", "Round or aviator frames suit you best."
    elif abs(height - width) < 0.05:
        return "Round", "Rectangle or square frames will balance your features."
    elif width > height:
        return "Square", "Round or oval glasses add softness."
    else:
        return "Heart", "Bottom-heavy frames balance your look."

# Load and process image
image_path = "test_images/Snapchat-499812639.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if not results.multi_face_landmarks:
    print("No face detected.")
    exit()

landmarks = results.multi_face_landmarks[0].landmark

# Measure key distances
face_width = distance(landmarks[LANDMARKS["left_cheek"]], landmarks[LANDMARKS["right_cheek"]])
face_height = distance(landmarks[LANDMARKS["forehead"]], landmarks[LANDMARKS["chin"]])

shape, suggestion = classify_face_shape(face_width, face_height)

# Draw text on image
cv2.putText(image, f"Shape: {shape}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(image, suggestion, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Save output
cv2.imwrite("face_shape_result.jpg", image)
print(f"Face shape: {shape}\nSuggestion: {suggestion}\nSaved as face_shape_result.jpg")
