import cv2  # OpenCV library for image and video processing (reading frames, drawing, detection)
import os  # Operating-system utilities (used for filesystem/path operations if needed)
import numpy as np  # Numerical Python: used here to create arrays for labels and numerical operations

# Step 1: Prepare training data (reference image and label)
reference_name = "Yeduguri Sandinti Jagan Mohan Reddy"  # Human-friendly label for the known person
reference_image_path = "ys.jpg"  # Path to the reference image file (can be relative or absolute)

# Load reference image in grayscale
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)  # Read the image file as a single-channel (grayscale) image
if reference_image is None:
    # If OpenCV couldn't read the file (wrong path, corrupted file, etc.), raise a clear error
    raise FileNotFoundError(f"Could not load image: {reference_image_path}")

# Initialize face detector using Haar Cascade classifier bundled with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# cv2.data.haarcascades provides the directory where OpenCV stores cascade xml files

# Detect face(s) in reference image. detectMultiScale returns a list of rectangles (x, y, w, h)
faces = face_cascade.detectMultiScale(reference_image, scaleFactor=1.1, minNeighbors=5)
if len(faces) == 0:
    # If no face was detected in the reference image, there is nothing to train on
    raise Exception("No face detected in reference image!")

(x, y, w, h) = faces[0]  # Choose the first detected face rectangle (x=left, y=top, w=width, h=height)
reference_face = reference_image[y:y+h, x:x+w]  # Crop the detected face region from the grayscale image (numpy slicing rows, cols)

# Step 2: Train recognizer with reference face
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create an LBPH face recognizer (requires opencv-contrib-python)
# NOTE: If you see AttributeError: module 'cv2' has no attribute 'face', install the contrib package:
#    pip uninstall opencv-python
#    pip install opencv-contrib-python

# The recognizer expects a list of face images and a NumPy array of integer labels
recognizer.train([reference_face], np.array([0]))  # Train on a single image with label '0' (numeric id for the reference person)

# Step 3: Start video capture
camera = cv2.VideoCapture(0)  # Open the default camera (device index 0). Use another index or a video file path if needed.

while True:  # Main loop: read frames from camera and process them in real time
    # Capture frame
    success, current_frame = camera.read()  # camera.read() returns (ret, frame). ret==True means a frame was grabbed
    if not success:
        # If grabbing the frame failed (camera disconnected, permission issue), exit the loop
        break

    # Convert the captured BGR frame to grayscale (face detection/recognition typically works on grayscale)
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame (returns a list of rectangles (x, y, w, h))
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:  # Loop over each detected face rectangle
        # Extract the region of interest (ROI) --- the face region --- from the grayscale frame
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Predict the label and confidence for this ROI using the trained recognizer
        # recognizer.predict returns (label, confidence). For LBPH, a lower confidence value means a closer match.
        label, confidence = recognizer.predict(roi_gray)

        # Decide whether the predicted label matches the known person and whether the confidence is acceptable
        if label == 0 and confidence < 70:  # Here '0' is the numeric id we trained above; threshold 70 is heuristic
            name = reference_name  # We consider this a match and set the human-readable name
            color = (0, 255, 0)  # Green rectangle for recognized faces (BGR color tuple)
        else:
            name = "Unknown"  # Not a match (either different label or confidence too high -> low match quality)
            color = (0, 0, 255)  # Red rectangle for unknown faces

        # Draw a rectangle around the face on the original (color) frame
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), color, 2)  # top-left and bottom-right coordinates, color, line thickness

        # Put the label text (name or "Unknown") above the face rectangle
        cv2.putText(current_frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Arguments: image, text, bottom-left origin of text, font, fontScale, color (BGR), thickness

    # Step 6: Show the annotated frame to the user in a window named "Face Recognition"
    cv2.imshow("Face Recognition", current_frame)

    # Wait for 1 millisecond for a key event. If the 'q' key is pressed, exit the loop and end the program.
    # cv2.waitKey returns a 32-bit int; & 0xFF normalizes it to the low byte (portable across platforms)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release the camera resource and close all OpenCV windows
camera.release()  # Releases the VideoCapture object and frees the camera for other applications
cv2.destroyAllWindows()  # Closes all HighGUI windows created by OpenCV
