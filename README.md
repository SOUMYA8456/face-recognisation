import cv2
import numpy as np
import os
from fer import FER

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture_images(user_id):
    """Capture images from the webcam and save them for training."""
    cam = cv2.VideoCapture(0)
    count = 0
    while count < 100:  # Capture 100 images
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Capturing Images', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def train_recognizer():
    """Train the face recognizer using captured images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []
    
    for image in os.listdir('dataset'):
        img_path = os.path.join('dataset', image)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        parts = image.split('.')
        if len(parts) < 3 or not parts[1].isdigit():
            continue  # Skip invalid filenames
        id = int(parts[1])  # Extract user ID
        faces.append(img)
        ids.append(id)
    
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')

def recognize_faces():
    """Recognize faces and detect emotions in real-time."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    emo_detector = FER(mtcnn=True)
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(roi_gray)
            if confidence < 100:
                name = f"User {id}"
            else:
                name = "Unknown"
            
            # Detect emotions
            emotion_data = emo_detector.top_emotion(frame[y:y+h, x:x+w])
            
            # Check if emotion_data is valid
            if emotion_data:
                emotion, score = emotion_data  # Unpack the tuple
                if score is not None:  # Ensure score is not None
                    score *= 100  # Convert to percentage
                    cv2.putText(frame, f"{emotion.capitalize()}: {score:.2f}%", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green color
                else:
                    cv2.putText(frame, "Emotion detected, but no score", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green color
            else:
                cv2.putText(frame, "No Emotion Detected", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red color
            
            # Draw a blue rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue color
            
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White color for name
        
        cv2.imshow('Recognizing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure the dataset directory exists
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Step 1: Capture images for a new user (replace '1' with the user ID)
    user_id = input("Enter user ID: ")
    capture_images(user_id)

    # Step 2: Train the recognizer
    train_recognizer()

    # Step 3: Recognize faces and detect emotions in real-time
    recognize_faces()
