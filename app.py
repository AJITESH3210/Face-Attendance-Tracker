from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import os
import numpy as np
import datetime

app = Flask(__name__)

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f'known_faces/{filename}')
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as name

load_known_faces()

attendance_records = []

def generate_frames():
    video_capture = cv2.VideoCapture(0)  # Use the default camera

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Log attendance if the person is recognized
                if name not in attendance_records:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_records.append(name)
                    with open('attendance.txt', 'a') as f:
                        f.write(f"{name}, {timestamp}\n")

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    records = []
    if os.path.exists('attendance.txt'):
        with open('attendance.txt', 'r') as f:
            records = f.readlines()
    return render_template('attendance.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)