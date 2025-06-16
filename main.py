import cv2
import face_recognition
import os
import time

# Cargar imágenes y codificaciones
known_faces = []
known_names = []
for filename in os.listdir('faces'):
    path = os.path.join('faces', filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

cap = cv2.VideoCapture(0)
process_every_n_frames = 3
frame_count = 0
last_recognition_time = 0
RECOGNITION_COOLDOWN = 10  # segundos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')

    # Dibujar rectángulos antes de decidir si se reconoce
    for (top, right, bottom, left) in face_locations:
        top_scaled = top * 4
        right_scaled = right * 4
        bottom_scaled = bottom * 4
        left_scaled = left * 4
        cv2.rectangle(frame, (left_scaled, top_scaled), (right_scaled, bottom_scaled), (0, 255, 0), 2)

    # Si pasó suficiente tiempo, hacer reconocimiento
    if frame_count % process_every_n_frames == 0 and (current_time - last_recognition_time > RECOGNITION_COOLDOWN):
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            if True in matches:
                matched_idx = matches.index(True)
                print(known_names[matched_idx])
                last_recognition_time = current_time
                break

    # Mostrar temporizador en pantalla
    cooldown_remaining = max(0, int(RECOGNITION_COOLDOWN - (current_time - last_recognition_time)))
    cv2.putText(frame, f'Cooldown: {cooldown_remaining}s', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Detección y reconocimiento', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
