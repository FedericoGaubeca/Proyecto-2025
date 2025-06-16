import cv2
import face_recognition
import os

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
recognized = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Solo procesar cada N frames
    if frame_count % process_every_n_frames == 0 and not recognized:
        # Reducir tamaño para acelerar procesamiento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            if True in matches:
                matched_idx = matches.index(True)
                print(known_names[matched_idx])
                recognized = True
                break
    else:
        face_locations = []

    # Dibujar rectángulos escalados al tamaño original
    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Detección y reconocimiento', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()