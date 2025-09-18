import cv2
import face_recognition
import requests
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://centinel-ai2025.vercel.app/person/getPersonsIA"        # GET: faces list
REPORT_URL = "https://centinel-ai2025.vercel.app/person/lastRecognized"   # POST: recognized user
CAM_INDEX = 0   # 0 = default webcam, change if needed
PROCESS_EVERY_N_FRAMES = 3
RECOGNITION_COOLDOWN = 10  # seconds
# -----------------------------

# Load known faces from backend
print("Fetching faces from backend...")
resp = requests.get(API_URL)
resp.raise_for_status()
faces_data = resp.json()

known_faces = []
known_ids = []

for item in faces_data:
    person_id = item["person_ID"]
    url = item["photo"]

    try:
        img_resp = requests.get(url, stream=True)
        img_resp.raise_for_status()
        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            known_faces.append(encodings[0])
            known_ids.append(person_id)
            print(f"[OK] Loaded person_ID={person_id}")
        else:
            print(f"[WARN] No face found in {url}")

    except Exception as e:
        print(f"[ERROR] Could not load person_ID={person_id}: {e}")

print(f"Total loaded: {len(known_faces)} faces")

# Video capture
cap = cv2.VideoCapture(CAM_INDEX)
frame_count = 0
last_recognition_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')

    # Draw rectangles
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)

    # Recognition (every N frames & cooldown)
    if frame_count % PROCESS_EVERY_N_FRAMES == 0 and (current_time - last_recognition_time > RECOGNITION_COOLDOWN):
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            if True in matches:
                matched_idx = matches.index(True)
                person_id = known_ids[matched_idx]
                print(f"[RECOGNIZED] person_ID={person_id}")

                # Report to backend
                try:
                    res = requests.get(REPORT_URL, json={"person_ID": person_id})
                    if res.status_code == 200:
                        print(f"[SENT] Reported person_ID={person_id} to backend")
                    else:
                        print(f"[ERROR] Backend returned status {res.status_code}")
                except Exception as e:
                    print(f"[ERROR] Could not report recognition: {e}")

                recognized = True
                break

        if not recognized and face_encodings:
            print("Try again")

        if face_encodings:
            last_recognition_time = current_time  # reset cooldown even if not recognized

    # Show cooldown
    cooldown_remaining = max(0, int(RECOGNITION_COOLDOWN - (current_time - last_recognition_time)))
    cv2.putText(frame, f'Cooldown: {cooldown_remaining}s', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display video
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()