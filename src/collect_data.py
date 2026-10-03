import cv2
import mediapipe as mp
import csv
import os
import time

from features import extract_features


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "data/processed/features.csv"

LABELS = [
    "Focused",
    "Writing",
    "Listening",
    "Distracted",
    "Away"
]

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# CSV SETUP
# ==========================================

file_exists = os.path.exists(DATA_PATH)

csv_file = open(
    DATA_PATH,
    "a",
    newline=""
)

writer = csv.writer(csv_file)


FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "average_ear",
    "head_turn",
    "head_vertical",
    "head_tilt",
    "nose_x_normalized",
    "nose_y_normalized",
    "nose_z_normalized",
    "mouth_opening",
    "eye_center_distance",
    "face_width",
    "face_height",
    "face_detected",
    "label"
]


if not file_exists:

    writer.writerow(
        FEATURE_NAMES
    )


# ==========================================
# WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam")

    csv_file.close()
    face_mesh.close()

    exit()


# ==========================================
# MENU
# ==========================================

print()
print("=" * 50)
print(" STUDENT ATTENTION DATA COLLECTION")
print("=" * 50)

print()
print("Available behaviors:")

for i, label in enumerate(
    LABELS,
    start=1
):

    print(
        f"{i}. {label}"
    )

print("0. Exit")


choice = input(
    "\nEnter behavior number: "
)


if choice == "0":

    camera.release()
    csv_file.close()
    face_mesh.close()

    exit()


try:

    label = LABELS[
        int(choice) - 1
    ]

except:

    print("Invalid choice.")

    camera.release()
    csv_file.close()
    face_mesh.close()

    exit()


# ==========================================
# START
# ==========================================

print()
print("=" * 50)
print(f"Selected behavior: {label}")
print("=" * 50)

print()
print("Instructions:")
print("S = Start / Stop recording")
print("Q = Quit")
print()

if label == "Focused":

    print("Sit normally and look at the screen.")

elif label == "Writing":

    print("Look down naturally as if writing notes.")

elif label == "Listening":

    print("Turn your head slightly as if listening.")

elif label == "Distracted":

    print("Look away from the screen for a sustained period.")

elif label == "Away":

    print("Move completely away from the camera.")


print()
print("Press S when you are ready.")


# ==========================================
# RECORDING VARIABLES
# ==========================================

recording = False

sample_count = 0

start_time = None


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read webcam frame.")

        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe processing
    results = face_mesh.process(
        rgb_frame
    )


    face_detected = 0

    features = None


    # ======================================
    # FACE FOUND
    # ======================================

    if results.multi_face_landmarks:

        face_detected = 1

        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        features = extract_features(
            face_landmarks
        )


        # Draw landmarks
        height, width, _ = frame.shape

        for landmark in face_landmarks.landmark:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )


    # ======================================
    # RECORD DATA
    # ======================================

    if recording:

        if label == "Away":

            # Away requires NO face
            if face_detected == 0:

                writer.writerow([
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    label
                ])

                sample_count += 1


        else:

            # Other classes require face
            if face_detected == 1:

                writer.writerow([

                    features["left_ear"],

                    features["right_ear"],

                    features["average_ear"],

                    features["head_turn"],

                    features["head_vertical"],

                    features["head_tilt"],

                    features[
                        "nose_x_normalized"
                    ],

                    features[
                        "nose_y_normalized"
                    ],

                    features[
                        "nose_z_normalized"
                    ],

                    features[
                        "mouth_opening"
                    ],

                    features[
                        "eye_center_distance"
                    ],

                    features[
                        "face_width"
                    ],

                    features[
                        "face_height"
                    ],

                    face_detected,

                    label

                ])

                sample_count += 1


    # ======================================
    # DISPLAY INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"Behavior: {label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    if recording:

        status = "RECORDING"

        status_color = (
            0,
            0,
            255
        )

    else:

        status = "PAUSED"

        status_color = (
            255,
            255,
            0
        )


    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )


    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    if face_detected:

        face_status = "FACE DETECTED"

        face_color = (
            0,
            255,
            0
        )

    else:

        face_status = "NO FACE"

        face_color = (
            0,
            0,
            255
        )


    cv2.putText(
        frame,
        face_status,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        face_color,
        2
    )


    cv2.putText(
        frame,
        "S = Start/Stop | Q = Quit",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(
        "Student Attention Data Collection",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("s"):

        recording = not recording

        if recording:

            start_time = time.time()

            print(
                f"\nRecording {label}..."
            )

        else:

            print(
                f"\nPaused. Samples: {sample_count}"
            )


    elif key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

csv_file.close()

face_mesh.close()

cv2.destroyAllWindows()


print()
print("=" * 50)
print(" DATA COLLECTION COMPLETED")
print("=" * 50)

print(
    f"Behavior: {label}"
)

print(
    f"Samples collected: {sample_count}"
)

print(
    f"Saved to: {DATA_PATH}"
)

print("=" * 50)