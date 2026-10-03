
import cv2
import time
import joblib
import pandas as pd
from collections import deque

from detection import FaceDetector
from features import extract_features


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
    "face_detected"
]

MODEL_PATH = "models/attention_model.pkl"

HISTORY_SECONDS = 2.0
AWAY_SECONDS = 3.0
DISTRACTION_SECONDS = 4.0
SLEEPY_SECONDS = 30.0

FACE_DOWN_THRESHOLD = 0.20
EYE_CLOSED_THRESHOLD = 0.18


def is_face_down(features):
    return features["head_vertical"] >= FACE_DOWN_THRESHOLD


def are_eyes_closed(features):
    return (
        features["left_ear"] < EYE_CLOSED_THRESHOLD
        and features["right_ear"] < EYE_CLOSED_THRESHOLD
    )


def majority_vote(labels):

    if not labels:
        return "Focused"

    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return max(
        counts,
        key=counts.get
    )


def get_engagement_score(state):

    scores = {
        "Focused": 100,
        "Writing": 90,
        "Listening": 85,
        "Distracted": 40,
        "Away": 0,
        "Sleepy/Tired": 10
    }

    return scores.get(state, 50)


def main():

    model = joblib.load(MODEL_PATH)

    if hasattr(model, "n_features_in_"):

        if model.n_features_in_ != len(FEATURE_NAMES):

            print(
                "ERROR: Model feature mismatch."
            )

            print(
                f"Model expects: {model.n_features_in_}"
            )

            print(
                f"Prediction provides: {len(FEATURE_NAMES)}"
            )

            return

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        return

    history = deque()

    away_start = None
    distraction_start = None
    sleepy_start = None

    session_start = time.time()

    total_frames = 0
    focused_frames = 0
    writing_frames = 0
    listening_frames = 0
    distracted_frames = 0
    away_frames = 0
    sleepy_frames = 0

    print("REAL-TIME STUDENT ATTENTION DETECTION STARTED")
    print("Press Q to quit")

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read webcam frame.")

            break

        current_time = time.time()

        total_frames += 1

        landmarks = detector.detect(frame)

        state = "Focused"

        face_status = "NORMAL"
        eyes_status = "OPEN"

        sleepy_duration = 0.0

        face_down = False
        eyes_closed = False

        if landmarks is None:

            face_status = "NOT DETECTED"

            if away_start is None:
                away_start = current_time

            away_duration = current_time - away_start

            history.clear()

            distraction_start = None
            sleepy_start = None

            if away_duration >= AWAY_SECONDS:
                state = "Away"
            else:
                state = "Focused"

        else:

            away_start = None

            features = extract_features(
                landmarks
            )

            face_down = is_face_down(
                features
            )

            eyes_closed = are_eyes_closed(
                features
            )

            if face_down:
                face_status = "DOWN"
            else:
                face_status = "NORMAL"

            if eyes_closed:
                eyes_status = "CLOSED"
            else:
                eyes_status = "OPEN"

            sleepy_condition = (
                face_down
                and eyes_closed
            )

            if sleepy_condition:

                if sleepy_start is None:
                    sleepy_start = current_time

                sleepy_duration = (
                    current_time - sleepy_start
                )

            else:

                sleepy_start = None
                sleepy_duration = 0.0

            input_values = [
                features[name]
                for name in FEATURE_NAMES
            ]

            input_data = pd.DataFrame(
                [input_values],
                columns=FEATURE_NAMES
            )

            prediction = model.predict(
                input_data
            )[0]

            history.append(
                (
                    current_time,
                    prediction
                )
            )

            while history:

                oldest_time = history[0][0]

                if (
                    current_time
                    - oldest_time
                    <= HISTORY_SECONDS
                ):
                    break

                history.popleft()

            recent_predictions = [
                item[1]
                for item in history
            ]

            smoothed_prediction = majority_vote(
                recent_predictions
            )

            if sleepy_duration > SLEEPY_SECONDS:

                state = "Sleepy/Tired"

                distraction_start = None

            elif smoothed_prediction == "Distracted":

                if distraction_start is None:
                    distraction_start = current_time

                distraction_duration = (
                    current_time
                    - distraction_start
                )

                if (
                    distraction_duration
                    >= DISTRACTION_SECONDS
                ):

                    state = "Distracted"

                else:

                    state = "Focused"

            else:

                distraction_start = None

                state = smoothed_prediction

        if state == "Focused":

            focused_frames += 1

        elif state == "Writing":

            writing_frames += 1

        elif state == "Listening":

            listening_frames += 1

        elif state == "Distracted":

            distracted_frames += 1

        elif state == "Away":

            away_frames += 1

        elif state == "Sleepy/Tired":

            sleepy_frames += 1

        engagement = get_engagement_score(
            state
        )

        session_time = (
            current_time
            - session_start
        )

        minutes = int(
            session_time // 60
        )

        seconds = int(
            session_time % 60
        )

        cv2.putText(
            frame,
            f"State: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Face: {face_status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eyes_status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Engagement: {engagement}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Session: {minutes:02d}:{seconds:02d}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if (
            landmarks is not None
            and sleepy_start is not None
        ):

            cv2.putText(
                frame,
                f"Sleep Timer: {sleepy_duration:.1f}s / 30s",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        cv2.imshow(
            "Real-Time Student Attention Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()

    detector.close()

    cv2.destroyAllWindows()

    print()
    print("REAL-TIME DETECTION STOPPED")
    print()
    print("SESSION SUMMARY")
    print("----------------")
    print(
        f"Total frames: {total_frames}"
    )
    print(
        f"Focused frames: {focused_frames}"
    )
    print(
        f"Writing frames: {writing_frames}"
    )
    print(
        f"Listening frames: {listening_frames}"
    )
    print(
        f"Distracted frames: {distracted_frames}"
    )
    print(
        f"Away frames: {away_frames}"
    )
    print(
        f"Sleepy/Tired frames: {sleepy_frames}"
    )


if __name__ == "__main__":
    main()

