import streamlit as st
import cv2
import time
import joblib
import pandas as pd
import sys
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detection import FaceDetector
from features import extract_features


st.set_page_config(
    page_title="Student Attention & Engagement",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


MODEL_PATH = ROOT / "models" / "attention_model.pkl"

CALIBRATION_SECONDS = 3.0
HISTORY_SECONDS = 2.0

EYE_CLOSED_THRESHOLD = 0.18

DISTRACTION_CONFIDENCE = 0.70


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


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


def get_state_icon(state):

    icons = {
        "Focused": "🟢",
        "Writing": "✍️",
        "Listening": "👂",
        "Distracted": "🔴",
        "Away": "🟠",
        "Sleepy/Tired": "😴"
    }

    return icons.get(
        state,
        "⚪"
    )


def get_state_description(state):

    descriptions = {
        "Focused":
            "Student is facing forward with a stable head position.",

        "Writing":
            "Student appears to be looking down while keeping the eyes open.",

        "Listening":
            "Student appears to be turning the head briefly.",

        "Distracted":
            "Sustained movement differs significantly from the normal position.",

        "Away":
            "A face has not been detected for the configured duration.",

        "Sleepy/Tired":
            "Face is down and both eyes have remained closed continuously."
    }

    return descriptions.get(
        state,
        "Monitoring visible behavior."
    )


def determine_behavior(
    features,
    baseline,
    ml_prediction,
    model_confidence
):

    if baseline is None:
        return "Focused"

    turn_difference = abs(
        features["head_turn"]
        -
        baseline["head_turn"]
    )

    vertical_difference = (
        features["head_vertical"]
        -
        baseline["head_vertical"]
    )

    tilt_difference = abs(
        features["head_tilt"]
        -
        baseline["head_tilt"]
    )

    left_ear = features["left_ear"]
    right_ear = features["right_ear"]

    eyes_closed = (
        left_ear < EYE_CLOSED_THRESHOLD
        and
        right_ear < EYE_CLOSED_THRESHOLD
    )

    eyes_open = not eyes_closed

    normal_head = (
        turn_difference < 0.16
        and
        abs(vertical_difference) < 0.12
        and
        tilt_difference < 0.25
    )

    looking_down = (
        vertical_difference > 0.10
    )

    head_turning = (
        turn_difference > 0.16
    )

    strong_movement = (
        turn_difference > 0.30
        or
        abs(vertical_difference) > 0.25
        or
        tilt_difference > 0.45
    )

    if normal_head and eyes_open:
        return "Focused"

    if looking_down and eyes_open:
        return "Writing"

    if head_turning and not strong_movement:
        return "Listening"

    if strong_movement:

        if (
            ml_prediction == "Distracted"
            and
            model_confidence >= DISTRACTION_CONFIDENCE
        ):
            return "Distracted"

    return "Focused"


def calculate_dynamic_score(
    state,
    features,
    baseline,
    model_confidence
):

    if state == "Away":
        return 0

    if state == "Sleepy/Tired":
        return 15

    score = 100.0

    if baseline is not None:

        turn_difference = abs(
            features["head_turn"]
            -
            baseline["head_turn"]
        )

        vertical_difference = abs(
            features["head_vertical"]
            -
            baseline["head_vertical"]
        )

        tilt_difference = abs(
            features["head_tilt"]
            -
            baseline["head_tilt"]
        )

        turn_penalty = clamp(
            turn_difference / 0.35 * 22,
            0,
            22
        )

        vertical_penalty = clamp(
            vertical_difference / 0.25 * 18,
            0,
            18
        )

        tilt_penalty = clamp(
            tilt_difference / 0.50 * 10,
            0,
            10
        )

        score -= (
            turn_penalty
            +
            vertical_penalty
            +
            tilt_penalty
        )

    average_ear = features["average_ear"]

    if average_ear < EYE_CLOSED_THRESHOLD:

        score -= 25

    elif average_ear < 0.22:

        score -= 8

    else:

        score += 3

    if state == "Focused":

        score += 5

    elif state == "Writing":

        score -= 5

    elif state == "Listening":

        score -= 8

    elif state == "Distracted":

        score -= (
            15
            +
            model_confidence * 20
        )

    score = clamp(
        score,
        0,
        100
    )

    return int(
        round(score)
    )


st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .state-box {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #444;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .state-title {
        font-size: 30px;
        font-weight: 700;
    }

    .state-description {
        font-size: 16px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:

    st.header("🎓 Student Monitor")

    st.divider()

    st.subheader("System")

    st.write("🟢 Model: **Random Forest**")
    st.write("👁️ Vision: **MediaPipe Face Mesh**")
    st.write("🎯 Calibration: **Personal Baseline**")
    st.write("⏱️ Analysis: **Temporal Analysis**")

    st.divider()

    st.subheader("Detection Settings")

    sleepy_limit = st.number_input(
        "Sleepy/Tired threshold (seconds)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )

    distracted_limit = st.number_input(
        "Distraction threshold (seconds)",
        min_value=1,
        max_value=30,
        value=4,
        step=1
    )

    away_limit = st.number_input(
        "Away threshold (seconds)",
        min_value=1,
        max_value=30,
        value=3,
        step=1
    )

    st.divider()

    st.info(
        "The first 3 seconds are used to learn your normal sitting position."
    )


st.markdown(
    '<div class="main-title">🎓 Real-Time Student Attention & Engagement Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Computer Vision + Random Forest + Personal Baseline + Temporal Analysis</div>',
    unsafe_allow_html=True
)


if "monitoring" not in st.session_state:

    st.session_state.monitoring = False


start_button = st.button(
    "▶ Start Monitoring",
    type="primary",
    use_container_width=True
)


if start_button:

    st.session_state.monitoring = True


if not st.session_state.monitoring:

    st.info(
        "Click **Start Monitoring** to begin webcam-based analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current State",
            "Waiting"
        )

    with col2:

        st.metric(
            "Engagement",
            "—"
        )

    with col3:

        st.metric(
            "Camera",
            "Ready"
        )

    st.divider()

    st.subheader("🔄 System Pipeline")

    st.code(
        """
Webcam
   ↓
Face Detection
   ↓
Facial Landmarks
   ↓
Feature Engineering
   ↓
Personal Baseline Calibration
   ↓
Random Forest
   ↓
Temporal Analysis
   ↓
Behavior Decision
   ↓
Dynamic Engagement Score
   ↓
Dashboard
        """,
        language="text"
    )

    st.subheader("🎯 Behavior Classes")

    st.write(
        "🟢 **Focused** — stable forward-facing behavior"
    )

    st.write(
        "✍️ **Writing** — looking down with eyes open"
    )

    st.write(
        "👂 **Listening** — brief head-turning behavior"
    )

    st.write(
        "🔴 **Distracted** — sustained strong movement"
    )

    st.write(
        "🟠 **Away** — face absent for several seconds"
    )

    st.write(
        "😴 **Sleepy/Tired** — face down + eyes closed continuously"
    )


else:

    if not MODEL_PATH.exists():

        st.error(
            "Model file not found: models/attention_model.pkl"
        )

        st.session_state.monitoring = False

        st.stop()


    model = joblib.load(
        MODEL_PATH
    )


    detector = FaceDetector()


    camera = cv2.VideoCapture(0)


    if not camera.isOpened():

        st.error(
            "Could not open webcam. Close other applications using the camera."
        )

        detector.close()

        st.session_state.monitoring = False

        st.stop()


    frame_placeholder = st.empty()


    col1, col2, col3, col4 = st.columns(4)

    state_placeholder = col1.empty()
    score_placeholder = col2.empty()
    face_placeholder = col3.empty()
    time_placeholder = col4.empty()


    st.divider()


    status_placeholder = st.empty()


    st.divider()


    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        behavior_placeholder = st.empty()


    with info_col2:

        confidence_placeholder = st.empty()


    with info_col3:

        calibration_placeholder = st.empty()


    st.divider()


    st.subheader(
        "📊 Live Session Statistics"
    )


    chart_placeholder = st.empty()


    history = deque()

    baseline_samples = []

    baseline = None

    calibration_start = time.time()

    session_start = time.time()

    away_start = None

    distraction_start = None

    sleepy_start = None

    previous_state = "Focused"

    smoothed_score = 100.0


    counts = {
        "Focused": 0,
        "Writing": 0,
        "Listening": 0,
        "Distracted": 0,
        "Away": 0,
        "Sleepy/Tired": 0
    }


    try:

        while True:

            success, frame = camera.read()


            if not success:

                break


            frame = cv2.flip(
                frame,
                1
            )


            current_time = time.time()


            landmarks = detector.detect(
                frame
            )


            state = "Focused"

            face_status = "NORMAL"

            model_prediction = "N/A"

            model_confidence = 0.0

            engagement_score = 100

            sleepy_duration = 0.0


            if landmarks is None:

                face_status = "NOT DETECTED"


                if away_start is None:

                    away_start = current_time


                away_duration = (
                    current_time
                    -
                    away_start
                )


                sleepy_start = None

                distraction_start = None

                history.clear()


                if away_duration >= away_limit:

                    state = "Away"

                else:

                    state = previous_state


                engagement_score = 0


            else:

                away_start = None


                features = extract_features(
                    landmarks
                )


                if baseline is None:

                    baseline_samples.append(
                        features
                    )


                    calibration_elapsed = (
                        current_time
                        -
                        calibration_start
                    )


                    remaining = max(
                        0,
                        CALIBRATION_SECONDS
                        -
                        calibration_elapsed
                    )


                    calibration_placeholder.info(
                        f"🔧 Calibrating normal position: {remaining:.1f}s"
                    )


                    if (
                        calibration_elapsed
                        >= CALIBRATION_SECONDS
                    ):

                        if baseline_samples:

                            baseline = {}

                            baseline["head_turn"] = (
                                sum(
                                    sample["head_turn"]
                                    for sample in baseline_samples
                                )
                                /
                                len(baseline_samples)
                            )

                            baseline["head_vertical"] = (
                                sum(
                                    sample["head_vertical"]
                                    for sample in baseline_samples
                                )
                                /
                                len(baseline_samples)
                            )

                            baseline["head_tilt"] = (
                                sum(
                                    sample["head_tilt"]
                                    for sample in baseline_samples
                                )
                                /
                                len(baseline_samples)
                            )


                        calibration_placeholder.success(
                            "✅ Calibration complete"
                        )


                if baseline is not None:

                    input_data = pd.DataFrame(
                        [[
                            features[name]
                            for name in FEATURE_NAMES
                        ]],
                        columns=FEATURE_NAMES
                    )


                    model_prediction = model.predict(
                        input_data
                    )[0]


                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model.predict_proba(
                                input_data
                            )[0]
                        )

                        model_confidence = float(
                            max(probabilities)
                        )


                    candidate_state = determine_behavior(
                        features,
                        baseline,
                        model_prediction,
                        model_confidence
                    )


                    history.append(
                        (
                            current_time,
                            candidate_state
                        )
                    )


                    while (
                        history
                        and
                        current_time
                        -
                        history[0][0]
                        >
                        HISTORY_SECONDS
                    ):

                        history.popleft()


                    recent_labels = [
                        item[1]
                        for item in history
                    ]


                    smoothed_prediction = majority_vote(
                        recent_labels
                    )


                    eyes_closed = (
                        features["left_ear"]
                        <
                        EYE_CLOSED_THRESHOLD
                        and
                        features["right_ear"]
                        <
                        EYE_CLOSED_THRESHOLD
                    )


                    face_down = (
                        features["head_vertical"]
                        -
                        baseline["head_vertical"]
                        >
                        0.10
                    )


                    if (
                        face_down
                        and
                        eyes_closed
                    ):

                        if sleepy_start is None:

                            sleepy_start = current_time


                        sleepy_duration = (
                            current_time
                            -
                            sleepy_start
                        )


                    else:

                        sleepy_start = None

                        sleepy_duration = 0.0


                    if (
                        sleepy_start is not None
                        and
                        sleepy_duration
                        >=
                        sleepy_limit
                    ):

                        state = "Sleepy/Tired"

                        distraction_start = None


                    elif (
                        smoothed_prediction
                        ==
                        "Distracted"
                    ):

                        if distraction_start is None:

                            distraction_start = current_time


                        distraction_duration = (
                            current_time
                            -
                            distraction_start
                        )


                        if (
                            distraction_duration
                            >=
                            distracted_limit
                            and
                            model_confidence
                            >=
                            DISTRACTION_CONFIDENCE
                        ):

                            state = "Distracted"

                        else:

                            state = "Focused"


                    else:

                        distraction_start = None

                        state = smoothed_prediction


                    engagement_score = calculate_dynamic_score(
                        state,
                        features,
                        baseline,
                        model_confidence
                    )


                    if previous_state == state:

                        smoothed_score = (
                            smoothed_score * 0.90
                            +
                            engagement_score * 0.10
                        )

                    else:

                        smoothed_score = (
                            smoothed_score * 0.70
                            +
                            engagement_score * 0.30
                        )


                    engagement_score = int(
                        round(
                            smoothed_score
                        )
                    )


                    eyes_text = (
                        "CLOSED"
                        if eyes_closed
                        else
                        "OPEN"
                    )


                    cv2.putText(
                        frame,
                        f"Eyes: {eyes_text}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Behavior: {state}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Model Confidence: {model_confidence * 100:.1f}%",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )


                    if sleepy_start is not None:

                        cv2.putText(
                            frame,
                            f"Sleep Timer: {sleepy_duration:.1f}s / {sleepy_limit}s",
                            (20, 195),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 165, 255),
                            2
                        )


            previous_state = state

            counts[state] += 1


            session_time = (
                current_time
                -
                session_start
            )


            minutes = int(
                session_time // 60
            )


            seconds = int(
                session_time % 60
            )


            icon = get_state_icon(
                state
            )


            state_placeholder.metric(
                "Current State",
                f"{icon} {state}"
            )


            score_placeholder.metric(
                "Engagement Score",
                f"{engagement_score}%"
            )


            face_placeholder.metric(
                "Face Detection",
                face_status
            )


            time_placeholder.metric(
                "Session Time",
                f"{minutes:02d}:{seconds:02d}"
            )


            status_placeholder.markdown(
                f"""
                <div class="state-box">

                    <div class="state-title">
                        {icon} {state}
                    </div>

                    <div class="state-description">
                        {get_state_description(state)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            behavior_placeholder.info(
                f"🎯 **Final Behavior**\n\n{state}"
            )


            confidence_placeholder.info(
                f"📈 **Model Confidence**\n\n{model_confidence * 100:.1f}%"
            )


            if baseline is not None:

                calibration_placeholder.success(
                    "🎯 **Baseline**\n\nCalibrated"
                )


            chart_data = pd.DataFrame(
                {
                    "Behavior": list(
                        counts.keys()
                    ),
                    "Frames": list(
                        counts.values()
                    )
                }
            )


            chart_placeholder.bar_chart(
                chart_data.set_index(
                    "Behavior"
                )
            )


            frame_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB",
                use_container_width=True
            )


            time.sleep(0.01)


    finally:

        camera.release()

        detector.close()

        cv2.destroyAllWindows()

        st.session_state.monitoring = False