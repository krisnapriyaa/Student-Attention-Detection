import numpy as np


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

NOSE = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
FOREHEAD = 10

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def distance(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z

    return np.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


def calculate_ear(landmarks, eye_points):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 +
        vertical_2
    ) / (
        2.0 * horizontal
    )

    return float(ear)


def extract_features(landmarks):
    nose = landmarks[NOSE]
    chin = landmarks[CHIN]

    left_ear_point = landmarks[LEFT_EAR]
    right_ear_point = landmarks[RIGHT_EAR]
    forehead = landmarks[FOREHEAD]

    face_width = distance(
        left_ear_point,
        right_ear_point
    )

    face_height = distance(
        forehead,
        chin
    )

    if face_width == 0:
        face_width = 0.000001

    if face_height == 0:
        face_height = 0.000001

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE
    )

    average_ear = (
        left_ear +
        right_ear
    ) / 2.0

    nose_x_normalized = (
        nose.x - 0.5
    )

    nose_y_normalized = (
        nose.y - 0.5
    )

    nose_z_normalized = nose.z

    face_center_x = (
        left_ear_point.x +
        right_ear_point.x
    ) / 2.0

    head_turn = (
        nose.x -
        face_center_x
    ) / face_width

    face_center_y = (
        forehead.y +
        chin.y
    ) / 2.0

    head_vertical = (
        nose.y -
        face_center_y
    ) / face_height

    dx = (
        right_ear_point.x -
        left_ear_point.x
    )

    dy = (
        right_ear_point.y -
        left_ear_point.y
    )

    head_tilt = np.arctan2(
        dy,
        dx
    )

    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]

    mouth_top = landmarks[MOUTH_TOP]
    mouth_bottom = landmarks[MOUTH_BOTTOM]

    mouth_width = distance(
        mouth_left,
        mouth_right
    )

    mouth_height = distance(
        mouth_top,
        mouth_bottom
    )

    if mouth_width == 0:
        mouth_width = 0.000001

    mouth_opening = (
        mouth_height /
        mouth_width
    )

    left_eye_center = landmarks[
        LEFT_EYE[0]
    ]

    right_eye_center = landmarks[
        RIGHT_EYE[0]
    ]

    eye_center_distance = (
        distance(
            left_eye_center,
            right_eye_center
        ) /
        face_width
    )

    features = {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "average_ear": float(average_ear),
        "head_turn": float(head_turn),
        "head_vertical": float(head_vertical),
        "head_tilt": float(head_tilt),
        "nose_x_normalized": float(
            nose_x_normalized
        ),
        "nose_y_normalized": float(
            nose_y_normalized
        ),
        "nose_z_normalized": float(
            nose_z_normalized
        ),
        "mouth_opening": float(
            mouth_opening
        ),
        "eye_center_distance": float(
            eye_center_distance
        ),
        "face_width": float(
            face_width
        ),
        "face_height": float(
            face_height
        ),
        "face_detected": 1
    }

    return features