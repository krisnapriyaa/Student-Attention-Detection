
import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawing = mp.solutions.drawing_utils

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0].landmark

    def close(self):

        self.face_mesh.close()
