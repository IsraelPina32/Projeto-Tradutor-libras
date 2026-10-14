"""Extração de pontos (keypoints) com MediaPipe Holistic.

IMPORTANTE: a mesma classe deve ser usada na coleta e na predição em tempo
real, para que os dados tenham sempre a mesma orientação (imagem espelhada)
e o mesmo formato de vetor.

Vetor de cada frame (NUM_FEATURES números):
    pose (33x4) | mão esquerda (21x3) | mão direita (21x3) | rosto (FACE_INDICES x3)
Quando uma parte não é detectada, ela vira zeros.
"""
import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    FACE_FEATURES,
    FACE_INDICES,
    HAND_FEATURES,
    NUM_FEATURES,
    POSE_FEATURES,
)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class KeypointExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Espelha o frame (visão de selfie) e devolve (frame_espelhado, results)."""
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        return frame, results

    @staticmethod
    def to_vector(results) -> np.ndarray:
        """Converte os resultados em um vetor de tamanho NUM_FEATURES."""
        if results.pose_landmarks:
            pose = np.array(
                [[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]
            ).flatten()
        else:
            pose = np.zeros(POSE_FEATURES)

        def hand(landmarks):
            if landmarks:
                return np.array([[p.x, p.y, p.z] for p in landmarks.landmark]).flatten()
            return np.zeros(HAND_FEATURES)

        if results.face_landmarks:
            lm = results.face_landmarks.landmark
            face = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in FACE_INDICES]).flatten()
        else:
            face = np.zeros(FACE_FEATURES)

        vec = np.concatenate(
            [pose, hand(results.left_hand_landmarks), hand(results.right_hand_landmarks), face]
        )
        assert vec.shape[0] == NUM_FEATURES
        return vec

    @staticmethod
    def has_hands(results) -> bool:
        return bool(results.left_hand_landmarks or results.right_hand_landmarks)

    @staticmethod
    def draw(frame, results):
        """Desenha os pontos detectados sobre o frame (apenas visual)."""
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            h, w = frame.shape[:2]
            lm = results.face_landmarks.landmark
            for i in FACE_INDICES:
                cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 1, (0, 255, 255), -1)

    def close(self):
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
