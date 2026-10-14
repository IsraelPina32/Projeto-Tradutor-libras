"""Configurações centrais do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
META_PATH = KEYPOINTS_DIR / "meta.json"
VOCAB_PATH = ROOT / "vocab.json"

# Cada amostra = SEQUENCE_LENGTH frames consecutivos (~1 s a 30 fps).
# Pode ser mudado na coleta com --frames, mas TODAS as amostras do dataset
# precisam ter o mesmo tamanho (o capture.py confere isso pelo meta.json).
SEQUENCE_LENGTH = 30
SAMPLES_PER_SIGN = 40
COUNTDOWN_SECONDS = 2

# Pontos do rosto usados (índices do MediaPipe Face Mesh):
# lábios, sobrancelhas, ponta do nariz (1) e queixo (152).
FACE_INDICES = (
    0, 1, 13, 14, 17, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 95, 105, 107, 146, 152, 178, 181, 185, 191, 267,
    269, 270, 276, 282, 283, 285, 291, 293, 295, 296, 300, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 334, 336, 375, 402, 405, 409, 415,
)

# Layout do vetor de cada frame: pose | mão esquerda | mão direita | rosto
POSE_FEATURES = 33 * 4          # x, y, z, visibilidade
HAND_FEATURES = 21 * 3          # x, y, z
FACE_FEATURES = len(FACE_INDICES) * 3
NUM_FEATURES = POSE_FEATURES + 2 * HAND_FEATURES + FACE_FEATURES  # 444

POSE_SLICE = slice(0, POSE_FEATURES)
HANDS_SLICE = slice(POSE_FEATURES, POSE_FEATURES + 2 * HAND_FEATURES)
FACE_SLICE = slice(POSE_FEATURES + 2 * HAND_FEATURES, NUM_FEATURES)

# Se menos que esta fração dos frames tiver alguma mão detectada,
# a amostra é descartada (provavelmente o sinal saiu do quadro).
MIN_HAND_FRAME_RATIO = 0.5
