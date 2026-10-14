"""Coleta de dados: grava sequências de keypoints para cada sinal do vocabulário.

Uso (a partir da pasta raiz do projeto):
    python -m src.capture                          # percorre todos os sinais
    python -m src.capture --sign obrigado          # só um sinal
    python -m src.capture --sign oi sim --samples 50
    python -m src.capture --frames 45              # sequências de 45 frames
    python -m src.capture --status                 # mostra o progresso

Teclas:  ESPAÇO = gravar amostra | N = pular sinal | Q = sair
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np

from src.config import (
    COUNTDOWN_SECONDS,
    FACE_INDICES,
    KEYPOINTS_DIR,
    META_PATH,
    MIN_HAND_FRAME_RATIO,
    NUM_FEATURES,
    SAMPLES_PER_SIGN,
    SEQUENCE_LENGTH,
    VOCAB_PATH,
)
from src.extract import KeypointExtractor

WINDOW = "Coleta de sinais - Libras"


def load_vocab() -> dict:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


def count_samples(sign: str) -> int:
    return len(list((KEYPOINTS_DIR / sign).glob("*.npy")))


def next_index(sign: str) -> int:
    existing = [int(p.stem) for p in (KEYPOINTS_DIR / sign).glob("*.npy") if p.stem.isdigit()]
    return max(existing, default=-1) + 1


def ensure_meta(seq_len: int) -> None:
    """Garante que todas as amostras do dataset tenham o mesmo formato.

    Grava um meta.json na primeira coleta e, nas seguintes, recusa gravar se
    o tamanho da sequência ou o layout do vetor for diferente.
    """
    expected = {
        "sequence_length": seq_len,
        "num_features": NUM_FEATURES,
        "face_points": len(FACE_INDICES),
    }
    has_samples = any(KEYPOINTS_DIR.glob("*/*.npy"))

    if META_PATH.exists():
        saved = json.loads(META_PATH.read_text(encoding="utf-8"))
        if saved != expected:
            sys.exit(
                "O dataset existente tem outro formato:\n"
                f"  salvo:  {saved}\n"
                f"  atual:  {expected}\n"
                f"Use --frames {saved.get('sequence_length')} para continuar com o mesmo formato,\n"
                "ou apague as pastas de data/keypoints (inclusive meta.json) para recomeçar."
            )
    elif has_samples:
        sys.exit(
            "Existem amostras antigas em data/keypoints sem meta.json (formato anterior).\n"
            "Apague as pastas de sinais em data/keypoints e grave de novo."
        )
    else:
        KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def put_text(frame, text, y, color=(255, 255, 255), scale=0.7):
    # contorno preto + texto, para ler em qualquer fundo
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def read(cap, extractor):
    ok, raw = cap.read()
    if not ok:
        return None, None
    return extractor.process(raw)


def show(frame, results, extractor, lines):
    extractor.draw(frame, results)
    for i, (text, color) in enumerate(lines):
        put_text(frame, text, 35 + i * 32, color)
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(1) & 0xFF


def record_sequence(cap, extractor, header, seq_len):
    """Grava seq_len frames. Retorna a matriz (frames x features), "sem_maos" ou None."""
    sequence, with_hands = [], 0
    while len(sequence) < seq_len:
        frame, results = read(cap, extractor)
        if frame is None:
            return None
        sequence.append(extractor.to_vector(results))
        with_hands += extractor.has_hands(results)
        show(frame, results, extractor, [
            (header, (255, 255, 255)),
            (f"GRAVANDO... {len(sequence)}/{seq_len}", (0, 0, 255)),
        ])
    if with_hands / seq_len < MIN_HAND_FRAME_RATIO:
        return "sem_maos"
    return np.array(sequence)


def capture_sign(cap, extractor, sign, label, target, seq_len) -> str:
    """Coleta amostras de um sinal. Retorna 'ok', 'skip' ou 'quit'."""
    sign_dir = KEYPOINTS_DIR / sign
    sign_dir.mkdir(parents=True, exist_ok=True)

    while count_samples(sign) < target:
        have = count_samples(sign)
        header = f"Sinal: {label.upper()}  |  amostra {have + 1}/{target}"

        # espera o usuário se preparar
        while True:
            frame, results = read(cap, extractor)
            if frame is None:
                print("Não foi possível ler a câmera.")
                return "quit"
            key = show(frame, results, extractor, [
                (header, (255, 255, 255)),
                ("ESPACO = gravar | N = pular sinal | Q = sair", (0, 255, 255)),
            ])
            if key == ord(" "):
                break
            if key in (ord("n"), ord("N")):
                return "skip"
            if key in (ord("q"), ord("Q")):
                return "quit"

        # contagem regressiva
        end = time.time() + COUNTDOWN_SECONDS
        while time.time() < end:
            frame, results = read(cap, extractor)
            if frame is None:
                return "quit"
            remaining = int(end - time.time()) + 1
            show(frame, results, extractor, [
                (header, (255, 255, 255)),
                (f"Prepare-se... {remaining}", (0, 165, 255)),
            ])

        data = record_sequence(cap, extractor, header, seq_len)
        if data is None:
            print("Falha ao ler a câmera durante a gravação.")
            return "quit"
        if isinstance(data, str):  # "sem_maos"
            print("  ! Mãos quase não apareceram. Amostra descartada, tente de novo.")
            continue

        path = sign_dir / f"{next_index(sign)}.npy"
        np.save(path, data)
        print(f"  salvo: {path.relative_to(KEYPOINTS_DIR.parent.parent)}")
    return "ok"


def print_status(vocab, target):
    print(f"{'sinal':<12}{'amostras':>10}   (meta: {target})")
    for sign in vocab:
        n = count_samples(sign)
        mark = "OK" if n >= target else ""
        print(f"{sign:<12}{n:>10}   {mark}")


def main():
    parser = argparse.ArgumentParser(description="Coleta de dados de sinais de Libras")
    parser.add_argument("--sign", nargs="*", help="sinais a coletar (padrão: todos do vocab.json)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SIGN, help="amostras por sinal")
    parser.add_argument("--frames", type=int, default=SEQUENCE_LENGTH,
                        help=f"frames por amostra (padrão: {SEQUENCE_LENGTH}); igual para todo o dataset")
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera")
    parser.add_argument("--status", action="store_true", help="mostra o progresso e sai")
    args = parser.parse_args()

    vocab = load_vocab()
    if args.status:
        print_status(vocab, args.samples)
        return

    signs = args.sign or list(vocab)
    unknown = [s for s in signs if s not in vocab]
    if unknown:
        sys.exit(f"Sinais fora do vocab.json: {unknown}\nDisponíveis: {list(vocab)}")

    ensure_meta(args.frames)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Não consegui abrir a câmera. Tente outro índice com --camera 1.")

    cv2.namedWindow(WINDOW)
    try:
        with KeypointExtractor() as extractor:
            for sign in signs:
                if count_samples(sign) >= args.samples:
                    print(f"[{sign}] já tem {count_samples(sign)} amostras, pulando.")
                    continue
                print(f"[{sign}] coletando ({count_samples(sign)}/{args.samples})")
                status = capture_sign(
                    cap, extractor, sign, vocab[sign]["pt"], args.samples, args.frames
                )
                if status == "quit":
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print_status(vocab, args.samples)


if __name__ == "__main__":
    main()
