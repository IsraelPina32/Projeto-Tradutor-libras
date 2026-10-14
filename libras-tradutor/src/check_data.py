"""Confere a qualidade do dataset coletado.

Uso:  python -m src.check_data

Para cada sinal mostra quantas amostras existem, em quantos frames (em média)
as mãos e o rosto foram detectados, e avisa sobre amostras suspeitas.
"""
import json

import numpy as np

from src.config import FACE_SLICE, HANDS_SLICE, KEYPOINTS_DIR, META_PATH, NUM_FEATURES, VOCAB_PATH

MIN_HANDS = 0.7   # abaixo disso a amostra é considerada suspeita
MIN_FACE = 0.7


def main():
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else None
    expected_len = meta["sequence_length"] if meta else None
    print(f"formato salvo: {meta}\n" if meta else "meta.json não encontrado (nenhuma coleta ainda)\n")

    print(f"{'sinal':<12}{'amostras':>9}{'mãos':>8}{'rosto':>8}   avisos")
    problems = 0
    for sign in vocab:
        files = sorted((KEYPOINTS_DIR / sign).glob("*.npy"))
        if not files:
            print(f"{sign:<12}{0:>9}{'-':>8}{'-':>8}")
            continue
        hand_ratios, face_ratios, warns = [], [], []
        for f in files:
            a = np.load(f)
            if a.ndim != 2 or a.shape[1] != NUM_FEATURES or (expected_len and a.shape[0] != expected_len):
                warns.append(f"{f.name}: formato {a.shape}")
                continue
            hr = (a[:, HANDS_SLICE] != 0).any(axis=1).mean()
            fr = (a[:, FACE_SLICE] != 0).any(axis=1).mean()
            hand_ratios.append(hr)
            face_ratios.append(fr)
            if hr < MIN_HANDS:
                warns.append(f"{f.name}: mãos em {hr:.0%} dos frames")
            if fr < MIN_FACE:
                warns.append(f"{f.name}: rosto em {fr:.0%} dos frames")
        problems += len(warns)
        mh = f"{np.mean(hand_ratios):.0%}" if hand_ratios else "-"
        mf = f"{np.mean(face_ratios):.0%}" if face_ratios else "-"
        print(f"{sign:<12}{len(files):>9}{mh:>8}{mf:>8}   {len(warns)} suspeita(s)" if warns else
              f"{sign:<12}{len(files):>9}{mh:>8}{mf:>8}")
        for w in warns[:5]:
            print(f"    - {w}")
        if len(warns) > 5:
            print(f"    ... e mais {len(warns) - 5}")
    print("\nTudo certo." if problems == 0 else
          f"\n{problems} aviso(s). Apague as amostras suspeitas (Remove-Item data\\keypoints\\<sinal>\\<n>.npy) e regrave.")


if __name__ == "__main__":
    main()
