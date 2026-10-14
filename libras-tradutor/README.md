# Libras Tradutor (MVP)

Reconhece sinais de Libras pela câmera e monta uma mensagem em português,
inglês e espanhol.

## Instalação (Windows, PowerShell)

Use Python 3.10, 3.11 ou 3.12 (o MediaPipe ainda não cobre versões mais novas).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Linux/macOS: `python3 -m venv .venv` e `.venv/bin/pip install -r requirements.txt`.

## Passo 1 – coletar dados

Rode sempre a partir da pasta raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m src.capture                  # todos os sinais do vocab.json
.\.venv\Scripts\python.exe -m src.capture --sign obrigado  # só um sinal
.\.venv\Scripts\python.exe -m src.capture --samples 5      # teste rápido
.\.venv\Scripts\python.exe -m src.capture --frames 45      # amostras mais longas
.\.venv\Scripts\python.exe -m src.capture --status         # progresso da coleta
.\.venv\Scripts\python.exe -m src.check_data               # qualidade das amostras
```

Teclas: `ESPAÇO` grava uma amostra, `N` pula o sinal, `Q` sai.
Pode parar e continuar depois: as amostras já gravadas são mantidas.

### Formato do dataset (não mude no meio da coleta)

- Cada amostra é uma matriz `frames x 444` salva em `data/keypoints/<sinal>/<n>.npy`.
- Vetor de cada frame: pose (132) | mão esquerda (63) | mão direita (63) | rosto (186: lábios, sobrancelhas, nariz e queixo).
- O tamanho da sequência (`--frames`, padrão 30 ≈ 1 s) e o layout ficam registrados em
  `data/keypoints/meta.json`. O `capture.py` recusa gravar se o formato for diferente do já salvo.
- Para recomeçar com outro formato, apague as pastas de sinais e o `meta.json` em `data/keypoints`.

### Dicas para um bom dataset

- Boa iluminação de frente, fundo simples, tronco, rosto e mãos sempre visíveis.
- Faça o movimento completo do sinal dentro da janela de gravação.
- Varie distância, posição e velocidade a cada amostra.
- Quanto mais pessoas gravarem, melhor o modelo generaliza.
- Valide cada sinal do `vocab.json` com uma pessoa surda ou intérprete de Libras.
- Rode `src.check_data` de vez em quando e regrave as amostras suspeitas.

## Estrutura

```
vocab.json          sinais e traduções (pt/en/es)
src/config.py       constantes (tamanho da sequência, layout do vetor, caminhos)
src/extract.py      MediaPipe -> vetor de 444 números por frame
src/capture.py      coleta de amostras pela câmera
src/check_data.py   confere a qualidade do dataset
data/keypoints/     amostras salvas + meta.json
```

## Próximos passos

2. `train.py`: treino do modelo (LSTM/GRU) com os `.npy`
3. `predict.py`: reconhecimento em tempo real
4. Montagem da mensagem, tradução PT/EN/ES e voz
5. Interface (Streamlit)
6. Caminho de volta: fala para texto, texto na tela e sinais em vídeo
