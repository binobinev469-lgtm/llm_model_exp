# Transformator

A small decoder-only transformer language model in PyTorch. It trains at character level on a text file and samples new text from the checkpoint.

## Setup

Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file pulls the CPU build of PyTorch, which is enough to train this model.

## Train

```bash
python train.py --steps 500
```

The default corpus is `data/sample.txt`. Point `--data` at any UTF-8 text file. The checkpoint lands in `checkpoints/model.pt`.

## Generate

```bash
python generate.py --prompt "The transformer "
```

`--temperature` and `--top-k` control how sharply the sampler picks the next character.

## Layout

- `transformator/model.py` — embeddings, causal attention, MLP, stacked blocks, generation
- `transformator/data.py` — character vocabulary and training batches
- `train.py` — AdamW loop
- `generate.py` — sampling from a checkpoint
