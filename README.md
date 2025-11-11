# GEC-BiCaps — GASD + CapsNet + Bidirectional Agreement for Arabic GEC

Low-resource Arabic Grammatical Error Correction (GEC) system built on a modified version of Transformer-based with: GASD noise method (two baselines: SDA word swapping, SEG spelling & normalization) to construct large synthetic parallel data, CapsNet (EM routing) for dynamic layer aggregation across encoder/decoder layers, Bidirectional agreement (R2L & L2R) with KL-divergence regularization to mitigate exposure bias, and Multi-pass correction and L2R re-ranking of R2L n-best candidates.

> This repository targets QALB-2014/2015 evaluation with the official M² (MaxMatch) scorer.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [1) GASD Synthetic Data Generation](#1-gasd-synthetic-data-generation)
  - [2) Training](#2-training)
  - [3) Decoding & Multi-pass Correction](#3-decoding--multi-pass-correction)
  - [4) Re-ranking](#4-re-ranking)
  - [5) Evaluation on QALB](#5-evaluation-on-qalb)
- [Configs](#configs)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)

---

### Requirements

- Python 3.8+
- CUDA-enabled PyTorch (optional but recommended)
- torch==1.13.1
- torchtext==0.14.1
- numpy
- pandas
- PyArabic
- tqdm
- pyyaml
- sentencepiece

### Project Structure
ArabicGEC/
│
├─ configs/
│ ├─ gasd.yml # GASD (SDA/SEG) settings
│ ├─ train.yml # model & training hyperparameters
│ └─ infer.yml # decoding, beam size, rerank flags
│
├─ data/
│ ├─ siiurce_text.txt # clean Arabic monolingual corpus (input to GASD)
│ ├─ vocab.txt # vocabulary for semantic/noise ops (optional)
│ ├─ train.csv # GASD output: parallel dataset (src,trg)
│ └─ qalb/
│ ├─ test.src # test source (noisy)
│ ├─ test.ref # test references (plain text; optional)
│ └─ test.gold.m2 # QALB-2014/2015 official M² annotations
│
├─ scripts/
│ ├─ gasd_generate.py # build synthetic parallel (CSV: src,trg)
│ ├─ train_caps_bid.py # train L2R + R2L with KL regularization
│ ├─ decode.py # decode with R2L (or L2R), write hypotheses
│ ├─ multipass.py # chain R2L→L2R or L2R→R2L
│ ├─ rerank.py # rerank R2L n-best with L2R
│ └─ make_vocab.py # build vocab.txt from mono.txt (optional)
│
├─ m2Scripts/
│ └─ m2scorer.py # official MaxMatch scorer (place here)
│
├─ checkpoints/ # saved models (L2R, R2L, joint)
├─ system_outputs/ # predictions, n-best, reranked results
├─ images-and-diagrams/ # optional figures for the README/paper
├─ requirements.txt
└─ README.md
