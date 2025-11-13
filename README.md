# ArabicGEC an GEC model with Bidirectional Agreement and CapsNet for low resource languages
ArabicGEC is an Arabic Grammatical Error Correction (GEC) framework that extends the Transformer architecture with several complementary techniques. We proposed GASD, a noise-based data augmentation method that combines two baselines: SDA, which performs word-order perturbation, and SEG, which introduces character-level spelling and normalization errors to construct large-scale synthetic parallel corpora. A CapsNet module with Expectation–Maximization (EM) routing is integrated into the encoder–decoder structure to enable dynamic layer aggregation and capture hierarchical linguistic dependencies. The system further incorporates bidirectional agreement between R2L and L2R decoders using Kullback–Leibler divergence regularization to mitigate exposure bias and improve consistency across decoding directions. Finally, it performs multi-pass correction and re-ranking, where R2L outputs are refined through L2R decoding and re-ranked to select the most grammatically accurate candidates. 

> This repository targets QALB-2014/2015 evaluation with the official M² (MaxMatch) scorer.


<p align="center">
  <img src="/images%20and%20diagrams/fig1.png" alt="GASD Architecture" width="70%">
  <br>
  <i>Architecture Overview of the Proposed Noise Method. The Red Box Indicates SDA and the Yellow Represents SEG</i>
</p>

<p align="center">
  <img src="/images%20and%20diagrams/ArabicGEC.jpg" alt="ArabicGEC Architecture" width="35%">
  <br>
  <i>Illustration of the Model Architecture Integrating GASD. (a)Highlights the Bidirectional GEC-R2L and GEC-L2R Models Over Two Iterations Governed by a Regularization Term. (b) Provides a Detailed View of the Encoder and Decoder Architecture with CapsNet</i>
</p>

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

```

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

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



> Install exact CUDA/PyTorch versions matching your system.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt



