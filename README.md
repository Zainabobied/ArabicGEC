# GEC-BiCaps — GASD + CapsNet + Bidirectional Agreement for Arabic GEC

Low-resource Arabic Grammatical Error Correction (GEC) system built on a Transformer backbone with:GASD noise method (two baselines: SDA word swapping, SEG spelling & normalization) to construct large synthetic parallel data, CapsNet (EM routing) for dynamic layer aggregation across encoder/decoder layers, Bidirectional agreement (R2L & L2R) with KL-divergence regularization to mitigate exposure bias, and Multi-pass correction and L2R re-ranking of R2L n-best candidates.

> This repository targets QALB-2014/2015 evaluation with the official M² (MaxMatch) scorer.
