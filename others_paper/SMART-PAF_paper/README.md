# SMART-PAF: Accurate Low-Degree Polynomial Approximation of Non-Polynomial Operators for Fast Private Inference in Homomorphic Encryption

**Authors:** Jianming Tong\*, Jingtian Dang\*, Anupam Golder, Arijit Raychowdhury, Cong Hao, Tushar Krishna  
**Affiliations:** Georgia Institute of Technology, Carnegie Mellon University  
**Code:** https://github.com/EfficientFHE/SmartPAF

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Background](#3-background)
4. [Problem Statement](#4-problem-statement)
5. [SMART-PAF Techniques](#5-smart-paf-techniques)
6. [Evaluation](#6-evaluation)
7. [Training Processing Deep Dive](#7-training-processing-deep-dive)
8. [Related Work](#8-related-work)
9. [Conclusion](#9-conclusion)
10. [Appendix](#10-appendix)
11. [References](#11-references)

---

## 1. Abstract

As machine learning (ML) permeates fields like healthcare, facial recognition, and blockchain, the need to protect sensitive data intensifies. Fully Homomorphic Encryption (FHE) allows inference on encrypted data, preserving the privacy of both data and the ML model. However, it slows down non-secure inference by up to five magnitudes, with a root cause of replacing non-polynomial operators (ReLU and MaxPooling) with high-degree Polynomial Approximated Function (PAF).

**SMART-PAF** is a framework to replace non-polynomial operators with low-degree PAF and then recover accuracy through four techniques:

1. **Coefficient Tuning (CT)** – adjust PAF coefficients based on input distributions before training
2. **Progressive Approximation (PA)** – progressively replace one non-polynomial operator at a time followed by fine-tuning
3. **Alternate Training (AT)** – alternate training between PAFs and other linear operators in a decoupled manner
4. **Dynamic Scale (DS) / Static Scale (SS)** – dynamically scale PAF input values within (−1,1) during training, and fix the scale as the running max value in FHE deployment

**Key Results:** For ResNet-18 under ImageNet-1k, SMART-PAF achieves 1.42× ∼ 13.64× accuracy improvement and 6.79× ∼ 14.9× speedup over prior works. A 14-degree PAF achieves 7.81× speedup compared to the 27-degree PAF from minimax approximation, with the same 69.4% post-replacement accuracy.

---

## 2. Introduction

### 2.1 Motivation

ML has become pervasive in healthcare, facial recognition, and blockchain. FHE provides privacy-preserving inference but introduces 5 orders of magnitude latency overhead. The primary bottleneck is non-polynomial operators (ReLU, MaxPooling) which must be replaced by Polynomial Approximated Functions (PAFs).

**The Dilemma:** High-degree PAFs offer better accuracy but higher latency. Low-degree PAFs are faster but suffer severe accuracy degradation.

### 2.2 Key Challenge

Previous coefficient fine-tuning techniques fail to converge for PAFs with degrees higher than 5. PAFs with degrees lower than 5 still suffer severe accuracy degradation even for simple 7-layer CNN on CIFAR-10. The challenge is to enable convergence of PAF-approximated models with arbitrary degrees.

### 2.3 Contributions

This is the first framework to:
1. Replace **all** non-polynomial operators with 8∼14-degree PAFs
2. Adopt proposed training techniques to minimize accuracy degradation

| Approach | Low Comm Overhead | Low Accuracy Degradation | Low Latency Overhead |
|---|---|---|---|
| SafeNet, CryptoGCN | ✗ | ✗ | ✓ |
| CryptoNet, CryptoDL, LoLa, CHE | ✗ | ✗ | ✓ |
| F1, CraterLake, BTS | ✓ | ✓ | ✗ |
| HEAX, Delphi, Gazelle, Cheetah | ✗ | ✗ | ✓ |
| SHE | ✓ | ✓ | ✗ |
| **SMART-PAF** | ✓ | ✓ | ✓ |

![Table 1](figures/Table_1.png)

---

## 3. Background

### 3.1 Non-polynomial Operators in FHE-based ML Inference

FHE uses the **CKKS** scheme (Cheon et al., 2016) which only supports polynomial operators (addition, multiplication). All non-polynomial operators (ReLU, MaxPooling) must be:
- **Approximated:** Replaced by PAFs (lower overhead)
- **Hybrid Scheme:** Offloaded to other schemes like Garbled Circuits (higher communication overhead)

### 3.2 Polynomial Approximated Function (PAF)

Rather than approximating ReLU/MaxPooling directly (which causes severe errors), prior work approximates the **sign(x)** function and constructs ReLU/Max from it:

- **ReLU:** (x + sign(x) · x) / 2
- **Max:** ((x + y) + (x − y) · sign(x − y)) / 2

### 3.3 PAF Forms and Multiplication Depth

| Form | Degree | Multiplication Depth |
|---|---|---|
| α = 10 (Lee et al.) | 27 | 10 |
| f₁² ∘ g₁² | 14 | 8 |
| f₂ ∘ g₃ | 12 | 6 |
| f₂ ∘ g₂ | 12 | 6 |
| f₁ ∘ g₂ | 10 | 6 |
| α = 7 | 5 | 5 |

*α indicates precision parameter. fᵢ, gᵢ refer to PAF base polynomials.*

![Table 2](figures/Table_2.png)

![Figure 2](figures/Figure_2.png)

---

## 4. Problem Statement

### 4.1 Polynomial Approximation as Optimization

The core problem is a **regression problem**: find coefficients aᵢ = {a₀ⁱ, ..., aɴⁱ} for an N-degree PAF such that the cumulative error of replacing the non-polynomial function R(xᵢ) is minimal:

$$\\min_{a_0,...,a_{D-1}} f(a,x) = \\sum_{i=0}^{D-1} \\frac{1}{N} \\sum_{i=0}^{N-1} (R(x_i, a_0,...,a_i) - a_i \\cdot x_i)^2$$

This is **non-convex** because optimising all D vectors of coefficients under changing input distributions simultaneously causes training divergence.

### 4.2 Single-Layer Optimization (Convex)

When optimizing only a single layer (input distribution is statistically fixed):

$$\\min_{a_i} f'(a_i) = \\frac{1}{N} \\sum_{i=0}^{N-1} (R(x_i) - a_i \\cdot x_i)^2$$

SGD can guarantee finding the optimal solution. The error converges at rate:

$$f'(a_i^T) - f'(a_i^*) = O\\left(\\frac{||a_i^0 - a_i^*||^2}{\\sqrt{T}}\\right)$$

A good initialization aᵢ⁰ reduces total error and improves convergence speed.

### 4.3 Post-replacement Model Retraining

PAF replacement changes model structure → changed data distributions for all subsequent layers → original coefficients become suboptimal → retraining is necessary.

---

## 5. SMART-PAF Techniques

### 5.1 Overview

Prior art training methods diverge when training PAF with degree >5. SMART-PAF introduces 4 synergistic techniques.

![Figure 1](figures/Figure_1.png)

### 5.2 Coefficient Tuning (CT)

**Problem:** Using a uniform PAF for all non-polynomial layers neglects variations in input distributions across layers.

**Solution:** Before training, profile input distributions for each non-polynomial layer using the training dataset. Then fine-tune PAF coefficients to fit the high-probability region of each layer's distribution.

**Benefit:** Provides closer-to-optimal initialization → higher accuracy and lower training time (1.04× ∼ 2.38× improvement without fine-tuning).

![Figure 3](figures/Figure_3.png)

### 5.3 Progressive Approximation (PA)

**Problem:** Replacing all non-polynomial operators simultaneously makes the regression non-convex (varying both coefficients AND inputs for all layers), causing training divergence.

**Solution:** Replace non-polynomial layers one at a time, in inference order. After each replacement, fine-tune until accuracy converges. This keeps the optimization convex (only one layer's coefficients vary at a time).

**Intuition:** PA applies approximation error progressively, keeping it within the optimizable range of the training algorithm.

![Figure 4](figures/Figure_4.png)

### 5.4 Alternate Training (AT)

**Problem:** Fine-tuning PAF coefficients and other layer parameters (Conv, BatchNorm, FC) simultaneously as a whole leads to suboptimal results.

**Solution:** Decouple the training:
1. Freeze PAF coefficients, train linear layer parameters
2. Freeze linear layer parameters, train PAF coefficients
3. Alternate between these two

### 5.5 Dynamic Scaling (DS) / Static Scaling (SS)

**Problem:** PAF approximations have high error outside [−1,1]. Input values can span a wide range.

**Dynamic Scaling (DS - Training only):**
- Add auxiliary scaling layer before each PAF
- Per batch: scale = max absolute input value
- Scale all inputs to [−1,1] to maximize distinguishability

**Static Scaling (SS - FHE Deployment):**
- FHE cannot compute batch-max (no value-dependent comparisons)
- Fix scale as the running maximum under the training dataset
- Relies on training/validation data similarity

### 5.6 SMART-PAF Framework Scheduler

The sequence of applying CT, PA, AT, and DS/SS significantly affects final accuracy. SMART-PAF includes a scheduler:

1. CT is applied offline before all steps
2. Each step replaces a single non-polynomial layer with PAF
3. Within each step: Training Group → Check Overfitting → Apply AT → Verify convergence → Next step
4. Dropout for overfitting mitigation, SWA for faster convergence

![Figure 5](figures/Figure_5.png)

![Figure 6](figures/Figure_6.png)

---

## 6. Evaluation

### 6.1 Experimental Setup

| Component | Details |
|---|---|
| **Models** | VGG-19 (CIFAR-10), ResNet-18 (ImageNet-1k) |
| **PAF Forms** | 6 forms (see Tab. 2) with minimal multiplication depth |
| **FHE Library** | Microsoft SEAL using CKKS (Degree: 32768, modulus bitwidth: 881) |
| **CPU** | AMD Threadripper 2990WX |
| **Training** | E=20 epochs per training group |

### 6.2 Coefficient Tuning Evaluation

**Key findings:**
- CT improves post-replacement accuracy without fine-tuning by **1.05× ∼ 3.32×**
- More beneficial for lower-degree polynomials (less capacity to fit entire input range)
- Higher-degree polynomials show less improvement (already have lower overall error)
- Approximating both ReLU AND MaxPooling causes 10.9% ∼ 21% more accuracy drop than ReLU only

![Figure 7](figures/Figure_7.png)

![Figure 8](figures/Figure_8.png)

### 6.3 Progressive Approximation Evaluation

- PA (orange bar) significantly outperforms direct replacement + direct training (blue bar)
- Direct replacement + progressive training alone (green) still suffers severe accuracy degradation
- Combined progressive replacement + progressive training = **best results**

### 6.4 Ablation Study (Table 3)

**ResNet-18 / ImageNet-1k (Original: 69.3%):**

| Technique Setup | f₁²∘g₁² (14°) | α=7 | f₂∘g₃ | f₂∘g₂ | f₁∘g₂ |
|---|---|---|---|---|---|
| baseline + DS | 59.6% | 66.2% | 62% | 49% | 37% |
| baseline + SS (prior work) | 25.5% | 47.1% | 23% | 4.2% | 0% |
| baseline + CT + PA + AT + DS | **69.9%** | 68% | 65.7% | 64.1% | 57.8% |
| **SMART-PAF (CT+PA+AT+SS)** | **69.4%** | **67%** | **65.3%** | **57.3%** | **6.5%** |
| Improvement over prior work | +43.9% (2.72×) | +19.9% (1.42×) | +42.3% (2.84×) | +53.1% (13.64×) | +6.5% (∞) |

**VGG-19 / CIFAR-10 (Original: 93.95%):**

| Technique Setup | f₁²∘g₁² | α=7 | f₂∘g₃ | f₂∘g₂ | f₁∘g₂ |
|---|---|---|---|---|---|
| SMART-PAF (CT+PA+AT+SS) | 92.16% | 92.62% | 91.51% | 88.45% | 76.93% |
| Improvement over prior work | +1.1% (1.01×) | +11.27% (1.14×) | +14.93% (1.2×) | +30.34% (1.52×) | +33.09% (1.75×) |

![Table 3](figures/Table_3.png)

### 6.5 Comparison with State-of-the-Art

**vs. Lee et al. 2021 (27-degree PAF) - VGG-19 / CIFAR-10 (Original: 93.95%):**

| PAF | SMART-PAF Acc. | Lee Acc. | Improvement | Speedup |
|---|---|---|---|---|
| f₁∘g₂ | 72.24% | - | -19.97% | 14.90× |
| f₂∘g₂ | 86.36% | - | -3.85% | 13.75× |
| f₂∘g₃ | 91.05% | 90.21% | +0.84% | 11.71× |
| α=7 | 91.82% | 90.21% | +1.61% | 6.79× |
| f₁²∘g₁² | 92.39% | 90.21% | +2.08% | 7.81× |

**ResNet-18 / ImageNet-1k:** The 14-degree PAF (f₁²∘g₁²) achieves **69.4% accuracy with 7.81× speedup** vs. 27-degree PAF at 69.3%.

![Table 4](figures/Table_4.png)

![Figure 9](figures/Figure_9.png)

---

## 7. Training Processing Deep Dive

### 7.1 Training Curve Analysis (Figure 9)

Comparison of baseline vs. SMART-PAF training curves for ResNet-18 (ImageNet-1k) with 14-degree PAF:

- **Baseline (blue):** Starts at ~0.1% accuracy, fluctuates wildly, never converges — demonstrating the invalidity of direct training
- **SMART-PAF (orange):** Each ReLU replacement (purple diamond) causes a small accuracy dip, rapidly recovered via SWA and AT
- AT improves accuracy after specific replacements (ReLU 0, 8, 9) but not all (ReLU 3, 6) — AT helps but doesn't guarantee improvement

### 7.2 Task Complexity Impact

ImageNet-1k (224×224, 1000 classes) is significantly harder than CIFAR-10 (32×32, 10 classes):
- Same PAF (f₁²∘g₁²): only 0.55% drop on VGG-19/CIFAR-10 vs. 30.3% drop on ResNet-18/ImageNet-1k without CT
- **Task-tailored PAF selection is crucial** for preserving accuracy

![Table 5](figures/Table_5.png)

---

## 8. Related Work

### 8.1 Hybrid Schemes

HEAX, Delphi, Gazelle, Cheetah use hybrid schemes (FHE + MPC/Garbled Circuits):
- Non-FHE schemes handle non-polynomial kernels
- **Limitation:** Large communication overhead due to data transfer between parties

### 8.2 Approximation-only Schemes

CryptoNet, CryptoDL, LoLa replace all non-polynomial kernels with the same low-degree PAF:
- 27-degree PAF achieves low accuracy degradation but high latency
- Long multiplication chain dominates overall latency in FHE accelerators (F1, BTS, CraterLake)

### 8.3 Fine-grained Replacement

SafeNet, CryptoGCN propose per-layer polynomial replacement with ML training to find parameters:
- **Limitation:** Training divergence for polynomials with degree >5
- No systematic training techniques existed before SMART-PAF

### 8.4 AESPA

Recent work on accuracy-preserving low-degree polynomial activation for fast private inference.

---

## 9. Conclusion

SMART-PAF is the first framework to:
1. Replace **all** non-polynomial operators (ReLU + MaxPooling) with 8∼14-degree PAFs
2. Enable convergence through four synergistic techniques: CT, PA, AT, and DS/SS

**Quantitative results:**
- 1.42× ∼ 13.64× accuracy improvement over prior works on ResNet-18/ImageNet-1k
- 14-degree PAF: 7.81× speedup with same accuracy as 27-degree minimax PAF
- Pareto-frontier dominance in latency-accuracy tradeoff space

---

## 10. Appendix

### B.1 PAF Form: α = 7 (Minimax, 27-degree)

The original minimax polynomial (α = 7) is a composite of two 7-degree polynomials:

$$p_7(x) = p_{7,2}(x) \\circ p_{7,1}(x)$$

$$p_{7,1}(x) = \\sum_{i=0}^{7} a_i \\times x^i, \\quad p_{7,2}(x) = \\sum_{i=0}^{7} b_i \\times x^i$$

Since sign(x) is odd, even-degree coefficients are negligible and can be removed:

$$p_7(x) = p^{odd\\_only}_{7,2}(x) \\circ p^{odd\\_only}_{7,1}(x)$$

$$p^{odd\\_only}_{7,1}(x) = a_1 x + a_3 x^3 + a_5 x^5 + a_7 x^7$$

Coefficients for α = 7 (uniform across all layers):

| Coeff | Value |
|---|---|
| a₁ | 7.304451 |
| a₃ | -34.68258667 |
| a₅ | 59.85965347 |
| a₇ | -31.87552261 |
| b₁ | 2.400856 |
| b₃ | -2.631254435 |
| b₅ | 1.549126744 |
| b₇ | -0.331172943 |

### B.2 PAF Form: f₁² ∘ g₁² (14-degree)

Building-block polynomials:
$$f_1(x) = c_1 \\cdot x + c_3 \\cdot x^3$$
$$g_1(x) = d_1 \\cdot x + d_3 \\cdot x^3$$
$$f_2(x) = c_1 \\cdot x + c_3 \\cdot x^3 + c_5 \\cdot x^5$$
$$g_2(x) = d_1 \\cdot x + d_3 \\cdot x^3 + d_5 \\cdot x^5$$
$$g_3(x) = d_1 \\cdot x + d_3 \\cdot x^3 + d_5 \\cdot x^5 + d_7 \\cdot x^7$$

The composite PAF: f₁² ∘ g₁²(x) = g₁(g₁(f₁(f₁(x))))

**Multiplication depth:** 5 (see Appendix Figure 10)

See appendix tables for per-layer coefficients (17 layers for ResNet-18).

![Table 6](figures/Table_6.png)

![Figure 10](figures/Figure_10.png)

![Table 7](figures/Table_7.png)

![Table 8](figures/Table_8.png)

![Table 9](figures/Table_9.png)

![Table 10](figures/Table_10.png)

![Table 11](figures/Table_11.png)

---

## 11. References

Key references cited in the paper:

1. **Albrecht et al.** Homomorphic encryption standard. *Protecting Privacy through Homomorphic Encryption*, 2021.
2. **Boyd et al.** Convex Optimization. Cambridge University Press, 2004.
3. **Chen et al.** Simple Encrypted Arithmetic Library - SEAL v2.1. *FC*, 2017.
4. **Cheon et al.** Homomorphic Encryption for Arithmetic of Approximate Numbers. *Cryptology ePrint Archive*, 2016.
5. **Cheon et al.** Efficient Homomorphic Comparison Methods with Optimal Complexity. *ASIACRYPT*, 2020.
6. **Gilad-Bachrach et al.** CryptoNets: Applying Neural Networks to Encrypted Data. *ICML*, 2016.
7. **He et al.** Deep Residual Learning for Image Recognition. *CVPR*, 2016. (ResNet)
8. **Hesamifard et al.** CryptoDL: Deep Neural Networks over Encrypted Data. *arXiv*, 2017.
9. **Juvekar et al.** Gazelle: A Low Latency Framework for Secure Neural Network Inference. *USENIX Security*, 2018.
10. **Kim et al.** BTS: An Accelerator for Bootstrappable Fully Homomorphic Encryption. *ISCA*, 2022.
11. **Lou et al.** SAFENet: Secure inference with low-degree polynomials. *arXiv*, 2021.
12. **Mishra et al.** Delphi: A Cryptographic Inference System for Neural Networks. *PPMLP*, 2020.
13. **Park et al.** AESPA: Accuracy Preserving Low-degree Polynomial Activation for Fast Private Inference. 2022.
14. **Ran et al.** CryptoGCN: Fast and Scalable Homomorphically Encrypted GCN Inference. *NeurIPS*, 2022.
15. **Reagen et al.** Cheetah: Optimizing and Accelerating Homomorphic Encryption for Private Inference. *HPCA*, 2021.
16. **Riazi et al.** HEAX: An Architecture for Computing on Encrypted Data. *ASPLOS*, 2020.
17. **Samardzic et al.** F1: A Fast and Programmable Accelerator for FHE. *MICRO*, 2021.
18. **Samardzic et al.** CraterLake: A Hardware Accelerator for Efficient Unbounded Computation on Encrypted Data. *ISCA*, 2022.
19. **Simonyan & Zisserman.** Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*, 2015. (VGG)
20. **Xie et al.** CHE: Homomorphic Encryption for Secure CNN Inference. 2022.
