# Histopathology OOD Classification

Solution to the **MVA — Deep Learning for Medical Imaging (DLMI) 2026 challenge**, by
Nicolas Beaujoin and Tanguy Lourme (team *tanguy nico*, Group 29).

**TL;DR:** Frozen embeddings from two complementary pathology foundation models
(UNI2-h ⊕ Virchow-2, 4096 dims) + a regularized logistic regression reach
**0.9889 validation** / **0.98891 public test** accuracy.

---

## 1. The problem: out-of-distribution tumor classification

The task is multi-center histopathology tumor classification, where the data comes
from **different hospital centers**:

| Split | Center |
|---|---|
| Train | 3 centers |
| Validation | a 4th (unseen) center |
| Test | a 5th (unseen) center |

Because staining protocols, scanners, and tissue preparation vary substantially
between hospitals, the images exhibit a strong **domain shift**: a classifier that
overfits to training-center artifacts (a particular scanner's color profile, a
particular lab's staining recipe) will generalize poorly. Standard classification
accuracy is therefore not enough on its own — the model must be **robust to
cross-center visual variation**. Both the validation and test sets come from
centers never seen during training, so they measure out-of-distribution (OOD)
generalization.

The provided baseline was a generic DINOv2-ViT-S/14 encoder (384-dim features)
followed by a linear probe, reaching ~0.84 validation accuracy.

## 2. Approach

The method decouples representation learning from classification:

1. **The encoder matters more than the head.** On frozen embeddings, the
   classifier accounts for very little of the performance ceiling. Switching the
   generic DINOv2 encoder for a histopathology foundation model (UNI2-h) was the
   single largest improvement: ~0.84 → ~0.98 accuracy.
2. **Pathology foundation models are already stain-robust.** They are
   self-supervised on 200M+ tiles from hundreds of thousands of slides with
   diverse staining protocols. Classical stain normalization (Macenko, Reinhard)
   actually *hurt* performance — it pushes images away from the distribution the
   encoders were trained on — so all stain normalization was discarded.
3. **Different foundation models capture complementary information.** UNI2-h and
   Virchow-2 are both ViT-H-scale models, but their training data and recipes
   differ, so their residual errors are partially uncorrelated. Concatenating
   their embeddings creates a joint space where domain-invariant patterns from
   one model compensate for center-specific noise in the other.

### Final architecture

1. **UNI2-h** (Mahmood Lab) — ViT-H/14, ~630M params, pretrained on ~200M tiles
   from 350k+ slides. CLS-token embedding: **1536 dims**.
2. **Virchow-2** (Paige) — ViT-H/14, ~632M params, pretrained on 3.1M whole-slide
   images from MSKCC. CLS token concatenated with the mean over the 256 patch
   tokens (as recommended by the official documentation): **2560 dims**.
3. **Feature concatenation** — the two embeddings are concatenated along the
   feature axis: **4096 dims** per tile.
4. **Classifier** — `StandardScaler` + `LogisticRegression(C=1e-3, max_iter=1000)`.

Both extractors stay **frozen** (no fine-tuning, not even LoRA): with encoders
this large and training data from only three centers, any parameter update risks
overfitting to the training centers and hurting OOD generalization — the exact
failure mode the challenge penalizes.

## 3. Implementation in `new_method_.py`

The pipeline in [`new_method_.py`](new_method_.py) is organized as follows:

```text
datasets/{train,val,test}.h5
        │
        ├── BaselineDataset ──► timm transform ──► UNI2-h / Virchow-2   (fp16 autocast)
        ├── CONCHDataset ─────► PIL conversion ───► CONCH               (fp16 autocast)
        │
        └── extract_and_save_*() ──► cached .pt files in
                features_uni2h/ · features_conch/ · features_virchow2/
                        │
                        ▼
        run(): concat UNI2-h ⊕ Virchow-2 (4096 dims)
                        │
                        ▼
        StandardScaler + LogisticRegression ──► submission_virchow2_uni2h_lr.csv
```

- **Datasets** — `BaselineDataset` reads the h5 files (`img` [C,H,W] tensor +
  `label`) and applies the timm preprocessing pipeline; `CONCHDataset` adds a
  tensor → PIL conversion because CONCH's preprocessing expects PIL images (a
  subtle compatibility issue that initially produced garbage features with
  ~0.54 validation accuracy).
- **Model loaders** — `load_uni2h()` (timm, `hf-hub:MahmoodLab/UNI2-h`),
  `load_virchow2()` (timm, `hf-hub:paige-ai/Virchow2`, SwiGLU), and
  `load_conch()` (via the `conch` package, `hf_hub:MahmoodLab/conch`).
- **Feature extraction and caching** — `extract_and_save_*()` run a single
  non-augmented fp16 (`torch.autocast`) forward pass per split and cache the
  embeddings to disk as `.pt` files, so all subsequent classifier experiments
  take seconds instead of hours. For Virchow-2, the CLS token is concatenated
  with the mean over patch tokens (`output[:, 0]` ⊕ `output[:, 5:].mean(1)` —
  the first 5 tokens are the CLS + 4 register tokens), giving the 2560-dim
  embedding.
- **Main entry points** — `get_uni2h_embeddings()`, `get_virchow2_embeddings()`
  and `get_conch_embeddings()` build the datasets/loaders and extract (or load
  cached) features; `run(predict=...)` loads the UNI2-h and Virchow-2 features,
  concatenates them, fits the logistic regression, reports validation accuracy
  and (with `predict=True`) writes `submission_virchow2_uni2h_lr.csv`.

Practical notes: embeddings for every extractor were cached (30–60 min per model
on a Colab T4 with fp16); no image augmentation (foundation models are already
invariant to flips/color jitter — augmented embeddings were nearly identical to
the originals); no stain normalization.

## 4. Results

Ablation of the main configurations (validation accuracy on the OOD val center;
test accuracy from the public Kaggle leaderboard):

| Configuration | Val acc. | Test acc. |
|---|---|---|
| DINOv2-ViT-S/14 + linear probe (provided baseline) | 0.84 | – |
| DINOv2-ViT-S/14 + linear probe + Macenko stain norm | 0.83 | – |
| UNI2-h + Linear (skorch SGD, 10 epochs) | 0.9859 | 0.9847 |
| UNI2-h + Logistic Regression (C=1e-3) | 0.9861 | 0.9851 |
| UNI2-h + LightGBM | 0.9824 | – |
| UNI2-h + KNN (cosine, k=11) | 0.9745 | – |
| UNI2-h + 2-layer MLP (BN, dropout 0.3) | 0.9854 | – |
| CONCH + Logistic Regression | 0.971 | – |
| Virchow-2 + Logistic Regression | 0.984 | – |
| UNI2-h ⊕ CONCH (2048 dims) + Logistic Regression | 0.9866 | – |
| **UNI2-h ⊕ Virchow-2 (4096 dims) + Logistic Regression** | **0.9889** | **0.98891** |

Once a strong pathology-pretrained encoder is used, the choice of classifier
changes validation accuracy by at most ±0.5%: the meaningful gains come from
(i) switching to a histopathology foundation model and (ii) concatenating two
complementary foundation models.

Things that did **not** work (and were removed): Macenko/Reinhard stain
normalization, KNN and LightGBM heads, Gaussian feature-level augmentation
(equivalent to weight decay), UNI2-h ⊕ CONCH (better than UNI2-h alone but
inferior to Virchow-2), MLP heads (on par with logistic regression), LoRA
fine-tuning of UNI2-h (discarded before submission).

The validation–test gap (0.9889 vs 0.98891) is essentially zero, which is
evidence that the validation procedure is well-calibrated and the representation
generalizes uniformly across centers.

## 5. Usage

```bash
pip install h5py timm huggingface_hub torch torchvision scikit-learn skorch pandas tqdm
pip install git+https://github.com/mahmoodlab/CONCH.git   # only for CONCH
```

1. Download the competition data (Kaggle:
   `mva-dlmi-2026-histopathology-ood-classification`) and place
   `train.h5`, `val.h5`, `test.h5` in `datasets/`.
2. Request access to the gated HuggingFace models (UNI2-h, Virchow-2, CONCH),
   then `huggingface_hub.login()`.
3. Run:

```bash
python new_method_.py            # trains and evaluates on the validation set
python new_method_.py            # with run(predict=True) to also write the test submission
```

The extraction entry points (`get_uni2h_embeddings()`, `get_virchow2_embeddings()`,
`get_conch_embeddings()`) can be called once each to build the cached feature
folders; after that, classifier experiments reload from cache in seconds.
`run.ipynb` reproduces the full setup (dependencies, HF login, data download,
execution) on a Colab GPU.

## 6. Papers and models

| Model | Paper | HuggingFace |
|---|---|---|
| **UNI2-h** (ViT-H/14, 1536-d, Mahmood Lab) | Chen et al., *Towards a general-purpose foundation model for computational pathology*, Nature Medicine 2024 — [arXiv:2308.15474](https://arxiv.org/abs/2308.15474) | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (gated) |
| **Virchow-2** (ViT-H/14, 2560-d, Paige) | Zimmermann et al., *Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology* — [arXiv:2408.00738](https://arxiv.org/abs/2408.00738) | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) (gated) |
| **CONCH** (ViT-B/16, 512-d, Mahmood Lab) | Lu et al., *CONCH: Challenges and Opportunities on the Path to Consolidated Vision-Language Models for Computational Pathology*, Nature Medicine 2024 — [arXiv:2306.07831](https://arxiv.org/abs/2306.07831) | [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) (gated), code: [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) |
| **DINOv2** (baseline encoder, ViT-S/14) | Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* — [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) |

Related work cited in the report:

- Vorontsov et al., *Virchow: A Million-Slide Digital Pathology Foundation Model* —
  [arXiv:2309.07778](https://arxiv.org/abs/2309.07778) (predecessor of Virchow-2).
- Kather et al., *Pan-cancer image-based detection of clinically actionable genetic
  alterations*, Nature Cancer 2021 — motivation for pathology foundation model
  ensembles on OOD tasks.
- Macenko et al., *A method for normalizing histology slides for quantitative
  analysis*, ISBI 2009 — the stain-normalization technique we tried and discarded.

The three model papers are also included in this repository as
[CONCH.pdf](CONCH.pdf), [UNI.pdf](UNI.pdf) and [VIRCHOW2.pdf](VIRCHOW2.pdf),
and the full challenge report is
[DLMI_Challenge_Report_Lourme_Beaujoin.pdf](DLMI_Challenge_Report_Lourme_Beaujoin.pdf).
