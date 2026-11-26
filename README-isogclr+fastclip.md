# CSCE 633 Project 1: iSogCLR vs FastCLIP-v3 with AdamW, Lion and LAMB --Ananya D

My part of the group project "Improving CLIP Training" (CSCE 633, Fall 2026). I compare two global contrastive losses, **iSogCLR** and **FastCLIP-v3**, each trained with three optimizers: **AdamW**, **Lion** and **LAMB**. I also tune the learning rate, weight decay and the temperature regularizer ρ.

**Setup (fixed by the project):**
- **Model:** CLIP with a ResNet-50 image encoder (ImageNet-pretrained, timm) and a DistilBERT text encoder, 256-d embeddings
- **Data:** trained on a 100k-pair subset of CC3M
- **Training:** batch size 128 for 30 epochs
- **Validation:** MSCOCO 5k image-text retrieval (Recall@1) and ImageNet-1k zero-shot classification (all 50k validation images)
- **Score** (the project's ranking metric) = (I2T R@1 + T2I R@1 + ImageNet top-1) / 3

All numbers below are from the full validation sets, using each run's best checkpoint. Training ran on single NVIDIA A100 GPUs on TAMU HPRC Grace. The default iSogCLR + AdamW run trained epochs 0–25 on a Colab T4 and resumed on Grace for epochs 26–29.

## Methods

| | iSogCLR | FastCLIP-v3 |
|---|---|---|
| Loss | robust global contrastive loss (RGCL) | RGCL-g, a global-temperature variant |
| Temperature | one learnable τ per image and per caption, updated by momentum SGD | one learnable global τ, updated by Adam |
| Inner moving average γ | constant 0.8 | cosine schedule from 1 to 0.2 |

Both losses estimate the full-dataset contrastive denominator with per-sample moving averages (FCCO), so the result doesn't depend on batch size the way the standard mini-batch CLIP loss does. ρ is the regularization weight in the objective τ·log g + ρ·τ. A larger ρ pushes the learned temperature lower.

## Results

### Best setting per method

| Method | Optimizer | Setting | I2T R@1 | T2I R@1 | IN top-1 | **Score** |
|---|---|---|---|---|---|---|
| FastCLIP-v3 | AdamW | lr 1e-4, ρ 5.5 | 13.54 | 10.94 | 27.10 | **17.19** |
| iSogCLR | AdamW | lr 1e-4, weight decay 0.1 | 14.42 | 10.15 | 26.53 | **17.03** |
| iSogCLR | AdamW | lr 5e-5 | 14.72 | 10.51 | 25.76 | 17.00 |

With default settings (lr 2e-4), FastCLIP-v3 led iSogCLR by 1.1 points (16.67 vs 15.56). After tuning the gap is **0.16 points**, which is within the run-to-run noise of a single seed. The tuning helped iSogCLR more (+1.47) than FastCLIP-v3 (+0.52).

### Optimizer comparison (best learning rate tried for each)

![Best score per optimizer](figures/optimizer_comparison.png)

| Optimizer | Learning rates tried | iSogCLR | FastCLIP-v3 |
|---|---|---|---|
| AdamW | 5e-5, 1e-4, **2e-4** (default), 4e-4 | 17.03 | 17.19 |
| Lion | **2e-5**, 4e-5, 1e-4 | 15.45 | 16.27 |
| LAMB | **1e-3**, 2e-3, 4e-3 | 11.60 | 11.95 |

The default learning rates are in bold. FastCLIP-v3 was not run at AdamW 5e-5, Lion 1e-4 or LAMB 4e-3. Lion used weight decay 0.1, and AdamW and LAMB used 0.02. The optimizers ranked the same way for both losses: **AdamW > Lion > LAMB**.
- **Lion** improved as its learning rate went down: 2e-5 beat 4e-5, and 1e-4 collapsed.
- **LAMB** got worse as its learning rate went up: 1e-3 scored 11.6, 2e-3 scored 8.6, and 4e-3 collapsed.
- Neither caught up with AdamW in the range tried.

### Learning-rate and regularization sweeps (AdamW)

| Run | iSogCLR | FastCLIP-v3 |
|---|---|---|
| lr 5e-5 | 17.00 | — |
| lr 1e-4 | 16.80 | 16.78 |
| lr 2e-4 (default) | 15.56 | 16.67 |
| lr 4e-4 | 13.60 | 14.12 |
| weight decay 0.1 (lr 2e-4) | 15.81 | — |
| lr 1e-4 + weight decay 0.1 | **17.03** | — |
| lr 1e-4 + ρ 5.0 (default 6.0) | 16.29 | — |
| lr 1e-4 + ρ 5.5 (default 6.5) | — | **17.19** |

### ρ controls FastCLIP-v3's learned temperature

![FastCLIP-v3 rho sweep](figures/fastclip_v3_rho_sweep.png)

The learned global temperature drops steadily as ρ increases, from 0.074 at ρ 4.5 to 0.013 at ρ 7.5. The score peaks at **ρ 5.5, where τ ≈ 0.05**. The default ρ 6.5 drives τ down to 0.028, which is slightly too low for this dataset.

### Per-run tables and training curves

- **Tables:** `results/iSogCLR/summary.md` and `results/FastCLIP/summary.md`, also as `.csv` and `.tex`.
- **Curves:** `results/*/curve_*.png` (validation score, both recalls, ImageNet top-1, training loss).
- **Temperature histograms:** `results/iSogCLR/tau_hist_*.png`, the per-sample temperature distributions.
- **Per-run files:** each folder `results/<method>/<run>/` contains the config (`args.json`), per-epoch metrics (`epochs.jsonl`), per-step logs (`steps.csv`), the full evaluation (`final_eval.json`) and temperature snapshots (`tau_epoch*.npz`).

## Findings

1. **Learning rate mattered more than the choice of loss.** Halving AdamW's learning rate from 2e-4 to 1e-4 gained +1.2 points for iSogCLR. That's more than the default-settings gap between the two losses.
2. **iSogCLR tends to overfit the 100k pairs.** At lr 1e-4 and 2e-4 its best epoch is 15, after which the validation score declines. A lower learning rate (best at epoch 21) or stronger weight decay (best at epoch 17) delays this and gives the best iSogCLR results. FastCLIP-v3 usually peaks at epochs 25–29.
3. **The two losses end up essentially tied once tuned:** 17.19 vs 17.03. FastCLIP-v3 is more robust to the default learning rate, and it trains slightly better with Lion.
4. **A too-high learning rate with Lion or LAMB causes a collapse.** It happened with iSogCLR at Lion 1e-4 and LAMB 4e-3. In both runs every per-sample temperature dropped to its lower bound (0.005), and the embeddings stopped carrying information. LAMB 4e-3 reached only 0.6% ImageNet top-1 (chance is 0.1%). The Lion 1e-4 run peaked after epoch 1 (5.55) and was stopped at epoch 15 when its loss turned into NaN.
5. **Training cost** is about 115 s per epoch on an A100 for AdamW, about 123 s for Lion and about 137 s for LAMB, with peak memory around 14 GB.

## Limitations

- **One seed per setting.** Differences of about 0.5 points or less, including the gap between the two best results, aren't conclusive.
- **Hyperparameters were chosen on the validation sets.** The test sets were not used.
- **Lion and LAMB got 3 learning rates each,** with weight decay kept at its default. AdamW was tuned more thoroughly.

## Reproducing

The code supports iSogCLR, FastCLIP-v1/v2/v3, SogCLR and CLIP, and the optimizers AdamW, LAMB, Lion, SGD, RAdam, Muon and (stochastic) L-BFGS.
- **Code and options:** see [`CODE_README.md`](CODE_README.md).
- **TAMU HPRC Grace (Slurm):** see [`hprc/README_HPRC.md`](hprc/README_HPRC.md). One run looks like:
  ```bash
  python train.py --method fastclip_v3 --optimizer adamw --lr 1e-4 --rho 5.5 \
      --data_root ./datasets --resume --save_every 2 --eval_every 2 --zs_subset 10000
  python eval_checkpoint.py --all --results_root results --data_root ./datasets
  python summarize.py
  ```
- **Google Colab:** `colab_run.ipynb`.

Model checkpoints (`checkpoint_best.pth`, about 360 MB each) are not stored in this repository, because GitHub rejects files over 100 MB. *[Add a link here if you attach them to a GitHub Release or share them on Google Drive.]*

## References

- Z.-H. Qiu et al., "Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization" (iSogCLR), ICML 2023. Code: https://github.com/zhqiu/contrastive-learning-iSogCLR
- X. Wei et al., "FastCLIP: A Suite of Optimization Techniques to Accelerate CLIP Training with Limited Resources," arXiv:2407.01445. Code: https://github.com/Optimization-AI/fast_clip
- Z. Yuan et al., "Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance" (SogCLR), ICML 2022.
- X. Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion), NeurIPS 2023.
- Y. You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes" (LAMB), ICLR 2020.

*[Acknowledgment: Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing.]
