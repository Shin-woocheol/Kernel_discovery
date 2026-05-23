# Kernel Discovery

This is the official code of **Automatic Kernel Discovery for High-Dimensional Bayesian Optimization**.

## Abstract

Gaussian Process (GP) kernels are central to Bayesian optimization (BO), yet designing effective kernels for high-dimensional problems still relies on extensive manual engineering. Existing automated approaches to kernel design are constrained on two fronts: kernel search methods restrict their search space to additive and multiplicative compositions of base kernels, while LLM-based BO approaches condition on raw observations, which becomes infeasible in high dimensions due to context-length constraints and poor pattern extraction from high-dimensional numerical streams. We introduce **Kernel Discovery**, a two-stage LLM-driven population-based evolutionary framework for high-dimensional BO that searches a broader kernel space beyond predefined composition rules without conditioning on raw observations. Motivated by the observation that directly prompting an LLM to generate kernel code yields syntactically varied but functionally identical kernels, we adopt a two-stage approach: an LLM first proposes novel mathematical forms, then a second LLM call converts each form into validated, executable code. We also propose a leave-one-out continuous ranked probability score (LOO-CRPS) as a held-out predictive scoring criterion that penalizes overconfident fits more directly than marginal log-likelihood. On five standard high-dimensional BO benchmarks, our method achieves an average rank of **1.2 out of 17**, consistently outperforming competitive baselines. We further analyze the discovered kernels to examine which kernel characteristics lead to superior performance in high-dimensional BO.

## Installation

```bash
bash setup.sh                  # creates conda env `bo` with all Python deps
WITH_MUJOCO=1 bash setup.sh    # also installs mujoco210 + mujoco-py (needs sudo for apt). Required for the Humanoid benchmark.
conda activate bo
```

Tested on Linux with CUDA 12.1 and Python 3.10. See `setup.sh` for the full dependency list (or `requirements.txt` for the pinned versions used in the paper).

## Data setup

Two benchmarks need external data files:

- **SVM_388** — place `CT_slice_X.npy` and `CT_slice_y.npy` (or their `.gz` counterparts) under `hdbo/data/svm/`. The source is the UCI CT-slice dataset (https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis).
- **Humanoid** — place the linear policy at `hdbo/benchsuite/utils/mujuco_policies/Humanoid-v1/lin_policy_plus.npz`. Note: the folder is named `Humanoid-v1` for legacy reasons; the gym environment we optimize over is `gym.make("Humanoid-v2")`.

`Mopta08`, `Rover`, and `Lasso-DNA` do not require additional data files.

## API key

Set your OpenAI API key in `.env` at the repository root:

```
OPENAI_API_KEY=sk-...
```

## Usage

To reproduce the "Ours" row of Table 1 across all 5 benchmarks × 4 seeds:

```bash
bash run_table1.sh
```

Each run writes to `hdbo/output/<bench>/<MMDD_HHMMSS>_seed_<S>_<model>_llm_crps_qlognei/`. The Table 1 value is the best entry of `y` in `<bench>_data.pt`:

```python
import torch
d = torch.load("<bench>_data.pt")
best = d["y"].min().item()   # SVM_388 / Mopta08 / Lasso-DNA are minimization
best = d["y"].max().item()   # Rover / Humanoid are maximization
```
