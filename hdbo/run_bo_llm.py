#!/usr/bin/env python3
import os
import sys

# Set memory allocator configuration to prevent fragmentation before torch is imported.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

"""
Run Bayesian optimization with formula-level LLM-driven kernel discovery.

Usage:
  python -m hdbo.run_bo_llm --benchmark SVM_388 --n_iter 50
"""
import argparse
import json
import math
import os
import shlex
import sys
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from hdbo.benchsuite import HDBO_BENCHMARKS
from hdbo.bo import BOTORCH_AVAILABLE, _clear_gpu_memory, _optimize_acquisition_with_fallback
from hdbo.kernel_evolver import evolve_kernel_for_bo
from hdbo.kernel_formula import fit_and_score_kernel


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-driven formula-level kernel discovery for high-dim BO")
    choices = list(HDBO_BENCHMARKS.keys())
    default_bench = choices[0] if choices else "SVM_388"
    parser.add_argument("--benchmark", nargs="+", default=[default_bench], choices=choices)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--acquisition", choices=["ei", "logei", "qlognei", "ts", "ucb"], default="qlognei")
    parser.add_argument("--kernel_gens", type=int, default=1)
    parser.add_argument("--kernel_mutations", type=int, default=1)
    parser.add_argument("--kernel_compositions", type=int, default=1)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--model_selection",
                        choices=["loss", "bic", "bic_correct", "crps_bic", "nlpd", "crps", "rank_based",
                                 "loss_prior", "nlpd_prior", "crps_prior", "rank_based_prior",
                                 "oracle", "softmax_crps", "crps_mle"],
                        default="crps",
                        help="'crps_bic' = 2*n*LOOCV_CRPS + k*log(n) like bic_correct; "
                             "'oracle' = per-kernel acqf+oracle then pick best batch; "
                             "'softmax_crps' = softmax sample proportional to -CRPS/T; "
                             "'crps_mle' = within-population z-score: crps_z + 0.3*nll_z")
    parser.add_argument("--softmax_temperature", type=float, default=1.0,
                        help="Temperature for softmax_crps sampling (higher = more uniform)")
    parser.add_argument("--simplicity_prompt", action="store_true",
                        help="Inject simplicity guidance into mutation/composition prompts")
    parser.add_argument("--prompt_highdim", type=str, choices=["on", "off"], default="on",
                        help="Include HIGH-DIMENSIONAL STRATEGIES block in mutation prompt (default: on)")
    parser.add_argument("--prompt_psd", type=str, choices=["on", "off"], default="on",
                        help="Include POSITIVE SEMI-DEFINITENESS block in mutation+composition prompts (default: on)")
    parser.add_argument("--base_kernel_preset", type=str, default="default",
                        choices=["default", "standard"],
                        help="Base kernel set: 'default'=(dsp,matern52,spherical,cylindrical,rq), "
                             "'standard'=(dsp,matern32,matern52,rq,linear,periodic)")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_fname", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hdbo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--base_kernel", type=str, default="llm")
    parser.add_argument("--max_eval_time", type=float, default=60.0)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--oracle_n_jobs", type=int, default=8)
    parser.add_argument("--llm_interval", type=int, default=1)
    parser.add_argument("--prune_baseline", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--verbose_log", action="store_true")
    parser.add_argument("--llm_backend", type=str, default="openai", choices=["openai", "vllm"])
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--llm_seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.3,
                        help="Temperature for code generation LLM calls (default: 0.3)")
    parser.add_argument("--max_retry", type=int, default=0)
    parser.add_argument("--score_subsample", type=float, default=1.0,
                        help="Fraction of data for LOOCV scoring (0-1, default: 1.0 = full data)")
    parser.add_argument("--num_eval", type=int, default=100,
                        help="LOOCV / rank score uses at most this many highest-y training points; "
                             "<= 0 means use all (after score_subsample). Default: 100")
    parser.add_argument("--fit_backend", choices=["scipy", "torch"], default="scipy",
                        help="GP fit optimizer: scipy L-BFGS-B (default) or torch Adam (ablation).")
    parser.add_argument("--acqf_backend", choices=["scipy", "torch"], default="scipy",
                        help="Acqf optimizer: scipy L-BFGS-B (default) or torch Adam (ablation).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to an existing bench run dir containing <bench>_data.pt. "
                             "If set, resume BO from that checkpoint (skips Sobol init, reuses same dir). "
                             "Ignores --data_fname.")
    return parser.parse_args()


def _run_bo_llm_single(
    benchmark_name: str,
    dim: Optional[int],
    n_init: int,
    n_iter: int,
    batch_size: int,
    acquisition: str,
    kernel_evolve_gens: int,
    kernel_evolve_mutations: int,
    kernel_evolve_compositions: int,
    population_size: int,
    client,
    seed: int,
    verbose: bool,
    save_dir: Optional[str],
    model_selection: str = "loss",
    max_eval_time: float = 120.0,
    max_workers: int = 5,
    llm_interval: int = 1,
    prune_baseline: bool = False,
    sequential: bool = False,
    on_eval=None,
    verbose_log: bool = False,
    temperature: float = 1.0,
    codegen_temperature: float = 0.3,
    max_retry: int = 0,
    simplicity_prompt: bool = False,
    score_subsample: float = 1.0,
    num_eval: int = 100,
    softmax_temperature: float = 1.0,
    resume_data: Optional[Dict[str, Any]] = None,
    method_tag: str = "",
    prompt_highdim: bool = True,
    prompt_psd: bool = True,
    base_kernel_preset: str = "default",
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Run BO with formula-level LLM kernel discovery."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    info = HDBO_BENCHMARKS[benchmark_name]
    f = info["f"]
    if hasattr(f, 'set_seed'):
        f.set_seed(seed)
    bounds = info["bounds"]
    dim = dim if dim is not None else info["dim"]
    if dim != info["dim"]:
        bounds_per_dim = info.get("bounds_per_dim")
        if bounds_per_dim is None:
            raise ValueError(f"Benchmark {benchmark_name} has fixed dimension {info['dim']}; --dim override not supported")
        bounds = [bounds_per_dim] * dim

    optimal_val = info.get("min_val")
    maximize = info.get("maximize", True)

    def best_val(y_slice: torch.Tensor) -> float:
        return y_slice.max().item() if maximize else y_slice.min().item()

    resume = resume_data is not None
    if resume:
        # --- Resume from checkpoint ---
        X = resume_data["X"].to(device=device, dtype=dtype)
        y = resume_data["y"].to(device=device, dtype=dtype)
        history: List[Dict[str, Any]] = list(resume_data.get("history", []))
        n_init = min(n_init, n_iter)
        if verbose:
            print(f"  Resumed from checkpoint: n_evals={X.shape[0]}, best={best_val(y):.4f}")
    else:
        # --- Initial Sobol points ---
        n_init = min(n_init, n_iter)
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
        X = sobol.draw(n_init).to(device=device, dtype=dtype)
        for d in range(dim):
            lo, hi = bounds[d]
            X[:, d] = X[:, d] * (hi - lo) + lo
        y = f(X).reshape(-1, 1).to(device=device, dtype=dtype)

        history: List[Dict[str, Any]] = []
        for i in range(n_init):
            n_evals = i + 1
            best_so_far = best_val(y[:n_evals])
            history.append({"n_evals": n_evals, "best": best_so_far, "incumbent": best_so_far})
            if on_eval:
                regret = (best_so_far - optimal_val) if optimal_val is not None else None
                on_eval(n_evals, best_so_far, regret)
        if verbose:
            print(f"  Init: n_evals={n_init}, best={best_val(y):.4f}")

    history_path = os.path.join(save_dir, "history.txt") if save_dir else None
    if history_path and not resume:
        with open(history_path, "w", encoding="utf-8") as hf:
            hf.write("# bo_iter oracle_queries_batch oracle_evals_cumulative best_so_far elapsed_sec acqf_random_fallback best_kernel\n")

    # Checkpoint helper: saves X/y/history after every oracle batch so runs are resumable.
    checkpoint_path = os.path.join(save_dir, f"{benchmark_name}_data.pt") if save_dir else None

    def _save_checkpoint():
        if not checkpoint_path:
            return
        torch.save({
            "X": X.cpu() if X.is_cuda else X,
            "y": y.cpu() if y.is_cuda else y,
            "benchmark": benchmark_name, "seed": seed,
            "method": method_tag, "acquisition": acquisition,
            "history": history,
        }, checkpoint_path)

    if not resume:
        _save_checkpoint()

    acqf_fail_count = 0
    acqf_total_count = 0
    start_time = time.time()

    # --- Verbose logging setup ---
    _client_generate_orig = None
    logger = None
    if verbose_log and save_dir:
        from hdbo.bo_logger import BoLogger
        from hdbo.llm_client import set_truncation_log
        log_dir = os.path.join(save_dir, "logs")
        logger = BoLogger(log_dir)
        set_truncation_log(os.path.join(log_dir, "llm_truncations.log"))

        llm_query_jsonl = os.path.join(log_dir, "llm_queries.jsonl")
        llm_query_txt = os.path.join(log_dir, "llm_query_history.txt")
        _llm_call_counter = [0]

        if client is not None:
            _client_generate_orig = client.generate
            _orig_generate = _client_generate_orig

            def _logged_generate(prompt, **kwargs):
                idx = _llm_call_counter[0]
                _llm_call_counter[0] += 1
                t0 = time.time()
                response = _orig_generate(prompt, **kwargs)
                elapsed = time.time() - t0
                rec = {
                    "call_index": idx,
                    "model": getattr(client, "model_name", "unknown"),
                    "prompt_len": len(prompt),
                    "response_len": len(response) if response else 0,
                    "latency_sec": round(elapsed, 4),
                    "_timestamp": time.time(),
                }
                try:
                    with open(llm_query_jsonl, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                except OSError:
                    pass
                sep = "=" * 80
                try:
                    with open(llm_query_txt, "a", encoding="utf-8") as fh:
                        fh.write(f"{sep}\n[LLM call #{idx}] latency={elapsed:.2f}s\n{sep}\n")
                        fh.write(f"--- PROMPT ---\n{prompt[:100000]}\n")
                        fh.write(f"--- RESPONSE ---\n{(response or '')[:100000]}\n{sep}\n\n")
                except OSError:
                    pass
                return response

            client.generate = _logged_generate

    n_remaining = n_iter - X.shape[0]
    num_rounds = max(0, (n_remaining + batch_size - 1) // batch_size) if n_remaining > 0 else 0

    # For oracle / softmax_crps / crps_mle modes, use CRPS as internal scoring
    # (the outer kernel-pick strategy differs from inner fitting metric)
    special_selection = model_selection in ("oracle", "softmax_crps")
    combined_selection = model_selection == "crps_mle"
    inner_model_selection = "crps" if (special_selection or combined_selection) else model_selection
    CRPS_MLE_LAMBDA_Z = 0.3  # z-score weight for NLL vs CRPS — tuned on fork data

    # cumulative kernels carry formula field
    cumulative_kernels: List[Dict[str, Any]] = []
    # Base kernels need this many consecutive bad rounds (no global improvement or acqf failure) before removal
    BASE_STRIKE_LIMIT = 3
    base_no_improve_strikes: Dict[str, int] = {}
    # Bases removed by discard must stay out: evolve_kernel_for_bo always re-injects template bases
    excluded_base_sources: set[str] = set()

    for it in range(num_rounds):
        bo_round = it + 1
        remaining = n_iter - X.shape[0]
        if remaining <= 0:
            break
        q = min(batch_size, remaining)

        _clear_gpu_memory()

        if verbose:
            print(f"\n[BO round {bo_round}/{num_rounds}] "
                  f"Kernel discovery (cumulative pool: {len(cumulative_kernels)} kernels)...")

        y_for_model = y if maximize else -y
        iter_save_dir = os.path.join(save_dir, "codebase") if save_dir else None

        skip_llm = (it % llm_interval != 0)
        curr_gens = 0 if skip_llm else kernel_evolve_gens
        curr_comp = 0 if skip_llm else kernel_evolve_compositions

        if skip_llm and verbose:
            print(f"  Skipping LLM evolution (interval={llm_interval}). Re-fitting best kernels...")

        evol_stats = {} if logger else None
        _t_iter_start = time.time()
        _t_evol_start = time.time()

        # --- Formula-level evolution ---
        code, gp_model, _, population = evolve_kernel_for_bo(
            X,
            y_for_model.squeeze(-1),
            client=client,
            n_generations=curr_gens,
            n_mutations=kernel_evolve_mutations,
            n_parents=population_size,
            population_size=population_size,
            kernel_compositions=curr_comp,
            seed=seed + it,
            verbose=verbose,
            bo_iter=bo_round,
            save_dir=iter_save_dir,
            seed_population=cumulative_kernels,
            model_selection=inner_model_selection,
            max_eval_time=max_eval_time,
            max_workers=max_workers,
            bounds=bounds,
            stats_out=evol_stats,
            logger=logger,
            temperature=temperature,
            codegen_temperature=codegen_temperature,
            max_retry=max_retry,
            simplicity_prompt=simplicity_prompt,
            score_subsample=score_subsample,
            num_eval=num_eval,
            excluded_base_sources=excluded_base_sources,
            prompt_highdim=prompt_highdim,
            prompt_psd=prompt_psd,
            base_kernel_preset=base_kernel_preset,
        )

        _t_evol_sec = time.time() - _t_evol_start

        if population:
            population = sorted(population, key=lambda p: p.get("loss", float("inf")))

        # --- Combined selection: crps_mle (z-score CRPS + λ·z-score NLL) ---
        if combined_selection and population:
            # preserve raw CRPS for logging before we overwrite "loss"
            for p in population:
                if "crps" not in p:
                    p["crps"] = p.get("loss", float("inf"))
            crps_arr = np.array([p.get("crps", float("inf")) for p in population], dtype=float)
            nll_arr  = np.array([p.get("nll",  float("inf")) for p in population], dtype=float)
            finite = np.isfinite(crps_arr) & np.isfinite(nll_arr)
            if finite.sum() >= 2:
                c_mean = crps_arr[finite].mean()
                c_std  = max(float(crps_arr[finite].std()), 1e-9)
                n_mean = nll_arr[finite].mean()
                n_std  = max(float(nll_arr[finite].std()),  1e-9)
                for p in population:
                    c = p.get("crps", float("inf")); n = p.get("nll", float("inf"))
                    if math.isfinite(c) and math.isfinite(n):
                        p["loss"] = (c - c_mean) / c_std + CRPS_MLE_LAMBDA_Z * (n - n_mean) / n_std
                    else:
                        p["loss"] = float("inf")
                population = sorted(population, key=lambda p: p.get("loss", float("inf")))

        # gp_model / `code` come from evolver's refit best; crps_mle re-sort can put another
        # kernel at population[0]. Align rank-1 with the surrogate actually used for acqf + discard.
        if code is not None and population:
            cs = code.strip()
            ix = next((i for i, p in enumerate(population) if p["code"].strip() == cs), None)
            if ix is not None and ix != 0:
                sel = population.pop(ix)
                population.insert(0, sel)

        # --- Special selection: oracle or softmax_crps ---
        # Build per-kernel (fit -> acqf -> (oracle if oracle mode)) records.
        # For oracle mode: pick kernel whose batch produced the best oracle y.
        # For softmax_crps mode: sample kernel with prob ∝ softmax(-crps/T), then oracle its batch.
        special_x_next = None
        special_y_next = None
        special_selected_idx = None
        if special_selection and population and BOTORCH_AVAILABLE:
            if verbose:
                print(f"  [{model_selection}] per-kernel acqf "
                      f"{'+ oracle query' if model_selection == 'oracle' else '(oracle only on selected)'}"
                      f" for {len(population)} kernels in pool")
            y_fm = y_for_model if y_for_model.dim() > 1 else y_for_model.unsqueeze(-1)
            f_best_special = y_fm.max().item()
            for p_idx, p in enumerate(population):
                if p.get("loss", float("inf")) == float("inf"):
                    p["batch_y_best"] = None
                    p["_x_batch"] = None
                    continue
                try:
                    score_k, model_k, _, fr_k, _, _, _ = fit_and_score_kernel(
                        p["code"], X, y_for_model.squeeze(-1) if y_for_model.dim() > 1 else y_for_model,
                        model_selection="crps", state_dict=p.get("state_dict"),
                        timeout=max_eval_time, bounds=bounds,
                        score_subsample=score_subsample, num_eval=num_eval,
                    )
                except Exception:
                    p["batch_y_best"] = None
                    p["_x_batch"] = None
                    continue
                if model_k is None or fr_k is not None:
                    p["batch_y_best"] = None
                    p["_x_batch"] = None
                    continue
                # Update the stored CRPS score in case re-fit changed it
                if score_k is not None and not np.isinf(score_k):
                    p["crps"] = float(score_k)
                else:
                    p["crps"] = float(p.get("loss", float("inf")))
                try:
                    md = next(model_k.parameters()).device
                    dt = next(model_k.parameters()).dtype
                    Xb = X if acquisition == "qlognei" else None
                    x_batch, fa_k = _optimize_acquisition_with_fallback(
                        model_k, bounds, acquisition, f_best_special, md, dt,
                        batch_size=q, X_baseline=Xb, verbose=False,
                        prune_baseline=prune_baseline, seed=seed, sequential=sequential,
                        bo_iter=bo_round, n_evals=X.shape[0] + p_idx,
                        best_kernel_source=p.get("source"),
                    )
                    if x_batch.dim() == 1:
                        x_batch = x_batch.unsqueeze(0)
                except Exception:
                    p["batch_y_best"] = None
                    p["_x_batch"] = None
                    del model_k
                    _clear_gpu_memory()
                    continue
                p["_x_batch"] = x_batch.detach().cpu()
                p["_acqf_failed"] = bool(fa_k)
                if model_selection == "oracle":
                    try:
                        y_batch = f(x_batch).reshape(-1, 1).to(device=device, dtype=dtype)
                        p["batch_y_best"] = float(best_val(y_batch))
                        p["_y_batch"] = y_batch.detach().cpu()
                    except Exception:
                        p["batch_y_best"] = None
                        p["_y_batch"] = None
                else:
                    p["batch_y_best"] = None  # softmax mode fills later
                    p["_y_batch"] = None
                del model_k
                _clear_gpu_memory()

            valid_idxs = [i for i, p in enumerate(population) if p.get("_x_batch") is not None]
            if not valid_idxs:
                if verbose:
                    print(f"  [{model_selection}] no valid kernel; falling back to default acqf path")
            elif model_selection == "oracle":
                scored = [i for i in valid_idxs if population[i].get("batch_y_best") is not None]
                if not scored:
                    pass
                else:
                    sign = 1 if maximize else -1
                    best_idx = max(scored, key=lambda i: sign * population[i]["batch_y_best"])
                    special_selected_idx = best_idx
                    special_x_next = population[best_idx]["_x_batch"].to(device=device, dtype=dtype)
                    special_y_next = population[best_idx]["_y_batch"].to(device=device, dtype=dtype)
                    if verbose:
                        print(f"  [oracle] selected {population[best_idx]['source']} "
                              f"(batch_y_best={population[best_idx]['batch_y_best']:.4f})")
            else:  # softmax_crps
                crps_vals = np.array([population[i]["crps"] for i in valid_idxs], dtype=float)
                # Replace inf with large positive (so they get tiny weight)
                crps_vals = np.where(np.isinf(crps_vals), crps_vals[~np.isinf(crps_vals)].max() * 2 if (~np.isinf(crps_vals)).any() else 1e9, crps_vals)
                logits = -crps_vals / max(softmax_temperature, 1e-6)
                logits -= logits.max()
                probs = np.exp(logits); probs /= probs.sum()
                rng = np.random.default_rng(seed + it * 1000 + 777)
                pick = rng.choice(len(valid_idxs), p=probs)
                sel_i = valid_idxs[pick]
                special_selected_idx = sel_i
                x_pick = population[sel_i]["_x_batch"].to(device=device, dtype=dtype)
                try:
                    y_pick = f(x_pick).reshape(-1, 1).to(device=device, dtype=dtype)
                    population[sel_i]["batch_y_best"] = float(best_val(y_pick))
                    population[sel_i]["_y_batch"] = y_pick.detach().cpu()
                    special_x_next = x_pick
                    special_y_next = y_pick
                    if verbose:
                        probs_str = ", ".join(f"{p:.2f}" for p in probs[:5])
                        print(f"  [softmax_crps] T={softmax_temperature} picked "
                              f"{population[sel_i]['source']} (p={probs[pick]:.3f}, top5 probs=[{probs_str}...])")
                except Exception:
                    special_x_next = None
                    special_y_next = None
            # Clean up x_batch / y_batch heavy tensors from non-selected; keep batch_y_best for logging
            for i, p in enumerate(population):
                if i != special_selected_idx:
                    p.pop("_x_batch", None)
                    p.pop("_y_batch", None)

            # Reorder population so rank-1 = selected kernel in log (oracle: by batch_y_best)
            if model_selection == "oracle":
                def _oracle_key(p):
                    v = p.get("batch_y_best")
                    if v is None:
                        return float("inf")
                    return -v if maximize else v
                population.sort(key=_oracle_key)
                if special_selected_idx is not None:
                    # After sort, find new index of the selected kernel via identity
                    sel_source = None
                    for i, p in enumerate(population):
                        if p.get("_x_batch") is not None:
                            sel_source = p.get("source")
                            special_selected_idx = i
                            break

        # --- Acquisition optimization (identical to v1) ---
        failed_acq = False
        _t_acqf_start = time.time()
        if special_x_next is not None:
            # Skip default acqf/oracle path — we already have x_next + y_next
            pass
        elif gp_model is None or not BOTORCH_AVAILABLE:
            failed_acq = True
            reason = "gp_model is None" if gp_model is None else "botorch not available"
            if verbose:
                print(f"  acqf skipped [{reason}]. Sobol fallback.")
            if logger:
                logger.log_acqf_error(
                    bo_iter=bo_round, n_evals=X.shape[0], acquisition=acquisition,
                    error_type="fitting_failed", error_msg=reason, tb=None,
                    best_kernel_source=population[0]["source"] if population else None,
                )
            fallback_seed = (seed + 88888 + it) if seed is not None else None
            sobol_fallback = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=fallback_seed)
            x_next = sobol_fallback.draw(q).to(device=device, dtype=dtype)
            for d in range(dim):
                lo, hi = bounds[d]
                x_next[:, d] = x_next[:, d] * (hi - lo) + lo
        else:
            f_best = y_for_model.max().item()
            model_device = next(gp_model.parameters()).device
            model_dtype = next(gp_model.parameters()).dtype
            X_baseline = X if acquisition == "qlognei" else None
            x_next, failed_acq = _optimize_acquisition_with_fallback(
                gp_model, bounds, acquisition, f_best, model_device, model_dtype,
                batch_size=q, X_baseline=X_baseline, verbose=verbose,
                prune_baseline=prune_baseline, seed=seed, sequential=sequential,
                logger=logger, bo_iter=bo_round, n_evals=X.shape[0],
                best_kernel_source=population[0]["source"] if population else None,
            )
            if x_next.dim() == 1:
                x_next = x_next.unsqueeze(0)

        _t_acqf_sec = time.time() - _t_acqf_start

        _t_oracle_start = time.time()
        if special_x_next is not None:
            x_next = special_x_next
            y_next_val = special_y_next
            # Ensure selected kernel is population[0] for downstream logging/cumulative
            if special_selected_idx is not None and special_selected_idx != 0:
                sel = population.pop(special_selected_idx)
                population.insert(0, sel)
                code = sel.get("code", code)
        else:
            y_next_val = f(x_next).reshape(-1, 1).to(device=device, dtype=dtype)
        _t_oracle_sec = time.time() - _t_oracle_start

        # `code` is the kernel that produced this step's proposal. Oracle re-sort can move rows so
        # population[0] != that kernel; align rank-1 before discard/strike logic.
        if code is not None and population:
            cs_align = code.strip()
            ix_align = next((i for i, p in enumerate(population) if p["code"].strip() == cs_align), None)
            if ix_align is not None and ix_align != 0:
                _s = population.pop(ix_align)
                population.insert(0, _s)

        best_before = best_val(y)
        X = torch.cat([X, x_next.to(device=device, dtype=dtype)], dim=0)
        y = torch.cat([y, y_next_val], dim=0)

        n_before = X.shape[0] - q
        for j in range(q):
            n_evals = n_before + j + 1
            best_so_far = best_val(y[:n_evals])
            history.append({"n_evals": n_evals, "best": best_so_far, "incumbent": best_so_far})
            if on_eval:
                regret = (best_so_far - optimal_val) if optimal_val is not None else None
                on_eval(n_evals, best_so_far, regret)

        # Persist after every oracle batch — lets us resume mid-run if the process dies.
        _save_checkpoint()

        # Pool as ranked *before* discard (matches acqf/oracle decision). Post-discard list can drop rows
        # and reorder rank-1, which made population_log disagree with the kernel that ran this step.
        population_pre_discard = list(population) if population else []
        acting_kernel_source = "unknown"

        # --- cumulative kernels carry formula ---
        if code is not None and population:
            cs_meta = code.strip()
            chosen_entry = next((p for p in population if p["code"].strip() == cs_meta), None)
            if chosen_entry is None:
                chosen_entry = population[0]
            chosen_src = chosen_entry.get("source", "")
            acting_kernel_source = chosen_src
            is_base = chosen_src.startswith("base_")
            best_after = best_val(y)
            improved = (best_after > best_before) if maximize else (best_after < best_before)
            # Discard if acqf failed, or if this batch did not improve global best when the proposal
            # came from the kernel (not Sobol fallback alone). Base kernels need BASE_STRIKE_LIMIT strikes.
            kernel_suggested = (not failed_acq) or (special_x_next is not None)
            bad_round = (failed_acq and gp_model is not None) or (kernel_suggested and not improved)

            discard_chosen = False
            if is_base:
                if bad_round:
                    n = base_no_improve_strikes.get(chosen_src, 0) + 1
                    base_no_improve_strikes[chosen_src] = n
                    if n >= BASE_STRIKE_LIMIT:
                        discard_chosen = True
                        base_no_improve_strikes.pop(chosen_src, None)
                        if verbose:
                            print(
                                f"  Base kernel {chosen_src} reached {BASE_STRIKE_LIMIT} strikes "
                                f"(no improvement or acqf failure). Removing."
                            )
                    elif verbose:
                        print(
                            f"  Base kernel {chosen_src} strike {n}/{BASE_STRIKE_LIMIT} "
                            f"(best_so_far {best_before:.6f} -> {best_after:.6f}, acqf_failed={failed_acq})."
                        )
            else:
                if failed_acq and gp_model is not None:
                    discard_chosen = True
                    if verbose:
                        print(f"  Acqf optimization failed. Discarding kernel: {chosen_src}")
                elif kernel_suggested and not improved:
                    discard_chosen = True
                    if verbose:
                        print(
                            f"  No improvement in best_so_far ({best_before:.6f} -> {best_after:.6f}). "
                            f"Discarding kernel: {chosen_src}"
                        )
            if discard_chosen:
                cs = code.strip()
                if chosen_src.startswith("base_"):
                    excluded_base_sources.add(chosen_src)
                    if verbose:
                        print(f"  Permanently excluding template base from pool until reset: {chosen_src}")
                population = [p for p in population if p["code"].strip() != cs]

        if code is not None:
            if population:
                has_base = any(p.get("source", "").startswith("base_") for p in population)
            else:
                has_base = False
            no_base_reset = bool(population) and not has_base
            if not population or not has_base:
                cumulative_kernels = []
                excluded_base_sources.clear()
                if no_base_reset:
                    population = []
                if verbose:
                    if no_base_reset:
                        print(
                            "  No template bases left in pool; cumulative + excluded bases reset "
                            "(re-seed from template bases next round). "
                            "base_no_improve_strikes are kept across this reset."
                        )
                    else:
                        print(
                            "  Population empty after discard; cumulative + excluded bases reset "
                            "(all template bases available next round). "
                            "base_no_improve_strikes are kept across this reset."
                        )
            else:
                cumulative_kernels = [
                    {"code": p["code"], "formula": p.get("formula", ""), "source": p.get("source", "unknown")}
                    for p in population
                ]

        elapsed = time.time() - start_time
        best_curr = best_val(y)

        acqf_total_count += 1
        if failed_acq:
            acqf_fail_count += 1

        best_kernel_source = acting_kernel_source
        if best_kernel_source == "unknown" and population:
            best_kernel_source = population[0].get("source", "unknown")
        if logger and population_pre_discard:
            # For oracle/softmax modes the stored `loss` is CRPS (internal score);
            # pass the inner label so the score column shows "CRPS" correctly.
            _log_label = inner_model_selection if special_selection else model_selection
            logger.log_population_snapshot(
                bo_iter=bo_round, n_evals=X.shape[0], best_so_far=best_curr,
                elapsed_sec=elapsed, population=population_pre_discard,
                selected_source=best_kernel_source if best_kernel_source != "unknown" else None,
                model_selection=_log_label,
            )
        if history_path:
            with open(history_path, "a", encoding="utf-8") as hf:
                hf.write(f"bo_iter={bo_round} oracle_queries_batch={q} oracle_evals_cumulative={X.shape[0]} "
                         f"best_so_far={best_curr} elapsed_sec={elapsed:.6f} "
                         f"acqf_random_fallback={failed_acq}({acqf_fail_count}/{acqf_total_count}) "
                         f"best_kernel={best_kernel_source}\n")

        if logger:
            logger.log_bo_iteration(
                bo_iter=bo_round, n_evals=X.shape[0], best_so_far=best_curr,
                elapsed_sec=elapsed, kernel_code=code, acqf_failed=failed_acq, extra=evol_stats,
            )

        if verbose:
            print(f"  -> New eval: n_evals={X.shape[0]}, best={best_so_far:.4f}")

        if gp_model is not None:
            del gp_model
        _t_gpu_start = time.time()
        _clear_gpu_memory()
        _t_gpu_sec = time.time() - _t_gpu_start

        if logger:
            _t_total_sec = time.time() - _t_iter_start
            _timing_extra = {}
            if evol_stats:
                _timing_extra = {k: evol_stats[k] for k in (
                    "t_base_validate_sec", "t_llm_formula_sec",
                    "t_llm_codegen_sec", "t_validate_sec",
                    "t_fit_sec", "t_surrogate_refit_sec",
                ) if k in evol_stats}
            logger.log_timing(
                bo_iter=bo_round, kernel_evol_sec=_t_evol_sec,
                acqf_opt_sec=_t_acqf_sec, oracle_sec=_t_oracle_sec,
                gpu_cleanup_sec=_t_gpu_sec, total_iter_sec=_t_total_sec,
                extra=_timing_extra,
            )

    if _client_generate_orig is not None and client is not None:
        client.generate = _client_generate_orig

    return X, y, history


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1 (got {args.batch_size})")

    os.environ["ORACLE_N_JOBS"] = str(args.oracle_n_jobs)
    if args.llm_interval <= 0:
        raise ValueError(f"--llm_interval must be >= 1 (got {args.llm_interval})")

    from hdbo.bo import set_backends
    set_backends(fit=args.fit_backend, acqf=args.acqf_backend)
    print(f"[backends] fit={args.fit_backend} acqf={args.acqf_backend}")

    from hdbo.llm_client import create_llm_client

    client = create_llm_client(
        backend=args.llm_backend,
        model_name=args.model_name,
        base_url=args.vllm_url if args.llm_backend == "vllm" else None,
        seed=args.llm_seed,
    )

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed.")
            use_wandb = False
    if use_wandb:
        from datetime import datetime as _dt
        _run_name = "_".join([
            _dt.now().strftime("%m%d_%H%M"),
            args.acquisition, args.model_name, f"seed{args.seed}",
        ])
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=_run_name, config=vars(args))

    output_dir = args.output_dir or os.path.join(_root, "hdbo", "output")
    method = "llm"
    if args.model_selection != "loss":
        method = f"{method}_{args.model_selection}"

    results: Dict[str, Any] = {}
    wandb_step_offset = [0]

    def make_on_eval(benchmark_name):
        if not use_wandb:
            return None
        def _on_eval(n_evals, best_val, regret):
            step = wandb_step_offset[0] + n_evals
            log_dict = {f"{benchmark_name}/best_val": best_val}
            if regret is not None:
                log_dict[f"{benchmark_name}/regret"] = regret
            wandb.log(log_dict, step=step)
        return _on_eval

    for bench in args.benchmark:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {bench} (formula-level kernel discovery)")
        print(f"{'=' * 60}")

        resume_data = None
        if args.resume_from:
            bench_dir = args.resume_from
            ckpt_path = os.path.join(bench_dir, f"{bench}_data.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"--resume_from={bench_dir} has no checkpoint {bench}_data.pt"
                )
            resume_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"  Resuming from {ckpt_path} (n_evals={resume_data['X'].shape[0]})")
            os.makedirs(bench_dir, exist_ok=True)
        else:
            safe_model = args.model_name.replace("/", "-")
            subdir = f"seed_{args.seed}_{safe_model}_{method}_{args.acquisition}"
            if args.dim is not None:
                subdir = f"{subdir}_dim{args.dim}"
            if args.data_fname:
                from datetime import datetime
                now = datetime.now()
                ts = now.strftime("%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
                subdir = f"{ts}_{subdir}"
            bench_dir = os.path.join(output_dir, bench, subdir)
            os.makedirs(bench_dir, exist_ok=True)

        if args.verbose_log:
            params_dir = os.path.join(bench_dir, "logs")
            os.makedirs(params_dir, exist_ok=True)
            params = {"benchmark": bench, "cli_argv": sys.argv, **vars(args)}
            with open(os.path.join(params_dir, "run_params.json"), "w") as _pf:
                json.dump(params, _pf, indent=2, default=str)
            with open(os.path.join(params_dir, "run_command.sh"), "w") as _cf:
                _cf.write("#!/bin/bash\n")
                _cf.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")

        info = HDBO_BENCHMARKS[bench]
        optimal_val = info.get("min_val")
        maximize = info.get("maximize", True)
        n_init = args.n_init if args.n_init is not None else info.get("n_init_default", 5)
        if n_init <= 0:
            raise ValueError(f"--n_init resolved to {n_init} for {bench}; must be >= 1")

        X, y, history = _run_bo_llm_single(
            benchmark_name=bench,
            dim=args.dim,
            n_init=n_init,
            n_iter=args.n_iter,
            batch_size=args.batch_size,
            acquisition=args.acquisition,
            kernel_evolve_gens=args.kernel_gens,
            kernel_evolve_mutations=args.kernel_mutations,
            kernel_evolve_compositions=args.kernel_compositions,
            population_size=args.population_size,
            client=client,
            seed=args.seed,
            verbose=not args.quiet,
            save_dir=bench_dir,
            model_selection=args.model_selection,
            max_eval_time=args.max_eval_time,
            max_workers=args.max_workers,
            llm_interval=args.llm_interval,
            prune_baseline=args.prune_baseline,
            sequential=args.sequential,
            on_eval=make_on_eval(bench),
            verbose_log=args.verbose_log,
            temperature=args.temperature,
            codegen_temperature=args.codegen_temperature,
            max_retry=args.max_retry,
            simplicity_prompt=args.simplicity_prompt,
            score_subsample=args.score_subsample,
            num_eval=args.num_eval,
            softmax_temperature=args.softmax_temperature,
            resume_data=resume_data,
            method_tag=method,
            prompt_highdim=(args.prompt_highdim == "on"),
            prompt_psd=(args.prompt_psd == "on"),
            base_kernel_preset=args.base_kernel_preset,
        )

        best_val_result = y.max().item() if maximize else y.min().item()
        best_idx = y.argmax().item() if maximize else y.argmin().item()
        best_x = X[best_idx].tolist()
        print(f"Best: {best_val_result:.6f} at x = {best_x}")
        print(f"Known optimal: {optimal_val}" + (" (minimize)" if not maximize else ""))

        n_evals = [h["n_evals"] for h in history]
        best_vals = [h["best"] for h in history]
        regret = [b - optimal_val for b in best_vals] if optimal_val is not None else None

        if use_wandb:
            wandb_step_offset[0] += X.shape[0]
            wandb.log({
                f"{bench}/best_value": best_val_result,
                f"{bench}/regret_final": regret[-1] if regret else None,
            })

        actual_dim = X.shape[1]
        regret_trace = {
            "benchmark": bench, "dim": actual_dim, "seed": args.seed,
            "method": method, "acquisition": args.acquisition,
            "optimal_val": optimal_val, "n_init": n_init,
            "n_evals": n_evals, "best_val": best_vals, "regret": regret,
        }
        results[bench] = {
            "best_value": best_val_result, "best_x": best_x,
            "history": history, "n_evals": X.shape[0],
            "acquisition": args.acquisition, "regret_trace": regret_trace,
        }

        with open(os.path.join(bench_dir, "result.json"), "w") as fh:
            json.dump({k: v for k, v in results[bench].items() if k != "regret_trace"}, fh, indent=2)
        with open(os.path.join(bench_dir, "regret_trace.json"), "w") as fh:
            json.dump(regret_trace, fh, indent=2)

        data_path = os.path.join(bench_dir, f"{bench}_data.pt")
        torch.save({
            "X": X.cpu() if X.is_cuda else X,
            "y": y.cpu() if y.is_cuda else y,
            "benchmark": bench, "seed": args.seed,
            "method": method, "acquisition": args.acquisition,
            "best_value": best_val_result, "best_x": best_x,
            "history": history, "regret_trace": regret_trace,
        }, data_path)
        print(f"  Saved: {data_path}, regret_trace.json")

    if use_wandb:
        wandb.log({"best_value": {b: r["best_value"] for b, r in results.items()}})
        wandb.finish()


if __name__ == "__main__":
    main()
