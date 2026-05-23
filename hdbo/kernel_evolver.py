"""
Formula-level kernel evolution: LLM-driven mutation/composition + code generation.

Architecture (2-phase per generation):
  Phase 1 (fast): Codegen + validate — parse kernel, check shapes/grad/PSD
  Phase 2 (heavy): Fit all validated kernels in one parallel ProcessPool batch

Mutation and composition operate on formulas (not code) and run in parallel;
validation is separated from fitting so the heavy work happens in a single
parallel batch.
"""
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp

from .bo import _clear_gpu_memory
from .mle_bo import MLEBackend, fix_code_with_llm
from .llm_client import LLMClient
from .kernel_formula import (
    BASE_FORMULAS,
    generate_formulas_mutation,
    generate_formulas_composition,
    generate_code_from_formula,
    fix_formula,
    validate_kernel_code,
    fit_and_score_kernel,
    is_composition_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str, verbose: bool, bo_iter: int | None = None):
    if verbose:
        prefix = f"[BO iter {bo_iter}] " if bo_iter is not None else ""
        print(f"  {prefix}{msg}")


def _save_code(code: str, path: str, save_dir: str | None = None):
    if save_dir:
        full_path = os.path.join(save_dir, path)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass


def _save_text(text: str, path: str, save_dir: str | None = None):
    if save_dir:
        full_path = os.path.join(save_dir, path)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# ProcessPool worker for fit phase
# ---------------------------------------------------------------------------

def _fit_worker(args):
    """Top-level worker for ProcessPoolExecutor — fit + score a single kernel."""
    idx, code, X, y, device_str, selection, timeout, bounds = args[:8]
    score_subsample = args[8] if len(args) > 8 else 1.0
    num_eval = args[9] if len(args) > 9 else 100
    X = X.to(device_str)
    y = y.to(device_str)
    from hdbo.kernel_formula import fit_and_score_kernel
    score, model, eval_time, fail_reason, tb_str, nll, timing = fit_and_score_kernel(
        code, X, y, model_selection=selection, timeout=timeout, bounds=bounds,
        score_subsample=score_subsample, num_eval=num_eval,
    )
    state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()} if model is not None else None
    return idx, code, score, state_dict, eval_time, fail_reason, tb_str, nll, timing


# ---------------------------------------------------------------------------
# Formula logging
# ---------------------------------------------------------------------------

def _save_formula_log(formulas_dir, gen, stage, formulas, results):
    if not formulas_dir:
        return
    os.makedirs(formulas_dir, exist_ok=True)
    formula_text = ""
    for i, f in enumerate(formulas):
        formula_text += f"=== {stage.upper()} {i} ===\n{f}\n\n"
    _save_text(formula_text, f"gen{gen}_{stage}s.txt", formulas_dir)

    jsonl_path = os.path.join(formulas_dir, f"gen{gen}_{stage}_results.jsonl")
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "w") as fh:
        for r in results:
            row = {
                "stage": stage, "gen": gen,
                "formula": r.get("formula", ""),
                "success": r.get("success", False),
                "loss": r.get("loss", float("inf")),
                "nll": r.get("nll", float("inf")),
                "eval_time": r.get("eval_time", 0.0),
                "fail_reason": r.get("fail_reason"),
                "tb_str": r.get("tb_str", ""),
                "source": r.get("source", ""),
            }
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Batch fit: run all validated kernels through ProcessPool
# ---------------------------------------------------------------------------

def _batch_fit(
    candidates: list[dict],
    train_x_cpu: torch.Tensor,
    train_y_cpu: torch.Tensor,
    device_str: str,
    model_selection: str,
    max_eval_time: float,
    bounds: list | None,
    max_workers: int,
    score_subsample: float = 1.0,
    num_eval: int = 100,
) -> list[dict]:
    """Fit all validated kernels in parallel. Returns updated candidate dicts."""
    if not candidates:
        return candidates

    worker_args = [
        (i, c["code"], train_x_cpu, train_y_cpu, device_str, model_selection, max_eval_time, bounds, score_subsample, num_eval)
        for i, c in enumerate(candidates)
    ]
    nw = min(len(worker_args), max_workers)
    raw_results = [None] * len(worker_args)

    if nw <= 1:
        for i, arg in enumerate(worker_args):
            try:
                raw_results[i] = _fit_worker(arg)
            except Exception as e:
                raw_results[i] = (i, "", float("inf"), None, 0.0, f"worker_exception: {e}", str(e), float("inf"), {})
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as executor:
            fut_map = {executor.submit(_fit_worker, a): i for i, a in enumerate(worker_args)}
            for fut in as_completed(fut_map):
                i = fut_map[fut]
                try:
                    raw_results[i] = fut.result()
                except Exception as e:
                    raw_results[i] = (i, "", float("inf"), None, 0.0, f"worker_exception: {e}", str(e), float("inf"), {})

    for c, raw in zip(candidates, raw_results):
        _, _, score, state_dict, eval_time, fail_reason, tb_str, nll, timing = raw
        success = score < float("inf") and not math.isnan(score)
        c["loss"] = score
        c["nll"] = nll
        c["eval_time"] = eval_time
        c["state_dict"] = state_dict
        c["success"] = success
        c["fail_reason"] = fail_reason if not success else None
        c["tb_str"] = tb_str if not success else ""
        src = c.get("source", "?")
        status = f"score={score:.4f}" if success else f"FAILED ({fail_reason})"
        print(f"    [BatchFit] {src}: {status}, eval_time={eval_time:.1f}s")

    return candidates


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def evolve_kernel_for_bo(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    client: LLMClient | None = None,
    n_generations: int = 2,
    n_mutations: int = 3,
    n_parents: int = 5,
    population_size: int = 5,
    kernel_compositions: int = 3,
    seed: int | None = None,
    verbose: bool = True,
    bo_iter: int | None = None,
    save_dir: str | None = None,
    seed_population: list | None = None,
    model_selection: str = "loss",
    max_eval_time: float = 120.0,
    max_workers: int = 5,
    bounds: list | None = None,
    stats_out: dict | None = None,
    logger=None,
    temperature: float = 1.0,
    codegen_temperature: float = 0.3,
    max_retry: int = 0,
    simplicity_prompt: bool = False,
    score_subsample: float = 1.0,
    num_eval: int = 100,
    excluded_base_sources: frozenset[str] | set[str] | None = None,
    prompt_highdim: bool = True,
    prompt_psd: bool = True,
    base_kernel_preset: str = "default",
) -> tuple:
    """Formula-level kernel evolution with 2-phase eval (validate → batch-fit).

    excluded_base_sources: template bases (sources like ``base_matern52``) not to add this round;
    used by BO outer loop after a base is discarded under the strike policy.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    backend = MLEBackend()
    input_dim = train_x.shape[1] if train_x.dim() > 1 else 1
    base_names, base_codes = backend.get_base_codes(input_dim, preset=base_kernel_preset)

    iter_dir = os.path.join(save_dir, f"bo_iter_{bo_iter}") if save_dir and bo_iter is not None else None
    formulas_dir = os.path.join(iter_dir, "logs", "formulas") if iter_dir else None

    if stats_out is not None:
        stats_out.update({
            "n_base_total": len(base_codes), "n_base_failed": 0, "base_fail_reasons": [],
            "n_cumulative_total": 0, "n_cumulative_failed": 0,
            "n_mutations_total": 0, "n_mutations_succeeded": 0, "n_mutations_failed": 0,
            "mutation_fail_reasons": [],
            "n_compositions_total": 0, "n_compositions_succeeded": 0, "n_compositions_failed": 0,
            "composition_fail_reasons": [],
            "n_codegen_failed": 0, "n_validate_failed": 0,
            "t_base_validate_sec": 0.0, "t_llm_formula_sec": 0.0,
            "t_llm_codegen_sec": 0.0, "t_validate_sec": 0.0,
            "t_fit_sec": 0.0, "t_surrogate_refit_sec": 0.0,
        })

    _tx = train_x.cpu()
    _ty = train_y.cpu()
    _dev = str(train_x.device)
    device = train_x.device
    dtype = train_x.dtype
    base_codes_stripped = {c.strip() for c in base_codes}
    _excluded_bases = frozenset(excluded_base_sources) if excluded_base_sources else frozenset()

    # ========================================================================
    # Collect all kernels to validate, then fit in one batch
    # ========================================================================

    # --- Base kernels: validate ---
    _t_val = time.time()
    to_fit: list[dict] = []  # all candidates that pass validation

    for i, code in enumerate(base_codes):
        base_src = f"base_{base_names[i]}"
        if base_src in _excluded_bases:
            _log(
                f"Base {base_names[i]} | Skipped (excluded after BO discard policy)",
                verbose, bo_iter,
            )
            continue
        ok, fail_reason, tb_str = validate_kernel_code(code, input_dim, device, dtype, train_x, bounds)
        formula = BASE_FORMULAS.get(base_names[i], f"[base: {base_names[i]}]")
        if ok:
            to_fit.append({
                "code": code.strip(), "formula": formula,
                "source": base_src, "is_base": True,
            })
            _log(f"Base {base_names[i]} | Validated OK", verbose, bo_iter)
        else:
            # Force-preserve base kernels even if validation fails (with inf loss)
            to_fit.append({
                "code": code.strip(), "formula": formula,
                "source": base_src, "is_base": True,
                "loss": float("inf"), "nll": float("inf"), "eval_time": 0.0,
                "state_dict": None, "success": False, "fail_reason": fail_reason,
                "skip_fit": True,
            })
            _log(f"Base {base_names[i]} | Validation FAILED ({fail_reason})", verbose, bo_iter)
            if stats_out:
                stats_out["n_base_failed"] += 1
                stats_out["base_fail_reasons"].append({"kernel": base_names[i], "reason": fail_reason})
            if logger:
                logger.log_kernel_error(bo_iter, "base", i, code, fail_reason, tb_str or "")

    # --- Cumulative kernels: validate ---
    if seed_population:
        cumul_kernels = [item for item in seed_population
                         if (item["code"] if isinstance(item, dict) else item).strip() not in base_codes_stripped]
        if stats_out:
            stats_out["n_cumulative_total"] = len(cumul_kernels)
        for c_idx, item in enumerate(cumul_kernels):
            code = item["code"] if isinstance(item, dict) else item
            ok, fail_reason, tb_str = validate_kernel_code(code, input_dim, device, dtype, train_x, bounds)
            original_source = item.get("source", f"cumulative_{c_idx}") if isinstance(item, dict) else f"cumulative_{c_idx}"
            original_formula = item.get("formula", "") if isinstance(item, dict) else ""
            if ok:
                to_fit.append({
                    "code": code.strip(), "formula": original_formula,
                    "source": original_source, "is_base": False,
                })
            else:
                _log(f"  Cumulative {c_idx+1} | Validation FAILED ({fail_reason})", verbose, bo_iter)
                if stats_out:
                    stats_out["n_cumulative_failed"] += 1

    if stats_out:
        stats_out["t_base_validate_sec"] = time.time() - _t_val

    # --- Fit all validated kernels in one parallel batch ---
    fitlist = [c for c in to_fit if not c.get("skip_fit")]
    _t_fit = time.time()
    if fitlist:
        _log(f"[Fit Phase] Fitting {len(fitlist)} validated kernels in parallel...", verbose, bo_iter)
        fitlist = _batch_fit(
            fitlist, _tx, _ty, _dev, model_selection, max_eval_time, bounds, max_workers,
            score_subsample=score_subsample, num_eval=num_eval,
        )
    if stats_out:
        stats_out["t_fit_sec"] += time.time() - _t_fit

    # Build initial population from fit results + skipped bases
    population: list[dict] = []
    for c in to_fit:
        if c.get("skip_fit"):
            population.append({
                "code": c["code"], "formula": c["formula"],
                "loss": float("inf"), "nll": float("inf"), "eval_time": 0.0,
                "state_dict": None, "source": c["source"],
            })
        else:
            success = c.get("success", False)
            if success:
                population.append({
                    "code": c["code"], "formula": c["formula"],
                    "loss": c["loss"], "nll": c["nll"], "eval_time": c["eval_time"],
                    "state_dict": c["state_dict"], "source": c["source"],
                })
                _log(f"  {c['source']} | {model_selection.upper()}: {c['loss']:.4f}", verbose, bo_iter)
            else:
                if c.get("is_base"):
                    population.append({
                        "code": c["code"], "formula": c["formula"],
                        "loss": float("inf"), "nll": float("inf"), "eval_time": 0.0,
                        "state_dict": None, "source": c["source"],
                    })
                _log(f"  {c['source']} | Fit FAILED ({c.get('fail_reason')})", verbose, bo_iter)

    _clear_gpu_memory()
    if not population:
        return None, None, float("inf"), []

    population.sort(key=lambda x: x["loss"])
    _log(f"Generation 0 | Best {model_selection.upper()}: {population[0]['loss']:.4f} (pool: {len(population)})",
         verbose, bo_iter)

    # ========================================================================
    # Generation loop
    # ========================================================================
    for gen in range(1, n_generations + 1):
        _log(f"--- Generation {gen} ---", verbose, bo_iter)

        top_k = population[:min(n_parents, len(population))]
        top_formulas_with_loss = [(p["formula"], p["loss"]) for p in top_k]

        comp_parents: list[dict] = []
        seen_cp: set[str] = set()
        for p in population:
            if is_composition_output(p):
                continue
            ck = p["code"].strip()
            if ck in seen_cp:
                continue
            seen_cp.add(ck)
            comp_parents.append(p)
            if len(comp_parents) >= n_parents:
                break
        comp_formulas_with_loss = [(p["formula"], p["loss"]) for p in comp_parents]

        # --- Phase 1a: Mutation + Composition formulas (parallel LLM) ---
        _t_formula = time.time()
        mutated_formulas, composed_formulas = [], []
        if client:
            with ThreadPoolExecutor(max_workers=2) as tex:
                mut_f = tex.submit(generate_formulas_mutation,
                                   top_formulas_with_loss, n_mutations, client, temperature, model_selection,
                                   simplicity_prompt, prompt_highdim, prompt_psd)
                if kernel_compositions > 0 and len(comp_formulas_with_loss) >= 2:
                    comp_f = tex.submit(generate_formulas_composition,
                                        comp_formulas_with_loss, kernel_compositions, client, temperature, model_selection,
                                        simplicity_prompt, prompt_psd)
                else:
                    comp_f = None
                    if kernel_compositions > 0 and verbose:
                        _log(
                            f"[Formula] Skipping composition LLM: need 2+ non-composition parents, "
                            f"have {len(comp_formulas_with_loss)}",
                            verbose, bo_iter,
                        )
                mutated_formulas = mut_f.result()
                composed_formulas = comp_f.result() if comp_f is not None else []
        if stats_out:
            stats_out["t_llm_formula_sec"] += time.time() - _t_formula
            stats_out["n_mutations_total"] += len(mutated_formulas)
            stats_out["n_compositions_total"] += len(composed_formulas)

        _log(f"[Formula] Got {len(mutated_formulas)} mutations, {len(composed_formulas)} compositions", verbose, bo_iter)

        # --- Phase 1b: Codegen (parallel LLM) ---
        all_items = []
        for m_idx, formula in enumerate(mutated_formulas):
            all_items.append({"idx": m_idx, "formula": formula,
                              "source": f"bo{bo_iter}_mutation_gen{gen}_m{m_idx}", "stage": "mutation"})
        for c_idx, formula in enumerate(composed_formulas):
            all_items.append({"idx": c_idx, "formula": formula,
                              "source": f"bo{bo_iter}_composition_gen{gen}_c{c_idx}", "stage": "composition"})

        _t_cg = time.time()
        def _codegen_one(item):
            code = generate_code_from_formula(item["formula"], client, temperature=codegen_temperature)
            return {**item, "code": code}

        if client and all_items:
            with ThreadPoolExecutor(max_workers=min(len(all_items), 10)) as tex:
                cg_futures = {tex.submit(_codegen_one, it): it for it in all_items}
                codegen_results = []
                for f in as_completed(cg_futures):
                    try:
                        codegen_results.append(f.result())
                    except Exception:
                        codegen_results.append({**cg_futures[f], "code": None})
        else:
            codegen_results = [{**it, "code": None} for it in all_items]

        if stats_out:
            stats_out["t_llm_codegen_sec"] += time.time() - _t_cg

        # --- Phase 1c: Validate + retry ---
        _t_val = time.time()
        new_to_fit = []  # validated, ready for fit
        gen_log_results = []  # for formula logging

        for cr in codegen_results:
            if cr["code"] is None:
                _log(f"  -> {cr['stage'].capitalize()} {cr['source']} | codegen_failed", verbose, bo_iter)
                gen_log_results.append({**cr, "success": False, "fail_reason": "codegen_failed", "tb_str": ""})
                if stats_out:
                    stats_out["n_codegen_failed"] += 1
                    stats_out[f"{cr['stage']}_fail_reasons"].append("codegen_failed")
                continue

            if iter_dir:
                _save_code(cr["code"].strip(), f"{cr['source']}.py", iter_dir)

            ok, fail_reason, tb_str = validate_kernel_code(cr["code"], input_dim, device, dtype, train_x, bounds)
            if ok:
                new_to_fit.append({
                    "code": cr["code"].strip(), "formula": cr["formula"],
                    "source": cr["source"], "stage": cr["stage"], "idx": cr["idx"],
                })
                continue

            # Validation failed — log and try retry
            _log(f"  -> {cr['stage'].capitalize()} {cr['source']} | FAILED ({fail_reason})", verbose, bo_iter)
            if stats_out:
                stats_out["n_validate_failed"] += 1
                stats_out[f"{cr['stage']}_fail_reasons"].append(fail_reason)
            if logger and cr["code"]:
                logger.log_kernel_error(bo_iter, f"{cr['stage']}_gen{gen}", cr["idx"],
                                        cr["code"], fail_reason, tb_str or "")

            # --- Retry ---
            rescued = False
            if max_retry > 0 and client:
                for attempt in range(max_retry):
                    error_msg = tb_str or fail_reason
                    if fail_reason == "psd_check":
                        fixed_formula = fix_formula(cr["formula"], error_msg, client)
                        if not fixed_formula:
                            break
                        retry_code = generate_code_from_formula(fixed_formula, client, temperature=codegen_temperature)
                        retry_formula = fixed_formula
                    else:
                        retry_code = fix_code_with_llm(cr["code"], fail_reason, error_msg, client=client, temperature=0.7)
                        retry_formula = cr["formula"]

                    if not retry_code or retry_code == cr["code"]:
                        break

                    ok2, fr2, tb2 = validate_kernel_code(retry_code, input_dim, device, dtype, train_x, bounds)
                    if ok2:
                        new_to_fit.append({
                            "code": retry_code.strip(), "formula": retry_formula,
                            "source": cr["source"] + f"_retry{attempt+1}",
                            "stage": cr["stage"], "idx": cr["idx"],
                        })
                        _log(f"  -> {cr['stage'].capitalize()} {cr['source']} | Retry {attempt+1} Validated OK", verbose, bo_iter)
                        if iter_dir:
                            _save_code(retry_code.strip(), f"{cr['source']}_retry{attempt+1}.py", iter_dir)
                        rescued = True
                        break
                    else:
                        _log(f"  -> {cr['stage'].capitalize()} {cr['source']} | Retry {attempt+1} FAILED ({fr2})", verbose, bo_iter)
                        if logger:
                            logger.log_kernel_error(bo_iter, f"{cr['stage']}_gen{gen}_retry{attempt+1}",
                                                    cr["idx"], retry_code, fr2, tb2 or "")

            gen_log_results.append({
                **cr, "success": rescued, "fail_reason": fail_reason if not rescued else None,
                "tb_str": tb_str if not rescued else "",
            })

        if stats_out:
            stats_out["t_validate_sec"] += time.time() - _t_val

        # --- Phase 2: Batch fit all validated new kernels ---
        _t_fit = time.time()
        if new_to_fit:
            _log(f"[Fit Phase] Fitting {len(new_to_fit)} validated new kernels in parallel...", verbose, bo_iter)
            new_to_fit = _batch_fit(
                new_to_fit, _tx, _ty, _dev, model_selection, max_eval_time, bounds, max_workers,
                score_subsample=score_subsample, num_eval=num_eval,
            )
        if stats_out:
            stats_out["t_fit_sec"] += time.time() - _t_fit

        # --- Collect results ---
        new_candidates = []
        for c in new_to_fit:
            stage = c.get("stage", "mutation")
            if c.get("success"):
                new_candidates.append({
                    "code": c["code"], "formula": c["formula"],
                    "loss": c["loss"], "nll": c["nll"], "eval_time": c["eval_time"],
                    "state_dict": c["state_dict"], "source": c["source"],
                    "stage": stage,
                })
                _log(f"  -> {stage.capitalize()} {c['source']} | Success | {model_selection.upper()}: {c['loss']:.4f}", verbose, bo_iter)
                if stats_out:
                    stats_out[f"n_{stage}s_succeeded"] += 1
            else:
                _log(f"  -> {stage.capitalize()} {c['source']} | Fit FAILED ({c.get('fail_reason')})", verbose, bo_iter)
                if stats_out:
                    stats_out[f"n_{stage}s_failed"] += 1
                if logger and c.get("code"):
                    logger.log_kernel_error(bo_iter, f"{stage}_gen{gen}_fit", c.get("idx", 0),
                                            c["code"], c.get("fail_reason", ""), c.get("tb_str", ""))

            gen_log_results.append({
                "formula": c["formula"], "source": c["source"], "stage": stage,
                "success": c.get("success", False), "loss": c.get("loss", float("inf")),
                "nll": c.get("nll", float("inf")), "eval_time": c.get("eval_time", 0.0),
                "fail_reason": c.get("fail_reason"), "tb_str": c.get("tb_str", ""),
            })

        # --- Save formula logs ---
        if formulas_dir:
            mut_logs = [r for r in gen_log_results if r.get("stage") == "mutation"]
            comp_logs = [r for r in gen_log_results if r.get("stage") == "composition"]
            _save_formula_log(formulas_dir, gen, "mutation", mutated_formulas, mut_logs)
            _save_formula_log(formulas_dir, gen, "composition", composed_formulas, comp_logs)
            for cr in codegen_results:
                if cr.get("code"):
                    _save_code(cr["code"], f"{cr['source']}_codegen.py", formulas_dir)

        # --- Selection ---
        population.extend(new_candidates)
        population.sort(key=lambda x: x["loss"])

        seen = set()
        unique = []
        for ind in population:
            c = ind["code"].strip()
            if c not in seen:
                seen.add(c)
                unique.append(ind)

        base_pool = [ind for ind in unique if ind["code"].strip() in base_codes_stripped]
        non_base_pool = [ind for ind in unique if ind["code"].strip() not in base_codes_stripped]
        n_non_base = max(0, population_size - len(base_pool))
        population = base_pool + non_base_pool[:n_non_base]
        population.sort(key=lambda x: x["loss"])

        _log(f"Generation {gen} Best {model_selection.upper()}: {population[0]['loss']:.4f}", verbose, bo_iter)
        if iter_dir:
            _save_code(population[0]["code"], f"gen_{gen}_best.py", iter_dir)
        _clear_gpu_memory()

    # ========================================================================
    # Final refit of best kernel
    # ========================================================================
    best = population[0]
    if iter_dir:
        _save_code(best["code"], "best.py", iter_dir)
    _t_refit = time.time()
    score, best_model, _, _, _, _, _ = fit_and_score_kernel(
        best["code"], train_x, train_y, model_selection=model_selection,
        state_dict=best.get("state_dict"), timeout=max_eval_time, bounds=bounds,
        score_subsample=score_subsample, num_eval=num_eval,
    )
    if stats_out:
        stats_out["t_surrogate_refit_sec"] = time.time() - _t_refit

    return best["code"], best_model, best["loss"], population
