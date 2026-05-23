"""
Structured JSONL logger for verbose BO diagnostics.

Usage:
    logger = BoLogger(log_dir)   # creates logs/ subdirectory
    logger.log_kernel_eval(...)  # append one record to kernel_eval.jsonl
    logger.log_kernel_error(...) # append one record to kernel_errors.jsonl
    logger.log_bo_iteration(...) # append one record to bo_iterations.jsonl
    logger.log_evolution_summary(...)  # append one record to evolution_summary.jsonl

All methods are no-ops when logger is None, so callers can do:
    if logger: logger.log_kernel_eval(...)
"""
import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Optional


class BoLogger:
    """Append-only JSONL logger for BO diagnostics."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._files = {
            "kernel_eval": os.path.join(log_dir, "kernel_eval.jsonl"),
            "kernel_errors": os.path.join(log_dir, "kernel_errors.log"),
            "bo_iterations": os.path.join(log_dir, "bo_iterations.jsonl"),
            "evolution_summary": os.path.join(log_dir, "evolution_summary.jsonl"),
            "population_log": os.path.join(log_dir, "population_log.txt"),
            "acqf_errors": os.path.join(log_dir, "acqf_errors.log"),
            "composition_errors": os.path.join(log_dir, "composition_errors.log"),
            "timing": os.path.join(log_dir, "timing.jsonl"),
        }

    def _append(self, key: str, record: dict):
        record["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._files[key], "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, indent=2) + "\n\n")
        except OSError:
            pass

    # --- kernel evaluation (every eval, success or fail) ---
    def log_kernel_eval(
        self,
        bo_iter: int,
        stage: str,
        kernel_index: int,
        code: str,
        score: float,
        eval_time: float,
        success: bool,
        extra: Optional[dict] = None,
    ):
        rec = {
            "bo_iter": bo_iter,
            "stage": stage,
            "kernel_index": kernel_index,
            "code": code,
            "score": score,
            "eval_time": eval_time,
            "success": success,
        }
        if extra:
            rec.update(extra)
        self._append("kernel_eval", rec)

    # --- kernel errors (detailed failure info, human-readable format) ---
    def log_kernel_error(
        self,
        bo_iter: int,
        stage: str,
        kernel_index: int,
        code: str,
        error_type: str,
        error_msg: str,
        tb: Optional[str] = None,
        attempt: int = 0,
        max_retry: int = 0,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        retry_tag = f"  attempt={attempt}/{max_retry}" if max_retry > 0 else ""
        lines = [
            sep,
            f"[{timestamp}]  bo_iter={bo_iter}  stage={stage}  kernel_index={kernel_index}{retry_tag}",
            f"ERROR TYPE: {error_type}",
            "",
            "--- ERROR MESSAGE ---",
            (error_msg or "").rstrip(),
        ]
        if tb and tb.strip() and tb.strip() != (error_msg or "").strip():
            lines += ["", "--- TRACEBACK ---", tb.rstrip()]
        lines += ["", "--- KERNEL CODE ---", (code or "").rstrip(), ""]

        entry = "\n".join(lines) + "\n"
        try:
            with open(self._files["kernel_errors"], "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass

    def log_kernel_retry_success(
        self,
        bo_iter: int,
        stage: str,
        kernel_index: int,
        fixed_code: str,
        original_error_type: str,
        attempt: int,
        max_retry: int,
        score: float,
    ):
        """Log when a previously failed kernel was successfully fixed by LLM retry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        lines = [
            sep,
            f"[{timestamp}]  bo_iter={bo_iter}  stage={stage}  kernel_index={kernel_index}  attempt={attempt}/{max_retry}",
            f"RETRY SUCCESS (fixed from: {original_error_type})  score={score:.6f}",
            "",
            "--- FIXED KERNEL CODE ---",
            (fixed_code or "").rstrip(),
            "",
        ]
        entry = "\n".join(lines) + "\n"
        try:
            with open(self._files["kernel_errors"], "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass

    # --- composition errors (separate file for easier diagnosis) ---
    def log_composition_error(
        self,
        bo_iter: int,
        stage: str,
        kernel_index: int,
        code: str,
        error_type: str,
        error_msg: str,
        tb: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        lines = [
            sep,
            f"[{timestamp}]  bo_iter={bo_iter}  stage={stage}  kernel_index={kernel_index}",
            f"ERROR TYPE: {error_type}",
            "",
            "--- ERROR MESSAGE ---",
            (error_msg or "").rstrip(),
        ]
        if tb and tb.strip() and tb.strip() != (error_msg or "").strip():
            lines += ["", "--- TRACEBACK ---", tb.rstrip()]
        lines += ["", "--- KERNEL CODE ---", (code or "").rstrip(), ""]
        entry = "\n".join(lines) + "\n"
        try:
            with open(self._files["composition_errors"], "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass

    # --- acquisition function failure ---
    def log_acqf_error(
        self,
        bo_iter: int,
        n_evals: int,
        acquisition: str,
        error_type: str,
        error_msg: str,
        tb: Optional[str] = None,
        best_kernel_source: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        lines = [
            sep,
            f"[{timestamp}]  bo_iter={bo_iter}  n_evals={n_evals}  acqf={acquisition}",
            f"ERROR TYPE: {error_type}",
            f"KERNEL: {best_kernel_source or 'unknown'}",
            "",
            "--- ERROR MESSAGE ---",
            (error_msg or "").rstrip(),
        ]
        if tb and tb.strip() and tb.strip() != (error_msg or "").strip():
            lines += ["", "--- TRACEBACK ---", tb.rstrip()]
        lines.append("")
        entry = "\n".join(lines) + "\n"
        try:
            with open(self._files["acqf_errors"], "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass

    # --- BO iteration summary ---
    def log_bo_iteration(
        self,
        bo_iter: int,
        n_evals: int,
        best_so_far: float,
        elapsed_sec: float,
        kernel_code: Optional[str],
        acqf_failed: bool,
        extra: Optional[dict] = None,
    ):
        rec = {
            "bo_iter": bo_iter,
            "n_evals": n_evals,
            "best_so_far": best_so_far,
            "elapsed_sec": elapsed_sec,
            "acqf_failed": acqf_failed,
            # "kernel_code": kernel_code,
        }
        if extra:
            rec.update(extra)
        self._append("bo_iterations", rec)

    # --- per-iteration timing breakdown ---
    def log_timing(
        self,
        bo_iter: int,
        kernel_evol_sec: float,
        acqf_opt_sec: float,
        oracle_sec: float,
        gpu_cleanup_sec: float,
        total_iter_sec: float,
        extra: Optional[dict] = None,
    ):
        """
        Log wall-clock timing breakdown for one BO round.

        kernel_evol_sec  : entire evolve_kernel_for_bo() call
          |- t_base_eval_sec        : base kernel eval wall time
          |- t_cumulative_eval_sec  : cumulative kernel eval wall time
          |- t_llm_mutation_sec     : LLM API wall time for mutation calls (parallel)
          |- t_mutation_eval_sec    : mutation kernel eval wall time
          |- t_llm_composition_sec  : LLM API wall time for composition calls
          |- t_composition_eval_sec : composition kernel eval wall time
          |- t_surrogate_refit_sec  : final best-kernel refit
          |- eval_breakdown         : per-stage breakdown (nested dict)
               |- base/cumulative/mutation/composition
                    |- validation_sec : agnostic+grad+psd checks
                    |- fit_sec        : GP model init + MLE fitting
                    |- scoring_sec    : model selection score (LOOCV etc.)
        acqf_opt_sec     : acquisition function build + optimize
        oracle_sec       : benchmark f(x_next) call
        gpu_cleanup_sec  : _clear_gpu_memory() calls combined
        total_iter_sec   : full BO round wall time
        """
        rec = {
            "bo_iter": bo_iter,
            "total_iter_sec": round(total_iter_sec, 4),
            "kernel_evol_sec": round(kernel_evol_sec, 4),
            "acqf_opt_sec": round(acqf_opt_sec, 4),
            "oracle_sec": round(oracle_sec, 4),
            "gpu_cleanup_sec": round(gpu_cleanup_sec, 4),
        }
        if extra:
            rec.update({k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()})
        self._append("timing", rec)

    # --- per-generation evolution summary ---
    def log_evolution_summary(
        self,
        bo_iter: int,
        generation: int,
        n_candidates: int,
        n_success: int,
        n_failed: int,
        best_score: float,
        population_size: int,
        extra: Optional[dict] = None,
    ):
        rec = {
            "bo_iter": bo_iter,
            "generation": generation,
            "n_candidates": n_candidates,
            "n_success": n_success,
            "n_failed": n_failed,
            "success_rate": n_success / max(n_candidates, 1),
            "best_score": best_score,
            "population_size": population_size,
        }
        if extra:
            rec.update(extra)
        self._append("evolution_summary", rec)

    # --- per-iteration population snapshot (human-readable) ---
    def log_population_snapshot(
        self,
        bo_iter: int,
        n_evals: int,
        best_so_far: float,
        elapsed_sec: float,
        population: list,
        selected_source: Optional[str] = None,
        model_selection: str = "loss",
    ):
        def _kernel_type(source: str) -> str:
            # strip optional bo{N}_ prefix
            s = source.split("_", 1)[1] if source.startswith("bo") and "_" in source and source[2:].split("_")[0].isdigit() else source
            if s.startswith("base_"):
                return "base"
            if "_comp_fallback" in s:
                return "comp_fallback"
            if s.startswith("composition_"):
                return "composition"
            if s.startswith("mutation_"):
                return "mutation(retry)" if "_retry" in s else "mutation"
            return "carried"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 80
        header = (
            f"BO Iter {bo_iter:>3}  |  n_evals={n_evals}  |  "
            f"best={best_so_far:.6f}  |  elapsed={elapsed_sec:.0f}s  |  {timestamp}"
        )
        lines = [sep, header, sep]

        if selected_source:
            lines.append(f"  Selected kernel: {selected_source}")
            lines.append("")

        col_w_source = max((len(p.get("source", "")) for p in population), default=20)
        col_w_source = max(col_w_source, 20)
        col_w_type = 14
        score_label = model_selection.upper()
        show_nll = model_selection != "loss" and any("nll" in p for p in population)
        # Show CRPS only when loss label is something other than CRPS
        show_crps_col = any("crps" in p for p in population) and score_label != "CRPS"
        show_ybest_col = any("batch_y_best" in p for p in population)
        show_extra = show_crps_col or show_ybest_col

        def _fmt_opt(v, w=12, prec=6):
            if v is None:
                return "N/A".ljust(w)
            if isinstance(v, float) and (v != v or v == float("inf") or v == -float("inf")):
                return "inf".ljust(w)
            try:
                return f"{float(v):.{prec}f}".ljust(w)
            except Exception:
                return str(v).ljust(w)

        # Build header/fmt string dynamically from the columns we actually want.
        cols = ["rank", "loss"]
        headers = {"rank": "Rank", "loss": score_label}
        widths = {"rank": 4, "loss": 12}
        if show_nll:
            cols.append("nll"); headers["nll"] = "NLL"; widths["nll"] = 12
        if show_crps_col:
            cols.append("crps"); headers["crps"] = "CRPS"; widths["crps"] = 12
        if show_ybest_col:
            cols.append("y_best"); headers["y_best"] = "batch_y"; widths["y_best"] = 12
        cols += ["source", "ktype"]
        widths.update({"source": col_w_source, "ktype": col_w_type})
        headers.update({"source": "Source", "ktype": "Type"})
        fmt_parts = []
        for c in cols:
            if c == "rank":
                fmt_parts.append(f"{{rank:>{widths['rank']}}}")
            else:
                fmt_parts.append(f"{{{c}:<{widths[c]}}}")
        fmt = "  " + "  ".join(fmt_parts)
        lines.append(fmt.format(**headers))
        sep_len = sum(widths[c] for c in cols) + 2 * (len(cols) - 1)
        lines.append("  " + "-" * sep_len)

        for rank, p in enumerate(population, start=1):
            source = p.get("source", "unknown")
            ktype = _kernel_type(source)
            marker = " <-- selected" if source == selected_source else ""
            row = {"rank": rank, "loss": f"{p['loss']:.6f}", "source": source, "ktype": ktype}
            if show_nll:
                nll_val = p.get("nll", float("inf"))
                row["nll"] = f"{nll_val:.6f}" if nll_val < float("inf") else "inf"
            if show_crps_col:
                row["crps"] = _fmt_opt(p.get("crps"))
            if show_ybest_col:
                row["y_best"] = _fmt_opt(p.get("batch_y_best"), prec=4)
            lines.append(fmt.format(**row) + marker)

        lines.append("")
        entry = "\n".join(lines) + "\n"
        try:
            with open(self._files["population_log"], "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass
