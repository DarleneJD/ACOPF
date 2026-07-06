# -*- coding: utf-8 -*-
"""Runtime patches for the ACOPF notebook workflow.

Apply after loading `opf_core` and `02_validate_policy_abc_opf.py`:

    import opf_runtime_patches as patches
    patches.apply(core, step2)

Purpose:
  1) tighten Pyomo voltage upper bounds for OpenDSS robustness;
  2) prevent PF=0.85 capacitive export for inverters near high voltage;
  3) export monophase PVSystems with kV=0.277 instead of 0.480;
  4) validate exported policy without relying on unexported curtailment;
  5) export a final DSS only if POLICY_LOCAL_TAP is approved.
"""
from __future__ import annotations

import math
from pathlib import Path

import pyomo.environ as pyo

V_MODEL_HI = 1.045
V_A_SAFE_MAX = 1.045


def _fix_no_curtailment(model):
    """Fix Pcurt=0 because the exported DSS does not include Volt-Watt/Pcurt."""
    if hasattr(model, "Pcurt"):
        for idx in model.Pcurt:
            model.Pcurt[idx].fix(0.0)


def _patch_build_opf(core):
    original = core.build_opf
    if getattr(original, "_runtime_patched", False):
        return

    def build_opf_patched(data, hours=None):
        if hours is None:
            hours = core.HOURS
        model, sets_info, i2_ws = original(data, hours)
        slack = data.get("slack")
        for (bus, ph) in sets_info.get("bph", []):
            if bus == slack:
                continue
            for t in sets_info.get("hours", []):
                var = model.V[bus, ph, t]
                ub = var.ub
                if ub is None or ub > V_MODEL_HI:
                    var.setub(V_MODEL_HI)
        return model, sets_info, i2_ws

    build_opf_patched._runtime_patched = True
    core.build_opf = build_opf_patched


def _patch_classification(core):
    original = core.classify_inverters_from_opf
    if getattr(original, "_runtime_patched", False):
        return

    def classify_patched(model, data, sets_info):
        groups, inferred_b, inv_data = original(model, data, sets_info)

        # PF capacitivo fixo só é seguro em barras/fases sem tensão alta.
        for key, d in list(inv_data.items()):
            if d.get("classification") == "A" and float(d["V_arr"].max()) > V_A_SAFE_MAX:
                if key in groups.get("A", []):
                    groups["A"].remove(key)
                if key not in groups.get("B", []):
                    groups.setdefault("B", []).append(key)
                d["classification"] = "B"

        # Recalcula as curvas do grupo B após eventual migração A -> B.
        inferred_b = []
        try:
            import numpy as np
            for (bus, ph) in groups.get("B", []):
                d = inv_data[(bus, ph)]
                v_arr = d["V_arr"]
                q_arr = d["Q_arr"]
                q_max_obs = float(np.percentile(np.abs(q_arr), 98))
                v1, v2, v3, v4 = core._fit_vv_curve(v_arr, q_arr, q_max_obs)
                q_pred = np.array([
                    core._vv_curve_eval(v, v1, v2, v3, v4, q_max_obs)
                    for v in v_arr
                ])
                rmse = float(np.sqrt(np.mean((q_arr - q_pred) ** 2)))
                rmse_pct = rmse / max(q_max_obs, 1e-6) * 100.0
                inferred_b.append({
                    "bus": bus,
                    "ph": ph,
                    "V1": v1,
                    "V2": v2,
                    "V3": v3,
                    "V4": v4,
                    "Q_max": q_max_obs / core.SBASE,
                    "rmse_pct": rmse_pct,
                })
        except Exception as exc:
            print(f"  [patch] aviso: não consegui recalcular curvas B ({type(exc).__name__}: {exc})")

        return groups, inferred_b, inv_data

    classify_patched._runtime_patched = True
    core.classify_inverters_from_opf = classify_patched


def _patch_export_pvsystems(core):
    original = core.export_pvsystems_to_dss
    if getattr(original, "_runtime_patched", False):
        return

    def export_patched(*args, **kwargs):
        out_path = kwargs.get("out_path", "pvsystems_opf.dss")
        result = original(*args, **kwargs)

        path = Path(out_path)
        if not path.exists():
            return result

        lines = path.read_text(encoding="utf-8").splitlines()
        patched = []
        last_pv_is_1ph = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("New PVSystem."):
                last_pv_is_1ph = "phases=1" in stripped
                if last_pv_is_1ph:
                    line = line.replace("kV=0.48", "kV=0.277").replace("kV=0.480", "kV=0.277")
                    line = line.replace(" kV={kv}", " kV=0.277")
            elif last_pv_is_1ph and stripped.startswith("~") and "kV=" in stripped:
                line = line.replace("kV=0.48", "kV=0.277").replace("kV=0.480", "kV=0.277")
            elif stripped and not stripped.startswith("~"):
                last_pv_is_1ph = False
            patched.append(line)

        path.write_text("\n".join(patched) + "\n", encoding="utf-8")
        print("  [patch] kV dos PVSystems monofásicos ajustado para 0.277 kV")
        return result

    export_patched._runtime_patched = True
    core.export_pvsystems_to_dss = export_patched


def _patch_validation(step2):
    # Caso tap fixo: não permitir que a validação use Pcurt que não será exportado.
    original_fixed = step2.case_policy_fixed_tap
    if not getattr(original_fixed, "_runtime_patched", False):
        def fixed_patched(data, sets_info, model, policy, tap_schedule, solver_getter,
                          timelimit=300, tee=False, diagnose_module=None, out_dir=None):
            _fix_no_curtailment(model)
            return original_fixed(data, sets_info, model, policy, tap_schedule, solver_getter,
                                  timelimit=timelimit, tee=tee,
                                  diagnose_module=diagnose_module, out_dir=out_dir)
        fixed_patched._runtime_patched = True
        step2.case_policy_fixed_tap = fixed_patched

    # Caso tap ótimo: idem.
    original_optimal = step2.case_policy_optimal_tap
    if not getattr(original_optimal, "_runtime_patched", False):
        def optimal_patched(data, sets_info, model, policy, solver_getter,
                            timelimit=300, tee=False, diagnose_module=None, out_dir=None):
            _fix_no_curtailment(model)
            return original_optimal(data, sets_info, model, policy, solver_getter,
                                    timelimit=timelimit, tee=tee,
                                    diagnose_module=diagnose_module, out_dir=out_dir)
        optimal_patched._runtime_patched = True
        step2.case_policy_optimal_tap = optimal_patched

    # Export final só com POLICY_LOCAL_TAP aprovado, pois o OpenDSS roda RegControl ativo.
    original_export = step2.export_validated_dss
    if not getattr(original_export, "_runtime_patched", False):
        def export_patched(data, policy, sets_info, results_by_case, out_dir):
            ok_local = results_by_case.get("POLICY_LOCAL_TAP", {}).get("aprovado") is True
            if not ok_local:
                print("\n  [EXPORT] REPROVADO — POLICY_LOCAL_TAP não passou. ")
                print("  .dss NÃO gerado para uso com RegControl ativo no OpenDSS.")
                return None
            return original_export(data, policy, sets_info, results_by_case, out_dir)
        export_patched._runtime_patched = True
        step2.export_validated_dss = export_patched

    step2.fix_no_curtailment = _fix_no_curtailment


def apply(core, step2=None):
    """Apply runtime patches to already-loaded modules."""
    core.V_MODEL_HI = V_MODEL_HI
    _patch_build_opf(core)
    _patch_classification(core)
    _patch_export_pvsystems(core)

    if step2 is not None:
        _patch_validation(step2)

    print("OK: patches OpenDSS-consistentes aplicados.")
    print(f"  V_MODEL_HI={V_MODEL_HI}; V_A_SAFE_MAX={V_A_SAFE_MAX}; PV 1ph kV=0.277")
