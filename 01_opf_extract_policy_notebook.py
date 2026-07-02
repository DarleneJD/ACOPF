# -*- coding: utf-8 -*-
"""
01_opf_extract_policy.py
=============================================================================
CÓDIGO 1 — OPF COORDENADO LIVRE + EXTRAÇÃO/CLASSIFICAÇÃO DA POLÍTICA A/B/C
=============================================================================
Responsabilidade ÚNICA: resolver o OPF livre (Qpv contínuo, tap discretizado
em 2 estágios) e traduzir o despacho ótimo numa política candidata (grupos
A/B/C + curvas Volt-VAR). NÃO valida a política, NÃO exporta .dss definitivo.

Reaproveita opf_core.py (load_data, build_opf, classify_inverters_from_opf,
_fit_vv_curve, export_pvsystems_to_dss) — funções já testadas e verificadas
contra os dados reais neste projeto. Não duplica essa lógica aqui: duplicar
seria reintroduzir o risco dos bugs de coluna/classificação que já corrigimos
no núcleo (aba Reg, aba Trafos, is3ph, dupla implementação de curva Volt-VAR).

SAÍDAS (ver seção OUTPUTS no final):
  opf_solution_timeseries.csv, tap_schedule_opf.csv, opf_diagnostics.json,
  inversores_classificacao.csv, curvas_voltvar_individuais.csv,
  curva_voltvar_agregada.csv, policy_abc.json,
  pvsystems_DRAFT_nao_validado.dss  (rascunho, ver aviso na função de export)
=============================================================================
"""
import os
import sys
import csv
import json
import time
import math
import datetime as _dt

import pyomo.environ as pyo

import opf_core as core


# =============================================================================
# CONFIGURAÇÃO DO SOLVER (mesma lógica de dois estágios do arquivo original)
# =============================================================================
CPLEX_EXE = os.environ.get('CPLEX_EXE', 'cplex')
TIMELIMIT_STAGE1 = int(os.environ.get('TIMELIMIT_STAGE1', 600))
TIMELIMIT_STAGE2 = int(os.environ.get('TIMELIMIT_STAGE2', 600))
MIPGAP = float(os.environ.get('MIPGAP', 0.01))


def _get_solver(prefer='cplex'):
    """Tenta CPLEX; cai para CBC como fallback (CBC não resolve as
    restrições cônicas SOCP — serve só para diagnosticar disponibilidade,
    não para o solve real de produção)."""
    try:
        opt = pyo.SolverFactory(prefer, executable=CPLEX_EXE)
        if opt.available():
            return opt, prefer
    except Exception:
        pass
    print("  [AVISO] CPLEX indisponível neste ambiente — tentando CBC "
          "(não resolve as restrições cônicas SOCP; use apenas para "
          "smoke-test da construção do modelo, não para o resultado final).")
    return pyo.SolverFactory('cbc'), 'cbc'


# =============================================================================
# ETAPA 2 — resolver o OPF em dois estágios (idêntico em espírito ao
# arquivo original: Estágio 1 relaxa tap_pos/u_tap a contínuo; Estágio 2
# fixa o tap arredondado e resolve o QP puro com Qpv ainda livre)
# =============================================================================
def _solve_two_stage(model, sets_info, timelimit1=TIMELIMIT_STAGE1,
                     timelimit2=TIMELIMIT_STAGE2, tee=False):
    hours = sets_info['hours']
    svr_phs = sets_info['svr_phs']

    # ── Estágio 1: relaxar tap_pos/u_tap para contínuo ──────────────────────
    print("\n[Estágio 1] Relaxando tap_pos/u_tap para contínuos (QP)...")
    for ph in svr_phs:
        for t in hours:
            model.tap_pos[ph, t].domain = pyo.NonNegativeReals
            model.u_tap[ph, t].domain = pyo.NonNegativeReals
            model.u_tap[ph, t].setub(1)

    opt1, which1 = _get_solver()
    opts1 = {}
    if which1 == 'cplex':
        opts1 = {'timelimit': timelimit1, 'mipgap': MIPGAP}
    t0 = time.time()
    res1 = opt1.solve(model, tee=tee, options=opts1) if opts1 else \
        opt1.solve(model, tee=tee)
    tc1 = str(res1.solver.termination_condition)
    print(f"  Estágio 1: {tc1} | {time.time()-t0:.1f}s")

    # Arredonda e fixa o tap por fase/período para o Estágio 2
    tap_fixed = {}
    for ph in svr_phs:
        for t in hours:
            try:
                pos = int(round(pyo.value(model.tap_pos[ph, t])))
            except Exception:
                pos = 0
            tap_fixed[(ph, t)] = max(0, min(sets_info['n_pos'] - 1, pos))

    # ── Estágio 2: fixar tap, resolver QP puro (Qpv continua livre) ────────
    print("\n[Estágio 2] Fixando tap arredondado, Qpv permanece LIVRE (QP)...")
    for ph in svr_phs:
        for t in hours:
            model.tap_pos[ph, t].domain = pyo.NonNegativeReals
            model.tap_pos[ph, t].fix(tap_fixed[(ph, t)])
            model.u_tap[ph, t].fix(0)  # contabilizado via mudança de posição

    opt2, which2 = _get_solver()
    opts2 = {}
    if which2 == 'cplex':
        opts2 = {'timelimit': timelimit2, 'mipgap': MIPGAP}
    t0 = time.time()
    res2 = opt2.solve(model, tee=tee, options=opts2) if opts2 else \
        opt2.solve(model, tee=tee)
    tc2 = str(res2.solver.termination_condition)
    print(f"  Estágio 2: {tc2} | {time.time()-t0:.1f}s")

    return {'stage1_termination': tc1, 'stage2_termination': tc2,
            'tap_fixed': tap_fixed, 'solver_stage1': which1,
            'solver_stage2': which2}


# =============================================================================
# ETAPA 3 — exportar a série temporal completa da solução ótima livre
# =============================================================================
def export_solution_timeseries(model, data, sets_info, out_dir):
    hours = sets_info['hours']
    svr = data['svr']
    path = os.path.join(out_dir, 'opf_solution_timeseries.csv')
    n_rows = 0
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'bus', 'phase', 'V_pu', 'Pavail_pu', 'Pdisp_pu',
                   'Pcurt_pu', 'Qpv_opf_pu', 'Qpv_opf_kvar', 'tap_pos'])
        for (b, ph) in sets_info['pvph']:
            for t in hours:
                try:
                    V = core.try_v(model, b, ph, t)
                    Pav = float(pyo.value(model.Pavail[b, ph, t]))
                    Pcurt = float(pyo.value(model.Pcurt[b, ph, t]))
                    Pdisp = Pav - Pcurt
                    Qpv_pu = float(pyo.value(model.Qpv[b, ph, t]))
                except Exception:
                    continue
                tap = ''
                if ph in svr:
                    try:
                        tap = int(round(pyo.value(model.tap_pos[ph, t])))
                    except Exception:
                        tap = ''
                w.writerow([t, b, ph, f"{V:.6f}", f"{Pav:.6f}",
                           f"{Pdisp:.6f}", f"{Pcurt:.6f}", f"{Qpv_pu:.6f}",
                           f"{Qpv_pu*core.SBASE:.4f}", tap])
                n_rows += 1
    print(f"  [CSV] {path} ({n_rows} linhas)")
    return path


def export_tap_schedule(model, sets_info, solve_meta, out_dir):
    hours = sets_info['hours']
    svr_phs = sets_info['svr_phs']
    path = os.path.join(out_dir, 'tap_schedule_opf.csv')
    ops_by_phase = {ph: 0 for ph in svr_phs}
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'phase', 'tap_pos', 'tap_pu', 'is_operation'])
        prev = {}
        for ph in svr_phs:
            for t in hours:
                pos = solve_meta['tap_fixed'][(ph, t)]
                tap_pu = sets_info['tap_min'] + pos * sets_info['tap_step']
                is_op = int(prev.get(ph) is not None and prev[ph] != pos)
                if is_op:
                    ops_by_phase[ph] += 1
                prev[ph] = pos
                w.writerow([t, ph, pos, f"{tap_pu:.5f}", is_op])
    total_ops = sum(ops_by_phase.values())
    print(f"  [CSV] {path} (total_ops={total_ops}, por fase={ops_by_phase})")
    return path, ops_by_phase, total_ops


# =============================================================================
# ETAPA 4/5 — classificar e exportar curvas (reaproveita core.classify_...)
# =============================================================================
def export_classification_and_curves(groups, inferred_B, inv_data, data,
                                     out_dir):
    pv_meta = data.get('pv_meta', {})
    recommend = {'A': 'FP_fixo_capacitivo_0.85',
                'B': 'Volt-VAR_IEEE1547_curva_inferida', 'C': 'FP_unitario_1.00'}

    inferred_lookup = {(c['bus'], c['ph']): c for c in inferred_B}

    # inversores_classificacao.csv — um por (bus,ph) classificado
    p1 = os.path.join(out_dir, 'inversores_classificacao.csv')
    n = 0
    with open(p1, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bus', 'phase', 'grupo', 'recomendacao',
                   'V_min', 'V_max', 'V_med', 'Q_min_kVAr', 'Q_max_kVAr',
                   'Q_med_kVAr', 'P_disp_max_kW', 'Q_max_teorico_kVAr',
                   'V1', 'V2', 'V3', 'V4'])
        for (b, ph), d in inv_data.items():
            inferred = inferred_lookup.get((b, ph), {})
            w.writerow([
                b, ph, d['classification'],
                recommend.get(d['classification'], '-'),
                f"{float(d['V_arr'].min()):.5f}",
                f"{float(d['V_arr'].max()):.5f}",
                f"{float(d['V_arr'].mean()):.5f}",
                f"{float(d['Q_arr'].min()):.3f}",
                f"{float(d['Q_arr'].max()):.3f}",
                f"{float(d['Q_arr'].mean()):.3f}",
                f"{float(d['P_arr'].max()):.3f}",
                f"{d['Q_max_inv']:.3f}",
                f"{inferred.get('V1',''):.4f}" if inferred else '',
                f"{inferred.get('V2',''):.4f}" if inferred else '',
                f"{inferred.get('V3',''):.4f}" if inferred else '',
                f"{inferred.get('V4',''):.4f}" if inferred else '',
            ])
            n += 1
    print(f"  [CSV] {p1} ({n} inversores)")

    # curvas_voltvar_individuais.csv — só o Grupo B, uma linha por curva
    p2 = os.path.join(out_dir, 'curvas_voltvar_individuais.csv')
    with open(p2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bus', 'phase', 'V1', 'V2', 'V3', 'V4',
                   'Q_max_kVAr', 'rmse_pct'])
        for c in inferred_B:
            w.writerow([c['bus'], c['ph'], f"{c['V1']:.4f}", f"{c['V2']:.4f}",
                       f"{c['V3']:.4f}", f"{c['V4']:.4f}",
                       f"{c['Q_max']*core.SBASE:.3f}", f"{c['rmse_pct']:.1f}"])
    print(f"  [CSV] {p2} ({len(inferred_B)} curvas individuais)")

    # curva_voltvar_agregada.csv — mediana por parâmetro
    p3 = os.path.join(out_dir, 'curva_voltvar_agregada.csv')
    if inferred_B:
        import statistics as st
        agg = {k: st.median(c[k] for c in inferred_B)
              for k in ('V1', 'V2', 'V3', 'V4')}
        agg['Q_max_kVAr'] = st.median(c['Q_max'] * core.SBASE for c in inferred_B)
        agg['rmse_pct_mediana'] = st.median(c['rmse_pct'] for c in inferred_B)
    else:
        agg = {'V1': None, 'V2': None, 'V3': None, 'V4': None,
              'Q_max_kVAr': None, 'rmse_pct_mediana': None}
    with open(p3, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(agg.keys()))
        w.writerow([f"{v:.4f}" if v is not None else '' for v in agg.values()])
    print(f"  [CSV] {p3}")

    return p1, p2, p3, agg


def export_policy_json(groups, agg_curve, inv_data, out_dir):
    """policy_abc.json — a política candidata em formato consumível pelo
    Código 2 (não é a política VALIDADA; ver aviso no cabeçalho do arquivo)."""
    path = os.path.join(out_dir, 'policy_abc.json')
    payload = {
        '_aviso': ('Política CANDIDATA extraída do OPF livre. NÃO validada. '
                  'Não use para implantação em campo nem para exportação '
                  'OpenDSS definitiva sem rodar o Código 2.'),
        'gerado_em': _dt.datetime.now().isoformat(timespec='seconds'),
        'grupos': {g: [{'bus': b, 'phase': ph} for (b, ph) in lst]
                  for g, lst in groups.items()},
        'contagem': {g: len(lst) for g, lst in groups.items()},
        'curva_agregada': agg_curve,
        'PF_A': 0.85,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] {path}")
    return path


def export_diagnostics_json(model, data, sets_info, solve_meta, ops_by_phase,
                            total_ops, groups, out_dir):
    path = os.path.join(out_dir, 'opf_diagnostics.json')
    try:
        fd = core.calc_fd_percent(model, data, sets_info)
    except Exception as e:
        fd = {'erro': str(e)}
    try:
        obj_val = float(pyo.value(model.obj))
    except Exception:
        obj_val = None
    payload = {
        'gerado_em': _dt.datetime.now().isoformat(timespec='seconds'),
        'estagio1_status': solve_meta['stage1_termination'],
        'estagio2_status': solve_meta['stage2_termination'],
        'solver_estagio1': solve_meta['solver_stage1'],
        'solver_estagio2': solve_meta['solver_stage2'],
        'F_O': obj_val,
        'tap_ops_por_fase_opf_livre': ops_by_phase,
        'tap_ops_total_opf_livre': total_ops,
        'classificacao': {g: len(v) for g, v in groups.items()},
        'FD_diagnostico': fd if isinstance(fd, dict) else str(fd),
        'pesos': {
            'W_DV': core.W_DV, 'W_UNBAL': core.W_UNBAL, 'W_TAP': getattr(core, 'W_TAP_OPS', None),
            'W_Q_USE': getattr(core, 'W_Q_USE', None),
            'W_CURT': core.W_CURT, 'W_CORE': core.W_CORE,
        },
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [JSON] {path}")
    return path


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def run_opf_and_extract_policy(buses_file, branches_file, out_dir='.',
                               timelimit1=TIMELIMIT_STAGE1,
                               timelimit2=TIMELIMIT_STAGE2, tee=False,
                               enable_phase_coupling=None):
    """Executa o pipeline completo do Código 1. Retorna um dict com todos
    os caminhos de arquivo gerados e os objetos em memória (model, groups,
    inferred_B, inv_data) — úteis para inspeção interativa ou testes.

    enable_phase_coupling: None mantém o default do opf_core (True); passe
    False para desligar o acoplamento entre fases MT — IMPORTANTE: use o
    MESMO valor no Código 1 e no Código 2, senão a política é extraída de
    uma física e validada contra outra."""
    if enable_phase_coupling is not None:
        core.ENABLE_PHASE_COUPLING = enable_phase_coupling
        print(f"  [config] ENABLE_PHASE_COUPLING = {core.ENABLE_PHASE_COUPLING}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  CÓDIGO 1 — OPF LIVRE + EXTRAÇÃO DE POLÍTICA CANDIDATA A/B/C")
    print("=" * 78)

    data = core.load_data(buses_file, branches_file)
    core.compute_warm_start(data, core.HOURS, core.IRRAD, core.LOAD_PROFILE)
    model, sets_info, I2_ws = core.build_opf(data, core.HOURS)

    solve_meta = _solve_two_stage(model, sets_info, timelimit1, timelimit2, tee)

    if solve_meta['stage2_termination'] not in ('optimal', 'globallyOptimal',
                                                'locallyOptimal'):
        print(f"\n  [AVISO] Estágio 2 terminou como "
              f"'{solve_meta['stage2_termination']}' — a solução pode não "
              f"ser confiável. Prosseguindo mesmo assim para permitir "
              f"inspeção, mas NÃO trate isso como política candidata válida.")

    print("\n  Exportando série temporal e cronograma de tap...")
    ts_path = export_solution_timeseries(model, data, sets_info, out_dir)
    tap_path, ops_by_phase, total_ops = export_tap_schedule(
        model, sets_info, solve_meta, out_dir)

    print("\n  Classificando inversores a partir do despacho ótimo livre...")
    groups, inferred_B, inv_data = core.classify_inverters_from_opf(
        model, data, sets_info)
    print(f"    A={len(groups['A'])} B={len(groups['B'])} C={len(groups['C'])}")

    p1, p2, p3, agg_curve = export_classification_and_curves(
        groups, inferred_B, inv_data, data, out_dir)
    policy_path = export_policy_json(groups, agg_curve, inv_data, out_dir)
    diag_path = export_diagnostics_json(
        model, data, sets_info, solve_meta, ops_by_phase, total_ops,
        groups, out_dir)

    # Rascunho .dss — EXPLICITAMENTE marcado como não-validado (item 6 do
    # Código 1: "pode exportar apenas um rascunho, deixando claro que a
    # política ainda não está validada").
    draft_path = os.path.join(out_dir, 'pvsystems_DRAFT_nao_validado.dss')
    try:
        core.export_pvsystems_to_dss(data, groups, inv_data, inferred_B,
                                     out_path=draft_path)
        with open(draft_path) as f:
            body = f.read()
        with open(draft_path, 'w') as f:
            f.write(
                "! ============================================================\n"
                "! RASCUNHO — NAO VALIDADO. NAO IMPLANTAR.\n"
                "! Gerado direto do OPF livre (Codigo 1), sem passar pela\n"
                "! validacao autoconsistente Qpv=f(V) do Codigo 2. Pode conter\n"
                "! ajustes que violam tensao quando avaliados de forma\n"
                "! fechada (V->Q->V). Rode 02_validate_policy_abc_opf.py antes\n"
                "! de usar este tipo de arquivo como proposta final.\n"
                "! ============================================================\n\n"
                + body)
        print(f"  [DSS-RASCUNHO] {draft_path} (NÃO VALIDADO — ver cabeçalho)")
    except Exception as e:
        print(f"  [AVISO] rascunho .dss não gerado: {e}")
        draft_path = None

    print("\n" + "=" * 78)
    print(f"  CÓDIGO 1 CONCLUÍDO | tap_ops_livre={total_ops} | "
          f"A={len(groups['A'])} B={len(groups['B'])} C={len(groups['C'])}")
    print("  Próximo passo: 02_validate_policy_abc_opf.py")
    print("=" * 78)

    return {
        'model': model, 'data': data, 'sets_info': sets_info,
        'groups': groups, 'inferred_B': inferred_B, 'inv_data': inv_data,
        'solve_meta': solve_meta, 'tap_ops_total': total_ops,
        'ops_by_phase': ops_by_phase,
        'paths': {
            'opf_solution_timeseries': ts_path,
            'tap_schedule_opf': tap_path,
            'inversores_classificacao': p1,
            'curvas_voltvar_individuais': p2,
            'curva_voltvar_agregada': p3,
            'policy_abc': policy_path,
            'opf_diagnostics': diag_path,
            'draft_dss': draft_path,
        },
    }


if __name__ == '__main__':
    BUSES = os.environ.get('BUSES_FILE', 'buses_1.xlsx')
    BRANCHES = os.environ.get('BRANCHES_FILE', 'branches_1.xlsx')
    OUT = os.environ.get('OUT_DIR', '.')
    run_opf_and_extract_policy(BUSES, BRANCHES, OUT)
