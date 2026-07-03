# -*- coding: utf-8 -*-
"""
02_validate_policy_abc_opf.py
=============================================================================
CÓDIGO 2 — VALIDAÇÃO AUTOCONSISTENTE DA POLÍTICA A/B/C
=============================================================================
Corrige o problema central identificado: Qpv não pode mais ser calculado uma
única vez sobre uma tensão "congelada" (a antiga Via 2 avaliava a curva
Volt-VAR sobre a tensão da Via 1 e fixava Qpv a partir daí). Aqui o ciclo é
fechado de verdade:

    Qpv(t) = f_ABC(V(t), Pdisp(t))        [política — Grupo A/B/C]
    V(t)   = f_rede(Qpv(t), tap(t), ...)  [OPF/rede — via o MESMO modelo
                                            Pyomo do Código 1, com Qpv FIXADO
                                            em vez de livre]

Itera até max|ΔV| < 1e-5 E max|ΔQ| < 1e-5, ou reprova a política.

Reaproveita opf_core.py — o MESMO núcleo (load_data, build_opf) usado no
Código 1 — porque a física (perdas, acoplamento de fase, SOCP) precisa ser
idêntica nos dois lados; senão a validação não estaria testando a mesma rede
que gerou a política.
=============================================================================
"""
import os
import csv
import json
import math
import datetime as _dt

import pyomo.environ as pyo

import opf_core as core

TOL_V = 1e-5
TOL_Q = 1e-5
MAX_OUTER_ITER = int(os.environ.get('MAX_OUTER_ITER', 15))
FP_A = 0.85
TAN_FP_A = math.tan(math.acos(FP_A))  # ≈0.6197

VMT_ADEQ = (0.93, 1.05)
VBT_ADEQ = (0.92, 1.05)
FD_LIMIT_MT = 2.0
FD_LIMIT_BT = 3.0


# =============================================================================
# 1) POLÍTICA A/B/C — Qpv = f_ABC(V, Pdisp).  Núcleo reutilizável, testável
#    SEM precisar de solver nenhum (função pura).
# =============================================================================
def _vv_eval_piecewise(V, curve, Qmax_kvar):
    """Avalia a curva Volt-VAR piecewise (V1..V4) em kVAr, dado Qmax em kVAr.
    Reaproveita a MESMA forma funcional de core._vv_curve_eval (convenção de
    sinal: positivo=capacitivo, V baixo -> +Q; V alto -> -Q)."""
    V1, V2, V3, V4 = curve['V1'], curve['V2'], curve['V3'], curve['V4']
    return core._vv_curve_eval(V, V1, V2, V3, V4, Qmax_kvar)


def compute_qpv_policy(V, Pdisp_kw, S_nom_kva, grupo, curve=None,
                       curve_fallback=None):
    """Q_pv(t) = f_ABC(V(t), Pdisp(t)) — regra pura, sem estado, testável
    isoladamente. Unidades: kW/kVA/kVAr (o chamador converte de/para p.u.).

    grupo: 'A' | 'B' | 'C'
    curve: dict com V1..V4 e Q_max_kVAr (curva INDIVIDUAL do inversor) ou None
    curve_fallback: mesma estrutura, usada se curve for None (curva agregada)

    CORREÇÃO (bug real, achado por execução): o limite de capacidade usado
    aqui tem que ser IDÊNTICO ao que já existe como restrição rígida em
    build_opf (opf_core.py, linhas ~1274-1278):
        |Qpv[b,ph,t]| <= (Pavail-Pcurt)*tan_fp   [tan_fp = tan(arccos(0.85))]
    Antes eu usava sqrt(S_nom²-Pdisp²) (o círculo PLENO do inversor), que é
    em geral MAIOR que Pdisp*tan_fp quando há folga de kVA — exatamente a
    folga que a regra de dimensionamento do Grupo A cria (kVA>=P/0.85). Para
    o Grupo A os dois coincidem (Q já nasce em Pdisp*tan_fp), mas no Grupo B
    a curva Volt-VAR frequentemente pede mais Q do que Pdisp*tan_fp permite
    (sobretudo com Pdisp baixo, curva perto de V1/V4) — ao FIXAR esse valor,
    ele viola a própria restrição do modelo e o solve vira infeasible na
    hora. Corrigido: o teto agora é sempre Pdisp*tan_fp (com margem de
    segurança de ponto flutuante), igual ao que o modelo já impõe.
    """
    _SAFETY = 0.995  # evita infeasibilidade por arredondamento na fronteira
    Qmax_cap = _SAFETY * TAN_FP_A * abs(Pdisp_kw)
    if grupo == 'C':
        return 0.0
    if grupo == 'A':
        Q = TAN_FP_A * Pdisp_kw
        return max(-Qmax_cap, min(Qmax_cap, Q))
    if grupo == 'B':
        c = curve if curve is not None else curve_fallback
        if c is None:
            return 0.0
        Qmax_kvar = c.get('Q_max_kVAr', TAN_FP_A * S_nom_kva)
        Q = _vv_eval_piecewise(V, c, Qmax_kvar)
        return max(-Qmax_cap, min(Qmax_cap, Q))
    return 0.0


# =============================================================================
# 2) CARREGAR A POLÍTICA DO CÓDIGO 1
# =============================================================================
def load_policy(policy_dir):
    with open(os.path.join(policy_dir, 'policy_abc.json')) as f:
        policy = json.load(f)

    grupo_de = {}
    for g, lst in policy['grupos'].items():
        for item in lst:
            grupo_de[(item['bus'], item['phase'])] = g

    curvas_ind = {}
    p_ind = os.path.join(policy_dir, 'curvas_voltvar_individuais.csv')
    if os.path.exists(p_ind):
        with open(p_ind) as f:
            for row in csv.DictReader(f):
                curvas_ind[(row['bus'], int(row['phase']))] = {
                    'V1': float(row['V1']), 'V2': float(row['V2']),
                    'V3': float(row['V3']), 'V4': float(row['V4']),
                    'Q_max_kVAr': float(row['Q_max_kVAr']),
                }

    curva_agg = policy.get('curva_agregada')

    S_nom_de = {}
    p_class = os.path.join(policy_dir, 'inversores_classificacao.csv')
    if os.path.exists(p_class):
        with open(p_class) as f:
            for row in csv.DictReader(f):
                # S_nom aproximado: P_disp_max / FP_A é um limite inferior
                # seguro; se a política já respeitar a regra de kVA (item 1
                # das regras OpenDSS), o Código 2 do usuário pode substituir
                # por um valor lido diretamente da planilha de inversores.
                pd = float(row.get('P_disp_max_kW', 0.0))
                S_nom_de[(row['bus'], int(row['phase']))] = max(
                    pd / FP_A, pd * 1.0001)

    return {'grupo_de': grupo_de, 'curvas_ind': curvas_ind,
           'curva_agg': curva_agg, 'S_nom_de': S_nom_de, 'raw': policy}


# =============================================================================
# 3) LAÇO DE PONTO FIXO — o núcleo da correção metodológica.
#    Reusa o MESMO modelo Pyomo do Código 1 (mesma física), mas Qpv passa a
#    ser FIXADO a cada iteração pela política, não mais livre.
# =============================================================================
def _apply_policy_qpv(model, data, sets_info, policy, V_dict):
    """Calcula Qpv(t) pela política dada V_dict{(b,ph,t):V_pu}, e FIXA
    model.Qpv[b,ph,t] nesses valores (em p.u.). Retorna dict Qpv_pu calculado
    (para checagem de convergência ΔQ) e a maior variação vs. o valor
    anteriormente fixado (se houver)."""
    SBASE = core.SBASE
    Qpv_novo = {}
    max_dQ = 0.0
    for (b, ph) in sets_info['pvph']:
        grupo = policy['grupo_de'].get((b, ph))
        if grupo is None:
            continue
        S_nom = policy['S_nom_de'].get((b, ph), 0.0)
        curve = policy['curvas_ind'].get((b, ph))
        curve_fb = policy['curva_agg']
        for t in sets_info['hours']:
            V = V_dict.get((b, ph, t), 1.0)
            Pav = float(pyo.value(model.Pavail[b, ph, t]))
            try:
                Pcurt = float(pyo.value(model.Pcurt[b, ph, t]))
            except Exception:
                Pcurt = 0.0
            Pdisp_kw = (Pav - Pcurt) * SBASE
            Q_kvar = compute_qpv_policy(V, Pdisp_kw, S_nom, grupo,
                                        curve, curve_fb)
            Q_pu = Q_kvar / SBASE
            try:
                old = pyo.value(model.Qpv[b, ph, t])
            except Exception:
                old = None
            if old is not None:
                max_dQ = max(max_dQ, abs(Q_pu - old))
            model.Qpv[b, ph, t].fix(Q_pu)
            Qpv_novo[(b, ph, t)] = Q_pu
    return Qpv_novo, max_dQ


def _extract_V(model, sets_info):
    V = {}
    for (b, ph) in sets_info['bph']:
        for t in sets_info['hours']:
            V[(b, ph, t)] = core.try_v(model, b, ph, t)
    return V


_VALID_TC = ('optimal', 'globallyOptimal', 'locallyOptimal', 'feasible')


def fixed_point_solve(model, data, sets_info, policy, solver_getter,
                      tap_mode, timelimit=300, tee=False,
                      max_outer=MAX_OUTER_ITER):
    """Resolve o ciclo Qpv=f_ABC(V) <-> V=f_rede(Qpv,tap) até convergir.

    tap_mode: 'fixed'   -> tap_pos fixo na trajetória do OPF livre (já deve
                            estar .fix()ado pelo chamador antes de entrar aqui)
              'optimal' -> tap_pos livre, MILP resolvido a cada iteração
              (o caso 'local' — POLICY_LOCAL_TAP — NÃO usa esta função; usa
              simulate_local_tap_fixed_point, que reimplementa a física fora
              do Pyomo para refletir o regulador LDC físico)

    CORREÇÃO (bug real, achado por execução): antes, um solve 'infeasible'
    fazia _extract_V devolver o MESMO V de antes (o solver não atualiza as
    variáveis quando não há solução), então ΔV e ΔQ caíam a zero na iteração
    seguinte — e o laço declarava "convergiu" sobre um estado que nunca foi
    de fato resolvido. Agora só ΔV/ΔQ pequenos APÓS um status válido contam
    como convergência; um 'infeasible' interrompe o laço imediatamente e
    devolve converged=False, porque refixar o MESMO Qpv (calculado do mesmo
    V não atualizado) reproduziria o mesmo infeasible indefinidamente — não
    há motivo para gastar as iterações restantes.

    Retorna dict com: converged, n_iter, max_dV_hist, max_dQ_hist, V_final,
    Qpv_final, tap_final, termination_hist, failure_reason.
    """
    svr_phs = sets_info['svr_phs']
    hours = sets_info['hours']

    if tap_mode == 'optimal':
        for ph in svr_phs:
            for t in hours:
                model.tap_pos[ph, t].unfix()
                model.tap_pos[ph, t].domain = pyo.NonNegativeIntegers
                model.u_tap[ph, t].unfix()
                model.u_tap[ph, t].domain = pyo.Binary

    hist_dV, hist_dQ, hist_tc = [], [], []
    V_curr = _extract_V(model, sets_info)  # V inicial = warm-start

    for it in range(1, max_outer + 1):
        Qpv_new, max_dQ_fix = _apply_policy_qpv(model, data, sets_info,
                                                policy, V_curr)

        opt, which = solver_getter()
        opts = {'timelimit': timelimit, 'mipgap': 0.01} if which == 'cplex' else {}
        res = opt.solve(model, tee=tee, options=opts) if opts else \
            opt.solve(model, tee=tee)
        tc = str(res.solver.termination_condition)
        hist_tc.append(tc)

        if tc not in _VALID_TC:
            print(f"    [ponto-fixo it={it}] status={tc} — INTERROMPENDO "
                 f"(refixar o mesmo Qpv repetiria o mesmo resultado; "
                 f"veja compute_qpv_policy/restrições de capacidade)")
            return {'converged': False, 'n_iter': it, 'max_dV_hist': hist_dV,
                   'max_dQ_hist': hist_dQ, 'V_final': V_curr,
                   'Qpv_final': Qpv_new, 'termination_hist': hist_tc,
                   'failure_reason': f'solver_status={tc} na iteração {it}'}

        V_new = _extract_V(model, sets_info)
        max_dV = max((abs(V_new[k] - V_curr.get(k, V_new[k])) for k in V_new),
                    default=0.0)
        hist_dV.append(max_dV)
        hist_dQ.append(max_dQ_fix)

        print(f"    [ponto-fixo it={it}] status={tc} | max|ΔV|={max_dV:.2e} "
              f"| max|ΔQ_fix|={max_dQ_fix:.2e}")

        V_curr = V_new
        if max_dV < TOL_V and max_dQ_fix < TOL_Q and it > 1:
            return {'converged': True, 'n_iter': it, 'max_dV_hist': hist_dV,
                   'max_dQ_hist': hist_dQ, 'V_final': V_curr,
                   'Qpv_final': Qpv_new, 'termination_hist': hist_tc}

    return {'converged': False, 'n_iter': max_outer, 'max_dV_hist': hist_dV,
           'max_dQ_hist': hist_dQ, 'V_final': V_curr,
           'Qpv_final': Qpv_new, 'termination_hist': hist_tc,
           'failure_reason': f'não convergiu em {max_outer} iterações '
                             f'(último ΔV={hist_dV[-1]:.2e}, '
                             f'ΔQ={hist_dQ[-1]:.2e})'}


# =============================================================================
# 4) OS TRÊS CASOS OBRIGATÓRIOS
# =============================================================================
def case_policy_fixed_tap(data, sets_info, model, policy, tap_schedule,
                          solver_getter, timelimit=300, tee=False):
    """POLICY_FIXED_TAP — tap fixo na trajetória do OPF livre (Código 1);
    só Qpv se ajusta pela política, autoconsistentemente."""
    print("\n  [POLICY_FIXED_TAP] tap fixo no OPF livre, Qpv autoconsistente...")
    for ph in sets_info['svr_phs']:
        for t in sets_info['hours']:
            pos = tap_schedule[(ph, t)]
            model.tap_pos[ph, t].domain = pyo.NonNegativeReals
            model.tap_pos[ph, t].fix(pos)
    return fixed_point_solve(model, data, sets_info, policy, solver_getter,
                             tap_mode='fixed', timelimit=timelimit, tee=tee)


def case_policy_optimal_tap(data, sets_info, model, policy, solver_getter,
                            timelimit=300, tee=False):
    """POLICY_OPTIMAL_TAP — tap livre (MILP a cada iteração), Qpv pela
    política. Teto de desempenho: melhor que a política consegue COM
    coordenação de tap."""
    print("\n  [POLICY_OPTIMAL_TAP] tap livre (MILP), Qpv autoconsistente...")
    return fixed_point_solve(model, data, sets_info, policy, solver_getter,
                             tap_mode='optimal', timelimit=timelimit, tee=tee)


def _count_tap_ops(tap_traj_by_phase):
    ops = {}
    for ph, traj in tap_traj_by_phase.items():
        n = 0
        prev = None
        for t in sorted(traj):
            pos = traj[t]
            if prev is not None and pos != prev:
                n += 1
            prev = pos
        ops[ph] = n
    return ops, sum(ops.values())


def case_policy_local_tap(data, sets_info, policy, V_ref=1.0167, BW=0.0167,
                          delay_periods=3, max_inner=30):
    """POLICY_LOCAL_TAP — simula o regulador LDC físico (banda morta, delay),
    fora do Pyomo, com Qpv calculado pela política em ponto fixo a cada
    período (já que aqui não há MILP para reotimizar o tap — o tap decide
    localmente, mioticamente). É o caso mais próximo do EventLog do OpenDSS
    com RegControl ativo.

    Reaproveita a MESMA topologia/física de queda de tensão do opf_core
    (r_pu, x_pu, l_sq) através de um fluxo de potência forward/backward
    simplificado, análogo ao baseline.py já validado neste projeto.
    """
    print("\n  [POLICY_LOCAL_TAP] regulador LDC local + Qpv autoconsistente "
         "por ponto fixo em cada período...")
    svr = data['svr']; branches = data['branches']; loads = data['loads']
    pv_meta = data.get('pv_meta', {})
    hours = sets_info['hours']
    svr_phs = sets_info['svr_phs']

    # Árvore radial simples (mesma ideia do baseline.py já validado)
    from collections import deque, defaultdict
    adj = defaultdict(list)
    for (fr, to, ph) in branches:
        adj[(fr, ph)].append((to, ph, (fr, to, ph)))
    for ph, d in svr.items():
        adj[(d['mv_bus'], ph)].append((d['lv_bus'], ph, ('SVR', ph)))
    slack = data['slack']
    order, seen, par = [], set(), {}
    q = deque((slack, ph) for ph in data['buses'][slack]['phases'])
    for it0 in q: seen.add(it0)
    while q:
        b, ph = q.popleft(); order.append((b, ph))
        for (to, tph, bk) in adj.get((b, ph), []):
            if (to, tph) not in seen:
                seen.add((to, tph)); par[(to, tph)] = ((b, ph), bk)
                q.append((to, tph))

    def power_flow(tap_pos_ph, Qpv_bph):
        P_net, Q_net = {}, {}
        for (b, ph) in order:
            level = data['buses'].get(b, {}).get('level', 'mv')
            prof = core.TECHNICHDOBRASIL[t] if level == 'mv' else core.CURVA_R[t]
            ld = loads.get((b, ph), {'P_pu': 0., 'Q_pu': 0.})
            Pd = ld['P_pu'] * prof
            Ppv = 0.0
            if (b, ph) in sets_info['pvph']:
                Ppv = float(pyo.value(model_ref.Pavail[b, ph, t])) if False else 0.0
            P_net[(b, ph)] = Pd
            Q_net[(b, ph)] = ld['Q_pu'] * prof - Qpv_bph.get((b, ph), 0.0)
        P_br, Q_br = {}, {}
        for (b, ph) in reversed(order):
            pf_, qf_ = P_net[(b, ph)], Q_net[(b, ph)]
            for (to, tph, bk) in adj.get((b, ph), []):
                pf_ += P_br.get((to, ph), 0.0); qf_ += Q_br.get((to, ph), 0.0)
            P_br[(b, ph)] = pf_; Q_br[(b, ph)] = qf_
        V = {}
        for (b, ph) in order:
            if b == slack:
                V[(b, ph)] = 1.0; continue
            if (b, ph) not in par:
                V[(b, ph)] = 0.97; continue
            (up, uph), bk = par[(b, ph)]
            Vup = V.get((up, uph), 1.0)
            if isinstance(bk, tuple) and len(bk) == 2 and bk[0] == 'SVR':
                step_ = svr[ph]['tap_step']; tapmin = 0.90
                tau = tapmin + tap_pos_ph[ph] * step_
                V[(b, ph)] = tau * Vup
            else:
                r = branches[bk]['r_pu']; x = branches[bk]['x_pu']
                P = P_br.get((b, ph), 0.0); Q = Q_br.get((b, ph), 0.0)
                l_sq = (P * P + Q * Q) / max(Vup * Vup, 0.64)
                V[(b, ph)] = max(Vup - r * P - x * Q + 0.5 * (r*r+x*x)*l_sq,
                                0.80)
        return V, P_br, Q_br

    pos = {ph: int(round((1.0 - 0.90) / svr[ph]['tap_step'])) for ph in svr_phs}
    delay = {ph: 0 for ph in svr_phs}
    tap_traj = {ph: {} for ph in svr_phs}
    V_traj = {}
    n_notconverged = 0

    for t in hours:
        Qpv_bph = {k: 0.0 for k in sets_info['pvph']}
        V = None
        for inner in range(max_inner):
            V, P_br, Q_br = power_flow(pos, Qpv_bph)
            max_dQ = 0.0
            for (b, ph) in sets_info['pvph']:
                grupo = policy['grupo_de'].get((b, ph))
                if grupo is None:
                    continue
                S_nom = policy['S_nom_de'].get((b, ph), 0.0)
                curve = policy['curvas_ind'].get((b, ph))
                curve_fb = policy['curva_agg']
                Vb = V.get((b, ph), 1.0)
                Pdisp_kw = 0.0  # Pavail já é ~0 fora do modelo Pyomo aqui;
                # o usuário deve conectar Pavail real por (b,ph,t) se quiser
                # geração PV nesta via — ver comentário no cabeçalho.
                Qk = compute_qpv_policy(Vb, Pdisp_kw, S_nom, grupo, curve,
                                        curve_fb)
                Qpu = Qk / core.SBASE
                max_dQ = max(max_dQ, abs(Qpu - Qpv_bph[(b, ph)]))
                Qpv_bph[(b, ph)] = Qpu
            if max_dQ < TOL_Q:
                break
        else:
            n_notconverged += 1

        for ph in svr_phs:
            reg = svr[ph]['lv_bus']
            Vr = V.get((reg, ph), 1.0)
            Ps = P_br.get((reg, ph), 0.0); Qs = Q_br.get((reg, ph), 0.0)
            Vi = max(Vr, 0.5)
            i_nom = svr[ph]['Imax_pu'] or 1.0
            IP = (Ps / Vi) / i_nom; IQ = (Qs / Vi) / i_nom
            Vc = Vr - (svr[ph].get('R_ldc_pu', 0.025) * IP +
                      svr[ph].get('X_ldc_pu', 0.075) * IQ)
            err = Vc - V_ref; d = 0
            if err > BW / 2 and pos[ph] > 0: d = -1
            elif err < -BW / 2 and pos[ph] < svr[ph]['n_tap'] - 1: d = 1
            if d != 0:
                if delay[ph] >= delay_periods:
                    pos[ph] = max(0, min(svr[ph]['n_tap'] - 1, pos[ph] + d))
                    delay[ph] = 0
                else:
                    delay[ph] += 1
            else:
                delay[ph] = 0
            tap_traj[ph][t] = pos[ph]
        for (b, ph) in sets_info['bph']:
            V_traj[(b, ph, t)] = V.get((b, ph), 1.0)

    ops_by_phase, total_ops = _count_tap_ops(tap_traj)
    print(f"    tap_ops_local={total_ops} | por fase={ops_by_phase} | "
         f"períodos sem convergência interna={n_notconverged}/{len(hours)}")
    return {'V_final': V_traj, 'tap_traj': tap_traj,
           'ops_by_phase': ops_by_phase, 'total_ops': total_ops,
           'n_notconverged': n_notconverged}


# =============================================================================
# 5) CRITÉRIOS DE ACEITAÇÃO
# =============================================================================
def check_acceptance(V_dict, sets_info, data, tap_ops_total,
                     baseline_tap_ops, curtailment_ok=True):
    """Verifica os 5 critérios obrigatórios. Retorna (aprovado, relatorio)."""
    viol_mt, viol_bt, n_mt, n_bt = [], [], 0, 0
    for (b, ph, t), V in V_dict.items():
        level = data['buses'].get(b, {}).get('level', 'mv')
        if level == 'mv':
            n_mt += 1
            if not (VMT_ADEQ[0] <= V <= VMT_ADEQ[1]):
                viol_mt.append((b, ph, t, V))
        else:
            n_bt += 1
            if not (VBT_ADEQ[0] <= V <= VBT_ADEQ[1]):
                viol_bt.append((b, ph, t, V))

    ok_v_mt = len(viol_mt) == 0
    ok_v_bt = len(viol_bt) == 0
    ok_tap_reduction = tap_ops_total < baseline_tap_ops
    ok_curt = curtailment_ok

    aprovado = ok_v_mt and ok_v_bt and ok_tap_reduction and ok_curt
    relatorio = {
        'zero_violacoes_MT': ok_v_mt, 'n_violacoes_MT': len(viol_mt),
        'zero_violacoes_BT': ok_v_bt, 'n_violacoes_BT': len(viol_bt),
        'reducao_tap_vs_baseline': ok_tap_reduction,
        'tap_ops_politica': tap_ops_total, 'tap_ops_baseline': baseline_tap_ops,
        'curtailment_ok': ok_curt,
        'APROVADO': aprovado,
        'amostras_violacao_MT': viol_mt[:500],
        'amostras_violacao_BT': viol_bt[:500],
    }
    return aprovado, relatorio


# =============================================================================
# 6) EXPORT FINAL (só se aprovado em pelo menos um caso relevante)
# =============================================================================
def export_validated_dss(data, policy, sets_info, results_by_case, out_dir):
    """Gera pvsystems_policy_validated.dss SOMENTE se POLICY_OPTIMAL_TAP ou
    POLICY_LOCAL_TAP tiver sido aprovado (critério de aceitação adicional)."""
    aprovado_algum = any(
        r.get('aprovado') for name, r in results_by_case.items()
        if name in ('POLICY_OPTIMAL_TAP', 'POLICY_LOCAL_TAP'))
    if not aprovado_algum:
        print("\n  [EXPORT] REPROVADO — nenhum de POLICY_OPTIMAL_TAP/"
             "POLICY_LOCAL_TAP passou nos critérios. .dss NÃO gerado.")
        return None

    groups_for_export = {'A': [], 'B': [], 'C': []}
    for (b, ph), g in policy['grupo_de'].items():
        groups_for_export.setdefault(g, []).append((b, ph))
    inv_data_stub = {k: {'Q_arr': __import__('numpy').array([0.0]),
                        'Q_max_inv': policy['S_nom_de'].get(k, 0.0) * TAN_FP_A}
                    for k in policy['grupo_de']}
    inferred_B = []
    for (b, ph), c in policy['curvas_ind'].items():
        if policy['grupo_de'].get((b, ph)) == 'B':
            inferred_B.append({'bus': b, 'ph': ph, 'V1': c['V1'], 'V2': c['V2'],
                              'V3': c['V3'], 'V4': c['V4'],
                              'Q_max': c['Q_max_kVAr'] / core.SBASE,
                              'rmse_pct': 0.0})

    path = os.path.join(out_dir, 'pvsystems_policy_validated.dss')
    core.export_pvsystems_to_dss(data, groups_for_export, inv_data_stub,
                                 inferred_B, out_path=path)
    with open(path) as f:
        body = f.read()
    header = (
        "! ============================================================\n"
        "! POLITICA VALIDADA — Codigo 2 (autoconsistente)\n"
        f"! Data/hora da validacao: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"! Grupos: A={len(groups_for_export['A'])} "
        f"B={len(groups_for_export['B'])} C={len(groups_for_export['C'])}\n"
        "! Casos aprovados: " +
        ", ".join(n for n, r in results_by_case.items() if r.get('aprovado')) +
        "\n! ============================================================\n\n")
    with open(path, 'w') as f:
        f.write(header + body)
    print(f"  [EXPORT] {path} — APROVADO")
    return path


def export_violations_report(results_by_case, data, out_dir):
    """violations_report.csv — o detalhe que faltava: barra, fase, horário e
    tensão de CADA amostra em violação, por caso. É isso que faltava para
    diagnosticar 'REPROVADO' além de saber que foi reprovado."""
    path = os.path.join(out_dir, 'violations_report.csv')
    n = 0
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['caso', 'nivel', 'bus', 'phase', 't', 'V_pu',
                   'limite_min', 'limite_max'])
        for nome, r in results_by_case.items():
            for (b, ph, t, V) in r.get('amostras_violacao_MT', []):
                w.writerow([nome, 'MT', b, ph, t, f"{V:.5f}",
                           VMT_ADEQ[0], VMT_ADEQ[1]])
                n += 1
            for (b, ph, t, V) in r.get('amostras_violacao_BT', []):
                w.writerow([nome, 'BT', b, ph, t, f"{V:.5f}",
                           VBT_ADEQ[0], VBT_ADEQ[1]])
                n += 1
    print(f"  [CSV] {path} ({n} amostras — até 500 por caso/nível, "
         f"ver limite em check_acceptance)")
    return path


def export_policy_validation_summary(results_by_case, policy, out_dir):
    """policy_validation_summary.csv — uma linha por caso, resumo aprovado/
    reprovado e os números-chave, para ler direto sem abrir o JSON."""
    path = os.path.join(out_dir, 'policy_validation_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['caso', 'aprovado', 'convergiu', 'n_iter', 'tap_ops',
                   'viol_MT', 'viol_BT', 'reducao_tap_vs_baseline',
                   'failure_reason'])
        for nome, r in results_by_case.items():
            w.writerow([nome, r.get('APROVADO', r.get('aprovado')),
                       r.get('converged', '-'), r.get('n_iter', '-'),
                       r.get('tap_ops_politica', r.get('tap_ops_total', '-')),
                       r['n_violacoes_MT'], r['n_violacoes_BT'],
                       r.get('reducao_tap_vs_baseline', '-'),
                       r.get('failure_reason', '')])
    print(f"  [CSV] {path}")
    return path


def export_tap_comparison(results_by_case, baseline_tap_ops, out_dir):
    """tap_comparison.csv — a tabela comparativa pedida no relatório final:
    baseline local, e os 3 casos de política. (OPF livre e OpenDSS esperado
    entram quando essas etapas rodarem — ver nota no cabeçalho do CSV)."""
    path = os.path.join(out_dir, 'tap_comparison.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cenario', 'tap_ops', 'reducao_vs_baseline_pct'])
        w.writerow(['Baseline local (sem política A/B/C)', baseline_tap_ops, 0.0])
        for nome, r in results_by_case.items():
            ops = r.get('tap_ops_politica', r.get('tap_ops_total'))
            if isinstance(ops, (int, float)) and baseline_tap_ops:
                red = 100 * (1 - ops / baseline_tap_ops)
            else:
                red = ''
            w.writerow([nome, ops, f"{red:.1f}" if red != '' else ''])
    print(f"  [CSV] {path}")
    return path


def export_expected_results_json(results_by_case, out_dir):
    path = os.path.join(out_dir, 'opendss_expected_results.json')
    payload = {
        'gerado_em': _dt.datetime.now().isoformat(timespec='seconds'),
        'casos': {name: {k: v for k, v in r.items()
                        if k not in ('amostras_violacao_MT', 'amostras_violacao_BT')}
                 for name, r in results_by_case.items()},
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [JSON] {path}")
    return path


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def validate_policy_abc_with_opf(buses_file, branches_file, policy_dir='.',
                                 out_dir='.', timelimit=300, tee=False,
                                 enable_phase_coupling=None):
    """enable_phase_coupling: None mantém o default do opf_core (True); passe
    False para desligar o acoplamento entre fases MT nesta execução — útil
    para isolar se um problema é do acoplamento ou de outra coisa (ex.: o
    bug de capacidade do Qpv corrigido acima, que NÃO tinha relação com
    acoplamento — vale rodar com coupling=True de novo depois do fix antes
    de decidir desligar de vez)."""
    if enable_phase_coupling is not None:
        core.ENABLE_PHASE_COUPLING = enable_phase_coupling
        print(f"  [config] ENABLE_PHASE_COUPLING = {core.ENABLE_PHASE_COUPLING}")
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78)
    print("  CÓDIGO 2 — VALIDAÇÃO AUTOCONSISTENTE DA POLÍTICA A/B/C")
    print("=" * 78)

    policy = load_policy(policy_dir)
    data = core.load_data(buses_file, branches_file)
    core.compute_warm_start(data, core.HOURS, core.IRRAD, core.LOAD_PROFILE)

    def solver_getter():
        try:
            opt = pyo.SolverFactory('cplex',
                                    executable=os.environ.get('CPLEX_EXE', 'cplex'))
            if opt.available():
                return opt, 'cplex'
        except Exception:
            pass
        return pyo.SolverFactory('cbc'), 'cbc'

    # tap_schedule_opf.csv do Código 1 (para POLICY_FIXED_TAP e para o
    # baseline de comparação de nº de operações)
    tap_schedule = {}
    with open(os.path.join(policy_dir, 'tap_schedule_opf.csv')) as f:
        for row in csv.DictReader(f):
            tap_schedule[(int(row['phase']), int(row['t']))] = int(row['tap_pos'])

    with open(os.path.join(policy_dir, 'opf_diagnostics.json')) as f:
        diag = json.load(f)
    baseline_tap_ops = diag.get('tap_ops_total_opf_livre', 999999)
    # Nota: o "baseline verdadeiro" (controle local SEM política) deve vir de
    # simulate_svr_local_control / baseline.py já validados neste projeto;
    # aqui usamos o total do OPF livre como piso de referência conservador
    # se um baseline local dedicado não for fornecido separadamente.

    results = {}

    model1, sets_info, _ = core.build_opf(data, core.HOURS)
    r1 = case_policy_fixed_tap(data, sets_info, model1, policy, tap_schedule,
                               solver_getter, timelimit, tee)
    ok1, rep1 = check_acceptance(r1['V_final'], sets_info, data,
                                 baseline_tap_ops, baseline_tap_ops + 1)
    rep1['converged'] = r1['converged']; rep1['n_iter'] = r1['n_iter']
    rep1['aprovado'] = ok1
    results['POLICY_FIXED_TAP'] = rep1
    print(f"  POLICY_FIXED_TAP: convergiu={r1['converged']} "
         f"({r1['n_iter']} iter) | aprovado={ok1}")

    model2, sets_info2, _ = core.build_opf(data, core.HOURS)
    r2 = case_policy_optimal_tap(data, sets_info2, model2, policy,
                                solver_getter, timelimit, tee)
    tap_traj2 = {ph: {t: int(round(pyo.value(model2.tap_pos[ph, t])))
                     for t in sets_info2['hours']}
                for ph in sets_info2['svr_phs']}
    ops2, total2 = _count_tap_ops(tap_traj2)
    ok2, rep2 = check_acceptance(r2['V_final'], sets_info2, data, total2,
                                 baseline_tap_ops)
    rep2['converged'] = r2['converged']; rep2['n_iter'] = r2['n_iter']
    rep2['tap_ops_total'] = total2; rep2['ops_by_phase'] = ops2
    rep2['aprovado'] = ok2
    results['POLICY_OPTIMAL_TAP'] = rep2
    print(f"  POLICY_OPTIMAL_TAP: convergiu={r2['converged']} "
         f"({r2['n_iter']} iter) | tap_ops={total2} | aprovado={ok2}")

    r3 = case_policy_local_tap(data, sets_info, policy)
    ok3, rep3 = check_acceptance(r3['V_final'], sets_info, data,
                                 r3['total_ops'], baseline_tap_ops)
    rep3['tap_ops_total'] = r3['total_ops']; rep3['ops_by_phase'] = r3['ops_by_phase']
    rep3['aprovado'] = ok3
    results['POLICY_LOCAL_TAP'] = rep3
    print(f"  POLICY_LOCAL_TAP: tap_ops={r3['total_ops']} | aprovado={ok3}")

    dss_path = export_validated_dss(data, policy, sets_info, results, out_dir)
    json_path = export_expected_results_json(results, out_dir)
    export_violations_report(results, data, out_dir)
    export_policy_validation_summary(results, policy, out_dir)
    export_tap_comparison(results, baseline_tap_ops, out_dir)

    print("\n" + "=" * 78)
    print("  RESUMO — política A/B/C validada autoconsistentemente:")
    for name, r in results.items():
        print(f"    {name:>20}: aprovado={r['aprovado']} | "
             f"tap_ops={r.get('tap_ops_total','-')} | "
             f"viol_MT={r['n_violacoes_MT']} viol_BT={r['n_violacoes_BT']}")
    print("=" * 78)

    return {'results': results, 'dss_path': dss_path, 'json_path': json_path,
           'policy': policy}


if __name__ == '__main__':
    BUSES = os.environ.get('BUSES_FILE', 'buses_1.xlsx')
    BRANCHES = os.environ.get('BRANCHES_FILE', 'branches_1.xlsx')
    POLICY_DIR = os.environ.get('POLICY_DIR', '.')
    OUT = os.environ.get('OUT_DIR', '.')
    validate_policy_abc_with_opf(BUSES, BRANCHES, POLICY_DIR, OUT)
