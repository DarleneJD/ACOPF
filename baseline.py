# -*- coding: utf-8 -*-
# =============================================================================
#  BASELINE.py  —  CONTAGEM DE OPERACOES DE TAP DO REGULADOR (SEM OTIMIZACAO)
#  Codigo UNICO e AUTOCONTIDO. Nao depende do arquivo do OPF.
# -----------------------------------------------------------------------------
#  O numero de operacoes de tap vem da logica de banda morta do regulador, que
#  decide com base na TENSAO estimada no centro de carga. Para obter essa tensao
#  e preciso um FLUXO DE POTENCIA. Este arquivo o implementa explicitamente como
#  varredura backward/forward (LinDistFlow com termo de perdas), periodo a
#  periodo (144 x 10 min). As equacoes do fluxo estao na funcao power_flow_sweep.
#
#  Cadeia logica:
#    perfis (irrad/carga/temp) -> injecao liquida por barra (carga - PV)
#      -> varredura BACKWARD (fluxos nos ramos)
#      -> varredura FORWARD  (tensoes; no SVR: V_lv = tau * V_mv)
#      -> LDC: V_centro = V_reg - (R_ldc*I_P + X_ldc*I_Q)
#      -> banda morta + delay -> decisao de tap -> contagem
#
#  Settings LDC:
#    O regulador opera com R=3 V / X=9 V (aba 'Reg'). O metodo do centro de
#    carga (Kersting, Cap.7) aplicado a barra 671 reproduz ~3.0/9.1 V, validando
#    o equipamento. Mantemos por padrao os valores IMPLANTADOS (modo 'deployed').
#
#  Uso:
#    python baseline.py                      # roda o baseline + sensibilidade
#    (ou) from baseline import run_baseline   # importavel
#
#  Requer: pandas, openpyxl. Ajuste BUSES_FILE / BRANCHES_FILE abaixo.
# =============================================================================
import math
import pandas as pd
from collections import deque, defaultdict

# Caminhos dos arquivos da rede (AJUSTE no Colab) -----------------------------
BUSES_FILE    = 'buses_1.xlsx'
BRANCHES_FILE = 'branches_1.xlsx'

# Horizonte -------------------------------------------------------------------
N_PERIODS = 144          # 24 h x 60 / 10
DT_MIN    = 10           # minutos por periodo
HOURS     = list(range(N_PERIODS))


IRRAD = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000008, 0.000080, 0.000672, 0.002286,
    0.007955, 0.015941, 0.022688, 0.037534, 0.037534, 0.037534, 0.037534, 0.037534,
    0.228267, 0.228267, 0.228267, 0.228267, 0.303663, 0.403642, 0.340885, 0.338831,
    0.475905, 0.497985, 0.531826, 0.571495, 0.598836, 0.629815, 0.658507, 0.687815,
    0.723270, 0.741506, 0.768072, 0.791615, 0.811002, 0.833043, 0.847065, 0.865453,
    0.889556, 0.909654, 0.535304, 0.609437, 0.972867, 0.718562, 0.821835, 1.000000,
    0.995747, 0.853749, 0.911717, 0.662080, 0.551868, 0.967518, 0.949435, 0.892242,
    0.972875, 0.919104, 0.651704, 0.353381, 0.547719, 0.361383, 0.326927, 0.700342,
    0.667077, 0.627656, 0.619742, 0.586301, 0.553395, 0.502454, 0.448827, 0.442983,
    0.400269, 0.374286, 0.326104, 0.297339, 0.269415, 0.237629, 0.210439, 0.169579,
    0.153966, 0.088307, 0.071950, 0.087652, 0.038829, 0.023192, 0.012304, 0.004485,
    0.002127, 0.000696, 0.000176, 0.000016, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
]
TECHNICHDOBRASIL = [
    0.909000, 0.909000, 0.909000, 0.909000, 0.909000, 0.909000, 0.721000, 0.721000,
    0.721000, 0.721000, 0.721000, 0.721000, 0.714000, 0.714000, 0.714000, 0.714000,
    0.714000, 0.714000, 0.721000, 0.721000, 0.721000, 0.721000, 0.721000, 0.721000,
    0.718000, 0.718000, 0.718000, 0.718000, 0.718000, 0.718000, 0.755000, 0.755000,
    0.755000, 0.755000, 0.755000, 0.755000, 0.761000, 0.761000, 0.761000, 0.761000,
    0.761000, 0.761000, 0.934000, 0.934000, 0.934000, 0.934000, 0.934000, 0.934000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.986000, 0.986000,
    0.986000, 0.986000, 0.986000, 0.986000, 0.992000, 0.992000, 0.992000, 0.992000,
    0.992000, 0.992000, 0.883000, 0.883000, 0.883000, 0.883000, 0.883000, 0.883000,
    0.783000, 0.783000, 0.783000, 0.783000, 0.783000, 0.783000, 0.780000, 0.780000,
    0.780000, 0.780000, 0.780000, 0.780000, 0.934000, 0.934000, 0.934000, 0.934000,
    0.934000, 0.934000, 0.919000, 0.919000, 0.919000, 0.919000, 0.919000, 0.919000,
    0.738000, 0.738000, 0.738000, 0.738000, 0.738000, 0.738000, 0.893000, 0.893000,
    0.893000, 0.893000, 0.893000, 0.893000, 0.889000, 0.889000, 0.889000, 0.889000,
    0.889000, 0.889000, 0.898000, 0.898000, 0.898000, 0.898000, 0.898000, 0.898000,
    0.885000, 0.885000, 0.885000, 0.885000, 0.885000, 0.885000, 0.792000, 0.792000,
    0.792000, 0.792000, 0.792000, 0.792000, 0.817000, 0.817000, 0.817000, 0.817000,
    0.817000, 0.817000, 0.926000, 0.926000, 0.926000, 0.926000, 0.926000, 0.926000,
]
CURVA_R = [
    0.289000, 0.307000, 0.319000, 0.330000, 0.342000, 0.362000, 0.206000, 0.223000,
    0.235000, 0.246000, 0.258000, 0.278000, 0.190000, 0.207000, 0.219000, 0.230000,
    0.242000, 0.262000, 0.185000, 0.202000, 0.214000, 0.225000, 0.237000, 0.257000,
    0.190000, 0.207000, 0.219000, 0.230000, 0.242000, 0.262000, 0.378000, 0.396000,
    0.408000, 0.419000, 0.431000, 0.451000, 0.488000, 0.506000, 0.518000, 0.529000,
    0.541000, 0.560000, 0.498000, 0.516000, 0.528000, 0.539000, 0.552000, 0.571000,
    0.493000, 0.511000, 0.523000, 0.534000, 0.546000, 0.566000, 0.383000, 0.401000,
    0.413000, 0.424000, 0.436000, 0.456000, 0.415000, 0.433000, 0.444000, 0.455000,
    0.468000, 0.487000, 0.504000, 0.522000, 0.534000, 0.545000, 0.557000, 0.576000,
    0.472000, 0.490000, 0.502000, 0.513000, 0.525000, 0.545000, 0.378000, 0.396000,
    0.408000, 0.419000, 0.431000, 0.451000, 0.415000, 0.433000, 0.444000, 0.455000,
    0.468000, 0.487000, 0.389000, 0.406000, 0.418000, 0.429000, 0.442000, 0.461000,
    0.425000, 0.443000, 0.455000, 0.466000, 0.478000, 0.498000, 0.713000, 0.731000,
    0.743000, 0.754000, 0.766000, 0.785000, 0.896000, 0.914000, 0.926000, 0.937000,
    0.950000, 0.969000, 0.880000, 0.899000, 0.911000, 0.922000, 0.934000, 0.953000,
    0.928000, 0.946000, 0.958000, 0.969000, 0.981000, 1.000000, 0.786000, 0.804000,
    0.816000, 0.827000, 0.840000, 0.859000, 0.744000, 0.762000, 0.775000, 0.786000,
    0.798000, 0.817000, 0.462000, 0.480000, 0.492000, 0.503000, 0.515000, 0.534000,
]
TEMP_AMB = [
    15.690000, 19.500000, 19.320000, 19.200000, 19.200000, 19.200000, 18.820000, 18.820000,
    18.770000, 18.800000, 18.800000, 18.660000, 18.500000, 18.400000, 18.390000, 18.320000,
    18.300000, 18.200000, 18.200000, 18.240000, 16.390000, 18.180000, 18.100000, 18.210000,
    18.200000, 18.280000, 18.470000, 18.640000, 18.700000, 18.643000, 18.706000, 18.749000,
    18.911000, 19.123000, 19.477000, 15.417000, 15.417000, 15.417000, 15.417000, 30.553000,
    30.553000, 30.553000, 30.553000, 30.553000, 33.090000, 37.848000, 36.005000, 35.705000,
    40.833000, 42.616000, 44.289000, 46.339000, 47.948000, 49.419000, 50.901000, 52.576000,
    54.942000, 56.025000, 57.023000, 58.524000, 59.512000, 60.533000, 61.751000, 62.310000,
    64.072000, 59.398000, 51.125000, 52.483000, 67.489000, 58.468000, 62.535000, 69.999000,
    70.013000, 64.843000, 66.918000, 57.740000, 53.172000, 69.220000, 69.173000, 67.697000,
    72.019000, 70.477000, 59.035000, 46.033000, 52.210000, 44.826000, 42.819000, 57.976000,
    57.766000, 56.095000, 55.925000, 54.838000, 53.272000, 50.641000, 48.424000, 47.706000,
    46.446000, 44.841000, 42.187000, 40.413000, 39.151000, 34.869000, 31.966000, 33.879000,
    33.068000, 30.352000, 29.152000, 29.156000, 26.738000, 25.957000, 25.261000, 24.705000,
    24.523000, 24.077000, 23.927000, 23.801000, 23.760000, 23.610000, 23.390000, 23.200000,
    23.210000, 23.300000, 23.390000, 23.200000, 22.980000, 22.900000, 22.900000, 22.880000,
    22.960000, 23.080000, 23.300000, 23.420000, 23.510000, 23.540000, 23.630000, 23.860000,
    23.860000, 24.030000, 23.790000, 23.830000, 23.900000, 23.930000, 23.830000, 23.710000,
]


# ─────────────────────────────────────────────────────────────────────────────
#  MODELO PV COM TEMPERATURA (Skoplaki & Palyvos 2009)
#    T_cell = T_amb + irrad*(NOCT-20);  derate = 1 + gamma*(T_cell-25)
#    PV_EFF_FACTOR[t] = irrad[t] * derate[t]   (P_efetiva / P_nominal)
# ─────────────────────────────────────────────────────────────────────────────
NOCT, GAMMA_PV, T_STC = 45.0, -0.0040, 25.0

def _pv_derate(irr_norm, t_amb):
    t_cell = t_amb + irr_norm * (NOCT - 20.0)
    return max(1.0 + GAMMA_PV * (t_cell - T_STC), 0.5)

PV_EFF_FACTOR = [IRRAD[t] * _pv_derate(IRRAD[t], TEMP_AMB[t]) for t in range(N_PERIODS)]

# Perfis de carga por nivel: MT -> TECHNICHDOBRASIL ; BT -> CURVA_R
LOAD_PROFILE_MT = TECHNICHDOBRASIL
LOAD_PROFILE_BT = CURVA_R

# ─────────────────────────────────────────────────────────────────────────────
#  BASES DO SISTEMA (identicas ao modelo do OPF)
# ─────────────────────────────────────────────────────────────────────────────
SBASE = 1000.0/3.0                     # kVA/fase
VMT   = 4.16/math.sqrt(3); ZMT = VMT**2/SBASE*1000; IMT = SBASE/VMT
VBT   = 0.480/math.sqrt(3); ZBT = VBT**2/SBASE*1000; IBT = SBASE/VBT
V_LN  = VMT*1000.0                     # tensao fase-neutro nominal (V)

# Parametros do controle local de tap (banda morta + delay) -------------------
TAP_MIN  = 0.90                        # tau minimo (pos=0)
DELAY_PER = 3                          # periodos de espera (~30 s) antes de comutar

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITARIOS DE LEITURA
# ─────────────────────────────────────────────────────────────────────────────
def _norm(v):
    s = str(v).strip(); return s[:-2] if s.endswith('.0') else s
def _ff(v, d=0.0):
    try:
        x = float(v); return d if math.isnan(x) else x
    except: return d
def _phs(v):
    out = []
    for p in str(v).strip().replace('.', ',').split(','):
        try:
            ph = int(float(p.strip()))
            if ph in (1, 2, 3): out.append(ph)
        except: pass
    return sorted(set(out))

# ─────────────────────────────────────────────────────────────────────────────
#  LEITURA DA REDE  ->  dict `data`
# ─────────────────────────────────────────────────────────────────────────────
def load_data(buses_file=BUSES_FILE, branches_file=BRANCHES_FILE):
    buses = {}; slack = None
    df = pd.read_excel(buses_file, 'MT'); df.columns = df.columns.str.strip()
    df = df[~df['name'].astype(str).str.contains('Source', case=False)]
    for _, r in df.iterrows():
        n = _norm(r['name']); tb = int(_ff(r.get('tb', 0)))
        buses[n] = {'level': 'mv', 'phases': _phs(r.get('phases', '1,2,3'))}
        if tb == 3 and slack is None: slack = n

    # cargas MT (P_D total da linha / nph / SBASE)
    loads = {}; loads_mt = {}
    dl = pd.read_excel(buses_file, 'Load_MT'); dl.columns = dl.columns.str.strip()
    for _, r in dl.iterrows():
        n = _norm(r['name']); phs = _phs(r.get('phases', '1,2,3')); nph = max(len(phs), 1)
        P = _ff(r.get('P_D')); Q = _ff(r.get('Q_D'))
        for ph in phs:
            loads_mt[(n, ph)] = {'P_pu': P/nph/SBASE, 'Q_pu': Q/nph/SBASE}

    # cargas BT
    db = pd.read_excel(buses_file, 'BT'); db.columns = db.columns.str.strip()
    for _, r in db.iterrows():
        n = _norm(r['name']); phs = _phs(r.get('phases', '1,2,3')); nph = max(len(phs), 1)
        buses[n] = {'level': 'lv', 'phases': phs}
        Pt = _ff(r.get('P_D')); Qt = _ff(r.get('Q_D'))
        for ph in phs:
            loads[(n, ph)] = {'P_pu': Pt/nph/SBASE, 'Q_pu': Qt/nph/SBASE}
    for n, d in buses.items():
        if d['level'] == 'mv':
            for ph in d['phases']:
                loads.setdefault((n, ph), loads_mt.get((n, ph), {'P_pu': 0., 'Q_pu': 0.}))

    # PV (100% penetracao): P_rated por (bus, ph)
    pv = {}
    dp = pd.read_excel(buses_file, 'PV'); dp.columns = dp.columns.str.strip()
    for _, r in dp.iterrows():
        bus = _norm(r['Bus'])
        if bus not in buses: continue
        phs = _phs(r.get('phases', '1,2,3')); nph = max(len(phs), 1)
        kva = _ff(r.get('kva')); Sf = kva/nph
        for ph in phs:
            P_rated = _ff(r.get(f'p_pv_{ph}'), Sf)
            pv[(bus, ph)] = {'P_rated_pu': P_rated/SBASE}

    # ramos MT / BT / trafos
    branches = {}
    dm = pd.read_excel(branches_file, 'MT'); dm.columns = dm.columns.str.strip()
    for _, r in dm.iterrows():
        fr = _norm(r['l']); to = _norm(r['k'])
        if to == 'RG60': fr, to = to, fr
        if 'Source' in (fr, to): continue
        R = _ff(r['R']); X = _ff(r['X'])
        for ph in _phs(r.get('phase', '1,2,3')):
            branches[(fr, to, ph)] = {'r_pu': R/ZMT, 'x_pu': X/ZMT, 'level': 'mv'}
    db2 = pd.read_excel(branches_file, 'BT'); db2.columns = db2.columns.str.strip()
    for _, r in db2.iterrows():
        fr = _norm(r['l']); to = _norm(r['k']); R = _ff(r['R']); X = _ff(r['X'])
        for ph in _phs(r.get('phase', '1,2,3')):
            branches[(fr, to, ph)] = {'r_pu': R/ZBT, 'x_pu': X/ZBT, 'level': 'lv'}
    dt = pd.read_excel(branches_file, 'Trafos'); dt.columns = dt.columns.str.strip()
    for _, r in dt.iterrows():
        if str(r['trafo_id']).strip() == 'Sub': continue
        mv = _norm(r['mv_bus']); lv = _norm(r['lv_bus']); phs = _phs(r.get('phases', '1,2,3'))
        Zb = VBT**2/SBASE*1000; R = _ff(r['R_ohm']); X = _ff(r['X_ohm'])
        for ph in phs:
            branches[(mv, lv, ph)] = {'r_pu': R/Zb, 'x_pu': X/Zb, 'level': 'trafo'}
    for (fr, to, ph) in list(branches):
        for b in (fr, to):
            buses.setdefault(b, {'level': 'mv', 'phases': [1, 2, 3]})
            loads.setdefault((b, ph), {'P_pu': 0., 'Q_pu': 0.})

    # SVR (aba Reg) — le os nomes REAIS de coluna (r_LDC_V, vreg_V, band_V, ...)
    svr = {}
    dr = pd.read_excel(branches_file, 'Reg'); dr.columns = dr.columns.str.strip()
    for _, r in dr.iterrows():
        ph = int(_ff(r.get('phases'), 1)); mv = _norm(r['mv_bus']); lv = _norm(r['lv_bus'])
        I = _ff(r.get('ctprim_A'), 700.); n_tap = int(_ff(r.get('TapNum'), 33))
        step = _ff(r.get('Step'), 0.00625)
        svr[ph] = {
            'mv_bus': mv, 'lv_bus': lv, 'Imax_pu': I/IMT,
            'tap_step': step, 'n_tap': n_tap, 'tap_init': _ff(r.get('Taps_init', 1.0), 1.0),
            # settings IMPLANTADOS no equipamento (volts -> pu na base 120 V)
            'R_ldc_pu': _ff(r.get('r_LDC_V'), 3.0)/120.0,
            'X_ldc_pu': _ff(r.get('X_LDC_V'), 9.0)/120.0,
            'V_ref':   _ff(r.get('vreg_V'), 122.0)/120.0,   # 122 V -> 1.0167 pu
            'BW':      _ff(r.get('band_V'), 2.0)/120.0,      # 2 V -> 0.0167 pu
        }
        for b in (mv, lv):
            buses.setdefault(b, {'level': 'mv', 'phases': [1, 2, 3]})

    return {'buses': buses, 'loads': loads, 'pv': pv,
            'branches': branches, 'svr': svr, 'slack': slack}


# =============================================================================
#  ARVORE RADIAL  +  FLUXO DE POTENCIA  (varredura backward/forward)
# =============================================================================
def build_tree(data):
    """Adjacencia, ordem BFS e parent[(b,ph)] a partir do slack.
    O SVR entra como aresta (mv_bus -> lv_bus)."""
    br = data['branches']; svr = data['svr']; buses = data['buses']; slk = data['slack']
    adj = defaultdict(list)
    for (fr, to, ph) in br: adj[(fr, ph)].append((to, ph, (fr, to, ph)))
    for ph, d in svr.items(): adj[(d['mv_bus'], ph)].append((d['lv_bus'], ph, ('SVR', ph)))
    order = []; seen = set(); par = {}
    q = deque((slk, ph) for ph in buses[slk]['phases'])
    for it in q: seen.add(it)
    while q:
        b, ph = q.popleft(); order.append((b, ph))
        for (to, tph, bk) in adj.get((b, ph), []):
            if (to, tph) not in seen:
                seen.add((to, tph)); par[(to, tph)] = ((b, ph), bk); q.append((to, tph))
    return adj, order, par


def power_flow_sweep(data, adj, order, par, t, tap_pos):
    """UM passo de fluxo de potencia no periodo t, dado o tap atual por fase.

    Retorna V[(b,ph)] e os fluxos P_br/Q_br nos ramos (vistos na barra a jusante).

    EQUACOES
    --------
    (1) Injecao liquida por barra:
            P_net = P_carga(t) - P_pv(t)            P_pv = P_rated * PV_EFF_FACTOR[t]
            Q_net = Q_carga(t)
        carga(t) = P_pu * perfil(t)   (perfil MT ou BT conforme o nivel da barra)

    (2) Varredura BACKWARD (folhas -> raiz): fluxo no ramo b = injecao de b
        + soma dos fluxos dos ramos a jusante.
            P_br[b] = P_net[b] + sum_{filhos} P_br[filho]

    (3) Varredura FORWARD (raiz -> folhas): queda de tensao linearizada com
        termo de perdas (LinDistFlow); no SVR aplica a razao de tap.
            l_sq   = (P_br^2 + Q_br^2) / V_mont^2                 (corrente^2)
            V_jus  = V_mont - r*P_br - x*Q_br + 0.5*(r^2+x^2)*l_sq
            (no SVR)  V_jus = tau * V_mont,  tau = TAP_MIN + pos*step
    """
    buses = data['buses']; loads = data['loads']; pv = data['pv']
    branches = data['branches']; svr = data['svr']; slack = data['slack']

    # (1) injecao liquida por barra ------------------------------------------
    P_net = {}; Q_net = {}
    for (b, ph) in order:
        prof = LOAD_PROFILE_MT[t] if buses.get(b, {}).get('level') == 'mv' else LOAD_PROFILE_BT[t]
        Pd = loads.get((b, ph), {'P_pu': 0.})['P_pu'] * prof
        Qd = loads.get((b, ph), {'Q_pu': 0.})['Q_pu'] * prof
        Ppv = pv.get((b, ph), {'P_rated_pu': 0.})['P_rated_pu'] * PV_EFF_FACTOR[t]
        P_net[(b, ph)] = Pd - Ppv
        Q_net[(b, ph)] = Qd

    # (2) backward sweep ------------------------------------------------------
    P_br = {}; Q_br = {}
    for (b, ph) in reversed(order):
        pf_ = P_net[(b, ph)]; qf_ = Q_net[(b, ph)]
        for (to, tph, bk) in adj.get((b, ph), []):
            pf_ += P_br.get((to, ph), 0.0); qf_ += Q_br.get((to, ph), 0.0)
        P_br[(b, ph)] = pf_; Q_br[(b, ph)] = qf_

    # (3) forward sweep -------------------------------------------------------
    V = {}
    for (b, ph) in order:
        if b == slack:
            V[(b, ph)] = 1.0; continue
        if (b, ph) not in par:
            V[(b, ph)] = 0.97; continue
        (up, uph), bk = par[(b, ph)]
        Vup = V.get((up, uph), 1.0)
        if isinstance(bk, tuple) and len(bk) == 2 and bk[0] == 'SVR':
            tau = TAP_MIN + tap_pos[ph] * svr[ph]['tap_step']
            V[(b, ph)] = tau * Vup
        else:
            r = branches[bk]['r_pu']; x = branches[bk]['x_pu']
            P = P_br.get((b, ph), 0.0); Q = Q_br.get((b, ph), 0.0)
            l_sq = (P*P + Q*Q) / max(Vup*Vup, 0.64)
            V[(b, ph)] = max(Vup - r*P - x*Q + 0.5*(r*r + x*x)*l_sq, 0.80)

    return V, P_br, Q_br


def power_flow_bfs(data, adj, order, par, t, tap_pos, max_iter=40, tol=1e-7):
    """Varredura BACKWARD/FORWARD ITERATIVA com fasores complexos (metodo
    ladder — o mesmo principio do fluxo radial do OpenDSS).

    Diferenca-chave para power_flow_sweep (linear):
      - tensoes e correntes COMPLEXAS; nada e linearizado;
      - cargas de POTENCIA CONSTANTE: I = conj(S/V) -> a corrente sobe quando
        V cai (model=1 do OpenDSS). Essa realimentacao V-I amplifica as
        excursoes de tensao, justamente o que a versao linear amortece;
      - itera ate convergir (|dV| < tol).

    Retorna (Vmag, P_br, Q_br) com a MESMA interface da versao linear, para
    poder ser usada de forma intercambiavel em simulate_baseline.
    """
    buses = data['buses']; loads = data['loads']; pv = data['pv']
    branches = data['branches']; svr = data['svr']; slack = data['slack']

    # potencia liquida (constante) por barra: carga(t) - PV(t); PV unity PF
    Snet = {}
    for (b, ph) in order:
        prof = LOAD_PROFILE_MT[t] if buses.get(b, {}).get('level') == 'mv' else LOAD_PROFILE_BT[t]
        Pd = loads.get((b, ph), {'P_pu': 0.})['P_pu'] * prof
        Qd = loads.get((b, ph), {'Q_pu': 0.})['Q_pu'] * prof
        Ppv = pv.get((b, ph), {'P_rated_pu': 0.})['P_rated_pu'] * PV_EFF_FACTOR[t]
        Snet[(b, ph)] = complex(Pd - Ppv, Qd)

    V = {(b, ph): complex(1.0, 0.0) for (b, ph) in order}
    Ibr = {}
    for _ in range(max_iter):
        Vmag_prev = {k: abs(v) for k, v in V.items()}
        # ---- BACKWARD: correntes complexas nos ramos (folhas -> raiz) -------
        Ibr = {}
        for (b, ph) in reversed(order):
            Vb = V[(b, ph)]
            Ii = (Snet[(b, ph)] / Vb).conjugate() if abs(Vb) > 1e-6 else 0j
            for (to, tph, bk) in adj.get((b, ph), []):
                Ic = Ibr.get((to, ph), 0j)
                if isinstance(bk, tuple) and len(bk) == 2 and bk[0] == 'SVR':
                    tau = TAP_MIN + tap_pos[ph] * svr[ph]['tap_step']
                    Ic = tau * Ic                      # I_mv = tau * I_lv
                Ii += Ic
            Ibr[(b, ph)] = Ii
        # ---- FORWARD: tensoes complexas (raiz -> folhas) --------------------
        for (b, ph) in order:
            if b == slack:
                V[(b, ph)] = complex(1.0, 0.0); continue
            if (b, ph) not in par:
                continue
            (up, uph), bk = par[(b, ph)]
            Vup = V[(up, uph)]
            if isinstance(bk, tuple) and len(bk) == 2 and bk[0] == 'SVR':
                tau = TAP_MIN + tap_pos[ph] * svr[ph]['tap_step']
                V[(b, ph)] = tau * Vup
            else:
                Z = complex(branches[bk]['r_pu'], branches[bk]['x_pu'])
                V[(b, ph)] = Vup - Z * Ibr[(b, ph)]
        # ---- convergencia ----------------------------------------------------
        if max(abs(abs(V[k]) - Vmag_prev[k]) for k in V) < tol:
            break

    # fluxos reais nos ramos (P+jQ = V * conj(I)) para o sensing do LDC
    P_br = {}; Q_br = {}
    for (b, ph) in order:
        S = V[(b, ph)] * Ibr.get((b, ph), 0j).conjugate()
        P_br[(b, ph)] = S.real; Q_br[(b, ph)] = S.imag
    Vmag = {k: abs(v) for k, v in V.items()}
    return Vmag, P_br, Q_br


# =============================================================================
#  CONFIGURACAO LDC PELO CENTRO DE CARGA (Kersting, Cap.7) — opcional
#    R_ldc_pu = Z_linha_ohm * CT_P / V_LN   (equivalencia p.u. na base 120 V)
#    'deployed' = mantem os 3/9 V do equipamento (padrao).
# =============================================================================
def _branch_lvl(branches, bk):
    if isinstance(bk, tuple) and len(bk) == 2 and bk[0] == 'SVR': return 'svr'
    return branches.get(bk, {}).get('level', 'mv')

def _path_Z_ohm(par, branches, mv, ph, reg_lv):
    R = X = 0.0; cur, cph = mv, ph; g = 0
    while cur != reg_lv and (cur, cph) in par and g < 10000:
        (up, uph), bk = par[(cur, cph)]
        if _branch_lvl(branches, bk) == 'mv':
            R += branches[bk]['r_pu'] * ZMT; X += branches[bk]['x_pu'] * ZMT
        cur, cph = up, uph; g += 1
    return R, X

def _tapoff(par, branches, b, ph, reg_lv):
    cur, cph = b, ph
    while True:
        if cur == reg_lv: return cur
        if (cur, cph) not in par: return cur
        (up, uph), bk = par[(cur, cph)]
        if _branch_lvl(branches, bk) == 'trafo': return up
        cur, cph = up, uph

def set_ldc(data, par, mode='deployed'):
    """Injeta R_ldc_pu/X_ldc_pu em data['svr']. Retorna {ph:(R_volts,X_volts)}.
    mode: 'deployed' (3/9 implantados) | 'centroide' | '671' | '675' | ..."""
    svr = data['svr']; branches = data['branches']; loads = data['loads']
    if mode == 'deployed':
        return {ph: (svr[ph]['R_ldc_pu']*120, svr[ph]['X_ldc_pu']*120) for ph in svr}
    n_pt = V_LN/120.0; out = {}
    for ph in sorted(svr):
        reg_lv = svr[ph]['lv_bus']; ct_p = svr[ph]['Imax_pu']*IMT
        if mode == 'centroide':
            nR = nX = den = 0.0
            for (b, p), ld in loads.items():
                if p != ph: continue
                S = math.hypot(ld.get('P_pu', 0.), ld.get('Q_pu', 0.))
                if S <= 1e-9: continue
                mt = _tapoff(par, branches, b, ph, reg_lv)
                Rp, Xp = _path_Z_ohm(par, branches, mt, ph, reg_lv)
                nR += S*Rp; nX += S*Xp; den += S
            R_lc, X_lc = (nR/den, nX/den) if den > 0 else (0., 0.)
        else:
            R_lc, X_lc = _path_Z_ohm(par, branches, mode, ph, reg_lv)
        Rv = R_lc*ct_p/n_pt; Xv = X_lc*ct_p/n_pt
        svr[ph]['R_ldc_pu'] = Rv/120.0; svr[ph]['X_ldc_pu'] = Xv/120.0
        out[ph] = (Rv, Xv)
    return out


# =============================================================================
#  SIMULACAO DO CONTROLE LOCAL  ->  CONTAGEM DE OPERACOES DE TAP
# =============================================================================
def simulate_baseline(data, verbose=True, pf_fn=power_flow_sweep):
    """Roda o fluxo de potencia em 144 periodos com a logica local de tap.

    pf_fn : funcao de fluxo de potencia a usar:
            power_flow_sweep -> LinDistFlow linear de passo unico (padrao);
            power_flow_bfs   -> ladder complexo iterativo (aprox. OpenDSS).

    Para cada periodo t e fase ph:
      1. pf_fn(...) com o tap atual  -> V e fluxos
      2. tensao sentida pelo rele via LDC:
             V_centro = V_reg - (R_ldc*I_P + X_ldc*I_Q)
         com I_P = (P_svr/V)/Imax_pu , I_Q = (Q_svr/V)/Imax_pu
      3. banda morta sobre (V_centro - V_ref) + delay -> sobe/desce/mantem tap
    Retorna (ops_por_fase, tap_traj, V_centro_traj).
    """
    adj, order, par = build_tree(data)
    svr = data['svr']; svr_phs = sorted(svr)
    step = {ph: svr[ph]['tap_step'] for ph in svr_phs}
    n_pos = {ph: svr[ph]['n_tap'] for ph in svr_phs}

    # posicao inicial = neutra (tau=1.0 -> pos = (1.0-TAP_MIN)/step)
    pos = {ph: int(round((1.0 - TAP_MIN)/step[ph])) for ph in svr_phs}
    delay = {ph: 0 for ph in svr_phs}
    ops = {ph: 0 for ph in svr_phs}
    tap_traj = {}; Vc_traj = {}

    for t in HOURS:
        V, P_br, Q_br = pf_fn(data, adj, order, par, t, pos)
        for ph in svr_phs:
            reg = svr[ph]['lv_bus']
            V_reg = V.get((reg, ph), 1.0)
            P_svr = P_br.get((reg, ph), 0.0); Q_svr = Q_br.get((reg, ph), 0.0)
            Vi = max(V_reg, 0.5)
            i_nom = svr[ph]['Imax_pu'] or 1.0
            I_P = (P_svr/Vi)/i_nom; I_Q = (Q_svr/Vi)/i_nom
            V_centro = V_reg - (svr[ph]['R_ldc_pu']*I_P + svr[ph]['X_ldc_pu']*I_Q)
            Vc_traj[(ph, t)] = V_centro

            err = V_centro - svr[ph]['V_ref']; BW = svr[ph]['BW']
            delta = 0
            if err > BW/2 and pos[ph] > 0:            delta = -1   # V alta -> desce
            elif err < -BW/2 and pos[ph] < n_pos[ph]-1: delta = +1  # V baixa -> sobe
            if delta != 0:
                if delay[ph] >= DELAY_PER:
                    pos[ph] = max(0, min(n_pos[ph]-1, pos[ph] + delta))
                    ops[ph] += 1; delay[ph] = 0
                else:
                    delay[ph] += 1
            else:
                delay[ph] = 0
            tap_traj[(ph, t)] = pos[ph]

    if verbose:
        print("  Fase   ops")
        for ph in svr_phs: print(f"    {ph}    {ops[ph]:>3}")
        print(f"  TOTAL  {sum(ops.values()):>3}")
    return ops, tap_traj, Vc_traj


# =============================================================================
#  SIMULACAO ESTILO OpenDSS (controle assenta na banda a cada periodo)
#    Delay_s=15 / tapdelay_s=2 << passo de 600 s  ->  o delay nao restringe;
#    a cada periodo o regulador comuta (ate maxtapchange) ate V_centro entrar
#    na banda. Conta CADA passo de tap, como o OpenDSS.
# =============================================================================
def simulate_baseline_opendss(data, verbose=True, pf_fn=power_flow_bfs,
                              maxtapchange=16):
    adj, order, par = build_tree(data)
    svr = data['svr']; svr_phs = sorted(svr)
    step = {ph: svr[ph]['tap_step'] for ph in svr_phs}
    n_pos = {ph: svr[ph]['n_tap'] for ph in svr_phs}
    pos = {ph: int(round((1.0 - TAP_MIN)/step[ph])) for ph in svr_phs}
    ops = {ph: 0 for ph in svr_phs}; tap_traj = {}

    for t in HOURS:
        # loop de controle dentro do periodo: comuta ate assentar na banda
        for _ in range(maxtapchange):
            V, P_br, Q_br = pf_fn(data, adj, order, par, t, pos)
            moved = False
            for ph in svr_phs:
                reg = svr[ph]['lv_bus']; Vr = V.get((reg, ph), 1.0)
                Ps = P_br.get((reg, ph), 0.0); Qs = Q_br.get((reg, ph), 0.0)
                Vi = max(Vr, 0.5); i_nom = svr[ph]['Imax_pu'] or 1.0
                I_P = (Ps/Vi)/i_nom; I_Q = (Qs/Vi)/i_nom
                V_centro = Vr - (svr[ph]['R_ldc_pu']*I_P + svr[ph]['X_ldc_pu']*I_Q)
                err = V_centro - svr[ph]['V_ref']; BW = svr[ph]['BW']
                if err > BW/2 and pos[ph] > 0:
                    pos[ph] -= 1; ops[ph] += 1; moved = True
                elif err < -BW/2 and pos[ph] < n_pos[ph]-1:
                    pos[ph] += 1; ops[ph] += 1; moved = True
            if not moved:
                break
        for ph in svr_phs: tap_traj[(ph, t)] = pos[ph]

    if verbose:
        print("  Fase   ops"); [print(f"    {ph}    {ops[ph]:>3}") for ph in svr_phs]
        print(f"  TOTAL  {sum(ops.values()):>3}")
    return ops, tap_traj


# =============================================================================
#  DRIVER
# =============================================================================
def run_baseline(buses_file=BUSES_FILE, branches_file=BRANCHES_FILE,
                 modes=('deployed', '671', 'centroide', '675')):
    data = load_data(buses_file, branches_file)
    print(f"Rede: slack={data['slack']} | barras={len(data['buses'])} | "
          f"ramos={len(data['branches'])} | SVRs={len(data['svr'])} | "
          f"PVs={len(data['pv'])}")
    s0 = data['svr'][sorted(data['svr'])[0]]
    print(f"Regulador: V_ref={s0['V_ref']:.4f} pu | BW={s0['BW']:.4f} pu | "
          f"deployed R={s0['R_ldc_pu']*120:.1f}V X={s0['X_ldc_pu']*120:.1f}V\n")

    _, order, par = build_tree(data)
    print("="*64)
    print(f"  {'Modo LDC':>11} {'R[V]':>6} {'X[V]':>6}  {'F1':>3} {'F2':>3} {'F3':>3} {'Total':>6}")
    print("-"*64)
    res = {}
    for mode in modes:
        sett = set_ldc(data, par, mode)
        Rm = sum(v[0] for v in sett.values())/len(sett)
        Xm = sum(v[1] for v in sett.values())/len(sett)
        ops, _, _ = simulate_baseline(data, verbose=False)
        tot = sum(ops.values())
        res[mode] = {'ops': dict(ops), 'total': tot, 'R': Rm, 'X': Xm}
        nota = '  <- equipamento (3/9)' if mode in ('deployed', '671') else ''
        print(f"  {mode:>11} {Rm:>6.2f} {Xm:>6.2f}  "
              f"{ops.get(1,0):>3} {ops.get(2,0):>3} {ops.get(3,0):>3} {tot:>6}{nota}")
    print("="*64)
    print("  'deployed'=3/9 V do regulador | '671'=centro de carga no tronco")
    print("  (valida o equipamento) | 'centroide'/'675'=sensibilidade.")
    return res


if __name__ == '__main__':
    run_baseline()
