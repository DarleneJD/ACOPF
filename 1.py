# -*- coding: utf-8 -*-
"""
================================================================================
  OPF BFM-SOCP — Branch Flow Model com Restrição Cônica de Segunda Ordem
  IEEE-13 Modificado | SVR Monofásico | Volt-VAR PV | 144 × 10 min
================================================================================
  Formulação:
    - Branch Flow Model (Baran & Wu 1989; Farivar & Low 2013)
    - Relaxação SOCP: l_sq * V_i >= P_ij² + Q_ij²  (cone de Lorentz)
    - Perdas r·l_sq e x·l_sq nas restrições de balanço (não na F.O.)
    - Tap SVR discreto (MISOCP): tap_pos∈{0..32}, tap_pu∈[1.0000,1.2000] p.u.
    - Binárias u_tap detectam eventos mecânicos (big-M)
    - Solver: CPLEX com SOCP nativo

  Solver:
    opt = SolverFactory('cplex',
              executable='/opt/ibm/ILOG/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex')
    results = opt.solve(model, tee=True).write()
================================================================================
"""

import math, sys, time
import pyomo.environ as pyo

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pip install pandas openpyxl numpy")

# ══════════════════════════════════════════════════════════════════════════════
# PARÂMETROS GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════

DT_MIN     = 10                      # minutos por período
N_PERIODS  = 144                     # 24h × 60min / 10min
HOURS      = list(range(N_PERIODS))

T_VV_START = 42                      # 07:00
T_VV_END   = 108                     # 18:00

# ═══════════════════════════════════════════════════════════════════════════
# PERFIS TEMPORAIS — 144 pontos diretos (Δt = 10 min, 24h)
# ═══════════════════════════════════════════════════════════════════════════
# Todos os perfis vêm de medições reais com resolução de 10 min, dispensando
# a etapa de interpolação. Os valores são fatores adimensionais em [0,1] que
# multiplicam os valores nominais declarados na planilha (kW, kVAr).
#
# Fontes:
#   TECHNICH BRASIL — curva de carga industrial/comercial MT característica
#                      (baseload elevado, range ≈ 0,71–1,00, pico às 16-17h)
#   curvaR          — curva de carga residencial BT (pico vespertino,
#                      range ≈ 0,19–1,00, máximo às 18-19h)
#   temp            — temperatura ambiente [°C] (perfil diário medido)
#   irrad           — irradiância global normalizada [W/m²/STC]
# ═══════════════════════════════════════════════════════════════════════════

# Curva TECHNICH BRASIL — cargas MT (perfil industrial/comercial real)
TECHNICHDOBRASIL = [
    0.909, 0.909, 0.909, 0.909, 0.909, 0.909,  # 00:00–01:00
    0.721, 0.721, 0.721, 0.721, 0.721, 0.721,  # 01:00–02:00
    0.714, 0.714, 0.714, 0.714, 0.714, 0.714,  # 02:00–03:00
    0.721, 0.721, 0.721, 0.721, 0.721, 0.721,  # 03:00–04:00
    0.718, 0.718, 0.718, 0.718, 0.718, 0.718,  # 04:00–05:00
    0.755, 0.755, 0.755, 0.755, 0.755, 0.755,  # 05:00–06:00
    0.761, 0.761, 0.761, 0.761, 0.761, 0.761,  # 06:00–07:00
    0.934, 0.934, 0.934, 0.934, 0.934, 0.934,  # 07:00–08:00
    1.000, 1.000, 1.000, 1.000, 1.000, 1.000,  # 08:00–09:00
    0.986, 0.986, 0.986, 0.986, 0.986, 0.986,  # 09:00–10:00
    0.992, 0.992, 0.992, 0.992, 0.992, 0.992,  # 10:00–11:00
    0.883, 0.883, 0.883, 0.883, 0.883, 0.883,  # 11:00–12:00
    0.783, 0.783, 0.783, 0.783, 0.783, 0.783,  # 12:00–13:00
    0.780, 0.780, 0.780, 0.780, 0.780, 0.780,  # 13:00–14:00
    0.934, 0.934, 0.934, 0.934, 0.934, 0.934,  # 14:00–15:00
    0.919, 0.919, 0.919, 0.919, 0.919, 0.919,  # 15:00–16:00
    0.738, 0.738, 0.738, 0.738, 0.738, 0.738,  # 16:00–17:00
    0.893, 0.893, 0.893, 0.893, 0.893, 0.893,  # 17:00–18:00
    0.889, 0.889, 0.889, 0.889, 0.889, 0.889,  # 18:00–19:00
    0.898, 0.898, 0.898, 0.898, 0.898, 0.898,  # 19:00–20:00
    0.885, 0.885, 0.885, 0.885, 0.885, 0.885,  # 20:00–21:00
    0.792, 0.792, 0.792, 0.792, 0.792, 0.792,  # 21:00–22:00
    0.817, 0.817, 0.817, 0.817, 0.817, 0.817,  # 22:00–23:00
    0.926, 0.926, 0.926, 0.926, 0.926, 0.926,  # 23:00–24:00
]
assert len(TECHNICHDOBRASIL) == N_PERIODS, "TECHNICHDOBRASIL deve ter 144 pontos"

# Curva R — cargas BT residenciais (pico vespertino)
CURVA_R = [
    0.289, 0.307, 0.319, 0.330, 0.342, 0.362,  # 00:00–01:00
    0.206, 0.223, 0.235, 0.246, 0.258, 0.278,  # 01:00–02:00
    0.190, 0.207, 0.219, 0.230, 0.242, 0.262,  # 02:00–03:00
    0.185, 0.202, 0.214, 0.225, 0.237, 0.257,  # 03:00–04:00
    0.190, 0.207, 0.219, 0.230, 0.242, 0.262,  # 04:00–05:00
    0.378, 0.396, 0.408, 0.419, 0.431, 0.451,  # 05:00–06:00
    0.488, 0.506, 0.518, 0.529, 0.541, 0.560,  # 06:00–07:00
    0.498, 0.516, 0.528, 0.539, 0.552, 0.571,  # 07:00–08:00
    0.493, 0.511, 0.523, 0.534, 0.546, 0.566,  # 08:00–09:00
    0.383, 0.401, 0.413, 0.424, 0.436, 0.456,  # 09:00–10:00
    0.415, 0.433, 0.444, 0.455, 0.468, 0.487,  # 10:00–11:00
    0.504, 0.522, 0.534, 0.545, 0.557, 0.576,  # 11:00–12:00
    0.472, 0.490, 0.502, 0.513, 0.525, 0.545,  # 12:00–13:00
    0.378, 0.396, 0.408, 0.419, 0.431, 0.451,  # 13:00–14:00
    0.415, 0.433, 0.444, 0.455, 0.468, 0.487,  # 14:00–15:00
    0.389, 0.406, 0.418, 0.429, 0.442, 0.461,  # 15:00–16:00
    0.425, 0.443, 0.455, 0.466, 0.478, 0.498,  # 16:00–17:00
    0.713, 0.731, 0.743, 0.754, 0.766, 0.785,  # 17:00–18:00
    0.896, 0.914, 0.926, 0.937, 0.950, 0.969,  # 18:00–19:00 PICO
    0.880, 0.899, 0.911, 0.922, 0.934, 0.953,  # 19:00–20:00
    0.928, 0.946, 0.958, 0.969, 0.981, 1.000,  # 20:00–21:00 PICO
    0.786, 0.804, 0.816, 0.827, 0.840, 0.859,  # 21:00–22:00
    0.744, 0.762, 0.775, 0.786, 0.798, 0.817,  # 22:00–23:00
    0.462, 0.480, 0.492, 0.503, 0.515, 0.534,  # 23:00–24:00
]
assert len(CURVA_R) == N_PERIODS, "CURVA_R deve ter 144 pontos"

# Temperatura ambiente [°C] — perfil diário medido (mesma data dos demais)
TEMP_AMB = [
    15.690, 19.500, 19.320, 19.200, 19.200, 19.200,
    18.820, 18.820, 18.770, 18.800, 18.800, 18.660,
    18.500, 18.400, 18.390, 18.320, 18.300, 18.200,
    18.200, 18.240, 16.390, 18.180, 18.100, 18.210,
    18.200, 18.280, 18.470, 18.640, 18.700, 18.643,
    18.706, 18.749, 18.911, 19.123, 19.477, 15.417,
    15.417, 15.417, 15.417, 30.553, 30.553, 30.553,
    30.553, 30.553, 33.090, 37.848, 36.005, 35.705,
    40.833, 42.616, 44.289, 46.339, 47.948, 49.419,
    50.901, 52.576, 54.942, 56.025, 57.023, 58.524,
    59.512, 60.533, 61.751, 62.310, 64.072, 59.398,
    51.125, 52.483, 67.489, 58.468, 62.535, 69.999,
    70.013, 64.843, 66.918, 57.740, 53.172, 69.220,
    69.173, 67.697, 72.019, 70.477, 59.035, 46.033,
    52.210, 44.826, 42.819, 57.976, 57.766, 56.095,
    55.925, 54.838, 53.272, 50.641, 48.424, 47.706,
    46.446, 44.841, 42.187, 40.413, 39.151, 34.869,
    31.966, 33.879, 33.068, 30.352, 29.152, 29.156,
    26.738, 25.957, 25.261, 24.705, 24.523, 24.077,
    23.927, 23.801, 23.760, 23.610, 23.390, 23.200,
    23.210, 23.300, 23.390, 23.200, 22.980, 22.900,
    22.900, 22.880, 22.960, 23.080, 23.300, 23.420,
    23.510, 23.540, 23.630, 23.860, 23.860, 24.030,
    23.790, 23.830, 23.900, 23.930, 23.830, 23.710,
]
assert len(TEMP_AMB) == N_PERIODS, "TEMP_AMB deve ter 144 pontos"

# Irradiância normalizada (G_t/G_STC) — medida no plano dos painéis
# Inclui efeitos de transitórios atmosféricos (nuvens) visíveis em t=66-90
IRRAD = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000008, 0.000080,
    0.000672, 0.002286, 0.007955, 0.015941, 0.022688, 0.037534,
    0.037534, 0.037534, 0.037534, 0.037534, 0.228267, 0.228267,
    0.228267, 0.228267, 0.303663, 0.403642, 0.340885, 0.338831,
    0.475905, 0.497985, 0.531826, 0.571495, 0.598836, 0.629815,
    0.658507, 0.687815, 0.723270, 0.741506, 0.768072, 0.791615,
    0.811002, 0.833043, 0.847065, 0.865453, 0.889556, 0.909654,
    0.535304, 0.609437, 0.972867, 0.718562, 0.821835, 1.000000,
    0.995747, 0.853749, 0.911717, 0.662080, 0.551868, 0.967518,
    0.949435, 0.892242, 0.972875, 0.919104, 0.651704, 0.353381,
    0.547719, 0.361383, 0.326927, 0.700342, 0.667077, 0.627656,
    0.619742, 0.586301, 0.553395, 0.502454, 0.448827, 0.442983,
    0.400269, 0.374286, 0.326104, 0.297339, 0.269415, 0.237629,
    0.210439, 0.169579, 0.153966, 0.088307, 0.071950, 0.087652,
    0.038829, 0.023192, 0.012304, 0.004485, 0.002127, 0.000696,
    0.000176, 0.000016, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
]
assert len(IRRAD) == N_PERIODS, "IRRAD deve ter 144 pontos"

# ═══════════════════════════════════════════════════════════════════════════
# MODELO PV COM TEMPERATURA DA CÉLULA — NREL/SAM (Sandia)
# ═══════════════════════════════════════════════════════════════════════════
# A potência efetiva do painel PV depende da irradiância E da temperatura:
#
#   T_cell = T_amb + (G_t / G_STC) · (NOCT - 20)         [°C]
#   P_eff  = P_STC · (G_t/G_STC) · [1 + γ·(T_cell - 25)] [kW]
#
# Onde:
#   NOCT = 45°C   (Nominal Operating Cell Temperature, valor típico)
#   γ    = -0.0040 /°C (coef. térmico de potência, típico de Si cristalino)
#   G_STC= 1000 W/m² (irradiância de referência)
#
# A temperatura da célula reduz a potência ativa: para T_cell=60°C,
# a perda é (60-25)·0.004 = 14% em relação ao STC.
# Fonte: Skoplaki & Palyvos (2009); IEC 61853-2
NOCT          = 45.0     # °C — temperatura nominal de operação da célula
GAMMA_PV      = -0.0040  # /°C — coeficiente térmico (Si cristalino típico)
T_STC         = 25.0     # °C — temperatura de referência STC
G_STC_NORM    = 1.0      # irradiância STC normalizada (G_t já vem em p.u.)

def pv_temperature_derate(irr_norm, t_amb_c):
    """Calcula T_cell e fator de derate térmico do painel PV.

    Args:
        irr_norm: irradiância normalizada (G_t/G_STC), em [0,1]
        t_amb_c:  temperatura ambiente [°C]
    Returns:
        (t_cell, derate): T_cell em °C e fator multiplicativo de P_avail
    """
    t_cell = t_amb_c + irr_norm * (NOCT - 20.0)
    derate = 1.0 + GAMMA_PV * (t_cell - T_STC)
    return t_cell, max(derate, 0.5)    # piso 50% para evitar valores absurdos

# Vetor Pavail_factor por período: P_eff/P_nom = irrad·derate(irrad, T_amb)
PV_EFF_FACTOR = []
PV_T_CELL     = []
for _k in range(N_PERIODS):
    _t_cell, _derate = pv_temperature_derate(IRRAD[_k], TEMP_AMB[_k])
    PV_T_CELL.append(_t_cell)
    PV_EFF_FACTOR.append(IRRAD[_k] * _derate)

# ═══════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE PERFIS DE CARGA POR BARRA
# ═══════════════════════════════════════════════════════════════════════════
# MT (Load_MT do xlsx) → curva TECHNICH BRASIL
# BT (laterais BT)     → curva R (residencial)
# Pode-se sobrepor por barra específica via _MT_LOAD_PROFILE_MAP se necessário
IRR_PROFILE      = IRRAD              # alias (compatibilidade)
LOAD_PROFILE     = CURVA_R            # default BT — residencial
LOAD_PROFILE_MT  = TECHNICHDOBRASIL   # default MT — technichdobrasil
LOAD_INDUSTRIAL  = TECHNICHDOBRASIL
LOAD_COMERCIAL   = TECHNICHDOBRASIL
LOAD_RESIDENCIAL = CURVA_R

# Mapeamento opcional barra → tipo (vazio = usa default MT/BT por level)
_MT_LOAD_PROFILE_MAP = {}   # vazio: todas as barras MT usam TECHNICHDOBRASIL
_LOAD_BY_TYPE = {
    'industrial':  TECHNICHDOBRASIL,
    'comercial':   TECHNICHDOBRASIL,
    'residencial': CURVA_R,
    'mt_default':  TECHNICHDOBRASIL,
    'bt_default':  CURVA_R,
}

# ═══════════════════════════════════════════════════════════════════════════
# PESOS DA FUNÇÃO OBJETIVO — Hierarquia ajustada
# ═══════════════════════════════════════════════════════════════════════════
# A hierarquia desejada é:
#   W_curt    >> W_dv > W_unbal > W_tap_ops > W_ramp
# Significado:
#   1) Curtailment é último recurso (peso máximo)
#   2) Desvio |V - V_ref| é a âncora principal de qualidade de tensão
#   3) Desequilíbrio entre fases é secundário
#   4) Operações de tap são preferencialmente evitadas em favor de Qpv,
#      mas autorizadas quando Qpv não basta
#   5) Suavização de Qpv é fina (não compete com objetivos maiores)
W_DV       = 10.0   # desvio |V - V_ref| — âncora principal (PRIORIDADE 1)
              # V_ref = 0.95 p.u. nas barras MT com carga modelo 1 (P-const);
              # V_ref = 1.0167 p.u. nas demais (modelos 2/5 ou sem carga MT)
W_UNBAL    = 3.3    # desequilíbrio |Vi - Vj| (PRIORIDADE 2)
W_TAP_OPS  = 10    # eventos mecânicos de tap (PRIORIDADE 3)
              # ↓ DRASTICAMENTE REDUZIDO (era 10.0) para permitir que
              # o tap atue quando Qpv não conseguir corrigir a tensão.
              # Solver usa primeiro Qpv (custo zero); aciona tap apenas
              # se a violação de tensão persistir após esgotar Qpv.
W_RAMP_DIA  = 0.10  # suavização ΔQpv (fora pico solar)
W_RAMP_PICO = 0.01  # suavização ΔQpv (pico solar)
IRR_PICO   = 0.50   # threshold pico solar
W_CURT     = 50.0   # curtailment PV — preservar geração disponível (TOPO)

# Controle
FP_MIN_VV  = 0.85   # fator de potência mínimo (IEEE 1547 Cat B)
# Operações de tap por fase: limite físico razoável.
# O número emerge da hierarquia W_dv > W_tap_ops, mas com um teto sensato:
# 12 ops/fase = 1 operação a cada 2 horas em média, compatível com a vida
# útil mecânica do comutador IEEE Cooper típico em operação normal.
# Acima disso, a baseline OpenDSS (33-39 ops/dia totais, ≈12/fase) já
# indica que o sistema está em estresse.
N_TAP_MAX  = 12     # ops máximas por fase no horizonte de 24h

# ═══════════════════════════════════════════════════════════════════════════
# BASES DO SISTEMA — por nível de tensão (linha-neutro)
# ═══════════════════════════════════════════════════════════════════════════
# Conforme Kersting (Distribution System Modeling and Analysis, cap. 6) e
# Bergen & Vittal (Power Systems Analysis), em sistemas multinível com
# transformadores, adota-se:
#   - S_base ÚNICA para todo o sistema (kVA/fase)
#   - V_base por nível de tensão (lado do transformador), linha-neutro
#   - Z_base = V_base² / S_base — única por nível de tensão
# A configuração do circuito (trifásico, bifásico, monofásico) NÃO altera a
# Z_base; afeta apenas o número de fases ativas e o conjunto de equações.
SBASE = 1000.0 / 3.0                  # kVA/fase
VMT   = 4.16  / math.sqrt(3)          # kV (linha-neutro, MT)
VBT   = 0.480 / math.sqrt(3)          # kV (linha-neutro, BT)
ZMT   = VMT**2 / SBASE * 1000         # Ω (única para todos os ramos MT)
ZBT   = VBT**2 / SBASE * 1000         # Ω (única para todos os ramos BT)
IMT   = SBASE / VMT                   # kA
IBT   = SBASE / VBT                   # kA
# Aliases retrocompatíveis (manter código existente funcionando)
VBT3 = VBT; ZBT3 = ZBT; IBT3 = IBT
VBT1 = VBT; ZBT1 = ZBT; IBT1 = IBT   # mesma base — sem distinção por nº de fases

# Limites PRODIST
FD_MT_MAX    = 2.0   # %
FD_BT_MAX    = 3.0   # %
K_FD_VUF     = math.sqrt(3) / 6
VUF_MAX_MT   = FD_MT_MAX / (K_FD_VUF * 100)   # ≈ 0.0693 p.u.
VUF_MAX_BT   = FD_BT_MAX / (K_FD_VUF * 100)   # ≈ 0.1039 p.u.
FD_ANEEL_MAX = FD_MT_MAX                        # alias — limite mais restritivo

TR_MT_NORMAL = 1.00
TR_MT_M25    = 0.95
V_ADEQ_MT_LO = 0.93 * TR_MT_NORMAL
V_ADEQ_MT_HI = 1.05 * TR_MT_NORMAL
V_PREC_MT_LO = 0.90 * TR_MT_NORMAL
V_ADEQ_BT_LO = 0.93
V_ADEQ_BT_HI = 1.05
V_PREC_BT_LO = 0.90

# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def _norm(val):
    s = str(val).strip()
    return s[:-2] if s.endswith('.0') else s

def _f(val, default=0.0):
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except:
        return default

def _phases(val):
    out = []
    for p in str(val).strip().replace('.', ',').split(','):
        try:
            ph = int(float(p.strip()))
            if ph in (1, 2, 3): out.append(ph)
        except:
            pass
    return sorted(set(out))

# ══════════════════════════════════════════════════════════════════════════════
# LEITURA DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(buses_file, branches_file):
    LOAD_EXP_P = {1: 0.0, 2: 2.0, 3: 0.0, 4: 1.0, 5: 1.0}
    LOAD_EXP_Q = {1: 0.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 1.0}

    df = pd.read_excel(buses_file, sheet_name='MT')
    df.columns = df.columns.str.strip()
    df = df[~df['name'].astype(str).str.contains('Source', case=False)]

    buses = {}; slack = None
    for _, r in df.iterrows():
        n = _norm(r['name']); tb = int(_f(r.get('tb', 0), 0))
        buses[n] = {'level': 'mv', 'v_nom_kv': VMT,
                    'phases': _phases(r.get('phases', '1,2,3')), 'tb': tb}
        if tb == 3 and slack is None:
            slack = n

    # Cargas MT
    loads_mt = {}
    try:
        df_lmt = pd.read_excel(buses_file, sheet_name='Load_MT')
        df_lmt.columns = df_lmt.columns.str.strip()
        for _, r in df_lmt.iterrows():
            n   = _norm(r['name'])
            phs = _phases(r.get('phases', '1,2,3'))
            nph = max(len(phs), 1)
            P   = _f(r.get('P_D'), 0.)
            Q   = _f(r.get('Q_D'), 0.)
            mod = int(_f(r.get('Model', 1), 1))
            conn = str(r.get('Conn', 'Wye')).strip().lower()
            # Cargas delta -- Simplificacao S5b:
            # ZY=ZD/3 invalida em sistema desequilibrado (assume Vab=Vbc=Vca).
            # Correcao exata nao-linear; S5b usa potencia de linha sem fator /3.
            is_delta = ('delta' in conn or conn.strip().lower() == 'd')
            for ph in phs:
                P_ph = P / nph / SBASE
                Q_ph = Q / nph / SBASE
                loads_mt[(n, ph)] = {
                    'P_pu':     P_ph,
                    'Q_pu':     Q_ph,
                    'model':    mod,
                    'conn':     conn,
                    'is_delta': is_delta,
                    'aP':       LOAD_EXP_P.get(mod, 0.0),
                    'aQ':       LOAD_EXP_Q.get(mod, 0.0),
                }
    except Exception as e:
        loads_mt = {}
        print(f"  Load_MT: {e}")

    # Cargas BT
    df = pd.read_excel(buses_file, sheet_name='BT')
    df.columns = df.columns.str.strip()
    loads = {}
    for _, r in df.iterrows():
        n = _norm(r['name']); phs = _phases(r.get('phases', '1,2,3'))
        nph = max(len(phs), 1); Vbt = VBT3 if nph > 1 else VBT1
        buses[n] = {'level': 'lv', 'v_nom_kv': Vbt, 'phases': phs, 'tb': 0}
        P_tot = _f(r.get('P_D'), 0.); Q_tot = _f(r.get('Q_D'), 0.)
        for ph in phs:
            # Suportar cargas por fase (colunas P_ph1/P_ph2/P_ph3 no xlsx)
            # Se não existirem, dividir uniformemente por nph
            P_ph = _f(r.get(f'P_ph{ph}'), P_tot / nph)
            Q_ph = _f(r.get(f'Q_ph{ph}'), Q_tot / nph)
            loads[(n, ph)] = {'P_pu': P_ph / SBASE,
                              'Q_pu': Q_ph / SBASE,
                              'model': 1, 'aP': 0.0, 'aQ': 0.0}

    for n, d in buses.items():
        if d['level'] == 'mv':
            for ph in d['phases']:
                if (n, ph) not in loads:
                    loads[(n, ph)] = loads_mt.get(
                        (n, ph),
                        {'P_pu': 0., 'Q_pu': 0., 'model': 1, 'aP': 0.0, 'aQ': 0.0})

    # PV — distinguir inversores trifásicos (parâmetros iguais por fase)
    # e monofásicos (parâmetros independentes por fase)
    # Inversor trifásico: usa kva/3 para cada fase (mesmo dispositivo);
    #   a potência nominal é comum → P_rated = S_nom/nph para as 3 fases.
    # Inversor monofásico: nph=1, parâmetros lidos individualmente.
    # Coluna 'type' na planilha PV: '3ph' (padrão) ou '1ph'.
    df = pd.read_excel(buses_file, sheet_name='PV')
    df.columns = df.columns.str.strip()
    pv = {}; pv_meta = {}   # pv_meta: metadados para pós-processamento
    for _, r in df.iterrows():
        bus  = _norm(r['Bus'])
        if bus not in buses: continue
        phs  = _phases(r.get('phases', '1,2,3'))
        nph  = max(len(phs), 1)
        kva  = _f(r.get('kva'), 0.)
        Sf   = kva / nph            # potência por fase do inversor
        ptype = str(r.get('type', '3ph')).strip().lower()
        is3ph = (nph == 3) or ('3' in ptype)  # True: trifásico
        if is3ph:
            # Trifásico: potência dividida igualmente, mesmo S_nom para as 3 fases
            P_rated_com = _f(r.get('p_pv_1'), Sf)  # usa fase 1 como referência
            S_nom_com   = max(Sf, P_rated_com / FP_MIN_VV) if P_rated_com > 0 else Sf
            for ph in phs:
                pv[(bus, ph)] = {'S_nom_pu':   S_nom_com / SBASE,
                                 'P_rated_pu': P_rated_com / SBASE,
                                 'is_3ph': True, 'nph': nph}
            pv_meta[bus] = {'type': '3ph', 'kva': kva, 'phases': phs}
        else:
            # Monofásico: parâmetros independentes por fase
            for ph in phs:
                P_rated = _f(r.get(f'p_pv_{ph}'), Sf)
                S_nom   = max(Sf, P_rated / FP_MIN_VV) if P_rated > 0 else Sf
                pv[(bus, ph)] = {'S_nom_pu':   S_nom / SBASE,
                                 'P_rated_pu': P_rated / SBASE,
                                 'is_3ph': False, 'nph': 1}
            pv_meta[bus] = {'type': '1ph', 'kva': kva, 'phases': phs}

    # SVR
    df = pd.read_excel(branches_file, sheet_name='Reg')
    df.columns = df.columns.str.strip()
    svr = {}
    for _, r in df.iterrows():
        ph = int(_f(r.get('phases'), 1))
        mv = _norm(r['mv_bus']); lv = _norm(r['lv_bus'])
        R  = _f(r.get('R_ohm'), 0.0)
        X  = _f(r.get('X_ohm'), 0.0)
        I  = _f(r.get('ctprim_A'), 700.)
        n_tap    = int(_f(r.get('TapNum'), 33))
        step     = _f(r.get('Step'), 0.00625)
        # Faixa bidirecional padrão IEEE Cooper: ±10% em 32 passos
        # tap_pu ∈ [0.900, 1.100] com posição neutra em pos=16 (1.000 p.u.)
        # Permite tanto buck (descer V) quanto boost (subir V) — essencial
        # para cenários com PV onde o fluxo pode se inverter.
        tap_min  = _f(r.get('Tap_min'), 0.90)
        tap_max  = tap_min + (n_tap - 1) * step      # com step=0.00625: 1.100
        tap_init = _f(r.get('Tap_init'), 1.0)         # posição neutra padrão
        delay_s  = _f(r.get('Delay_s'), 15.)
        d_per    = max(1, math.ceil(delay_s / (DT_MIN * 60.0)))
        # Line Drop Compensation (LDC) — estimar tensão no centro de carga
        # Valores típicos IEEE-13: R_ldc=3.0 Ω, X_ldc=7.5 Ω (em ohms primários)
        # V_load = V_lv - (R_ldc·I_P + X_ldc·I_Q) onde I é normalizado por CT
        R_ldc_ohm = _f(r.get('R_ldc'), 3.0)   # ohms primários (default OpenDSS)
        X_ldc_ohm = _f(r.get('X_ldc'), 9.0)   # ohms primários (default OpenDSS)
        # Bandwidth e referência do controle local (overridable por linha do xlsx)
        v_ref_lc = _f(r.get('Vreg_pu'), 1.0)        # referência em p.u.
        bw_lc    = _f(r.get('BW_pu'),   0.0167)     # 2V/120V em p.u.
        svr[ph]  = {
            'mv_bus': mv, 'lv_bus': lv,
            'r_pu':   R / ZMT, 'x_pu': X / ZMT,
            'Imax_pu': I / IMT,
            'tap_min': tap_min, 'tap_max': tap_max,
            'tap_step': step,   'tap_init': tap_init,
            'n_tap':   n_tap,   'delay_periods': d_per,
            # Parâmetros LDC (em p.u. da base MT)
            'R_ldc_pu':  R_ldc_ohm / ZMT,
            'X_ldc_pu':  X_ldc_ohm / ZMT,
            'V_ref_lc':  v_ref_lc,
            'BW_lc':     bw_lc,
        }
        for b in (mv, lv):
            if b not in buses:
                buses[b] = {'level': 'mv', 'v_nom_kv': VMT,
                            'phases': [1,2,3], 'tb': 0}

    # Ramos MT
    df = pd.read_excel(branches_file, sheet_name='MT')
    df.columns = df.columns.str.strip()
    branches = {}
    for _, r in df.iterrows():
        fr = _norm(r['l']); to = _norm(r['k'])
        if to == 'RG60': fr, to = to, fr
        if 'Source' in (fr, to): continue
        R = _f(r['R']); X = _f(r['X']); I = _f(r.get('Imax', 999))
        for ph in _phases(r.get('phase', '1,2,3')):
            branches[(fr, to, ph)] = {
                'r_pu': R/ZMT, 'x_pu': X/ZMT,
                'Imax_pu': I/IMT, 'level': 'mv'}

    # Ramos BT
    df = pd.read_excel(branches_file, sheet_name='BT')
    df.columns = df.columns.str.strip()
    for _, r in df.iterrows():
        fr = _norm(r['l']); to = _norm(r['k'])
        phs = _phases(r.get('phase', '1,2,3'))
        nph = len(phs)
        Zb = ZBT3 if nph > 1 else ZBT1
        Ib = IBT3 if nph > 1 else IBT1
        R = _f(r['R']); X = _f(r['X']); I = _f(r.get('Imax', 999))
        for ph in phs:
            branches[(fr, to, ph)] = {
                'r_pu': R/Zb, 'x_pu': X/Zb,
                'Imax_pu': I/Ib, 'level': 'lv'}

    # Trafos — armazenar parâmetros térmicos (IEEE C57.110 / C57.91)
    # G_core = P_NL_nom / V_nom² → perda no núcleo p.u. proporcional a V²
    # P_LL_R = perdas de carga nominais (Joule) = r · I_nom²
    # theta_TO_R = elevação nominal do óleo de topo (padrão: 55°C)
    df = pd.read_excel(branches_file, sheet_name='Trafos')
    df.columns = df.columns.str.strip()
    trafos = {}   # {trafo_id: {...}} — metadados térmicos
    for _, r in df.iterrows():
        if str(r['trafo_id']) == 'Sub': continue
        mv = _norm(r['mv_bus']); lv = _norm(r['lv_bus'])
        tid = str(r['trafo_id']).strip()
        phs = _phases(r.get('phases', '1,2,3'))
        nph = max(len(phs), 1)
        Vbt = VBT3 if nph == 3 else VBT1
        Zb = Vbt**2 / SBASE * 1000; Ib = SBASE / Vbt
        kva = _f(r['kva']); Sn = kva / nph
        R = _f(r['R_ohm']); X = _f(r['X_ohm'])
        # Perdas em vazio (núcleo): lidas do xlsx ou estimadas como 0.3% de S_nom
        P_NL_nom_kw  = _f(r.get('P_NL_kW'),  Sn * 0.003)   # kW/fase
        # Perdas de carga nominais: lidas ou estimadas como R·I²
        P_LL_R_kw    = _f(r.get('P_LL_kW'),  Sn * (R/Zb))  # kW/fase
        # Condutância de núcleo em p.u.: G_core = P_NL / (V_nom² · S_base)
        G_core_pu    = (P_NL_nom_kw / SBASE) / 1.0**2       # V_nom=1.0 p.u.
        # Elevação nominal do óleo de topo (°C) — padrão IEEE C57.91
        theta_TO_R   = _f(r.get('theta_TO_R'), 55.0)        # °C
        theta_g_R    = _f(r.get('theta_g_R'),  25.0)        # gradiente nominal
        if lv in buses: buses[lv]['v_nom_kv'] = Vbt
        for ph in phs:
            branches[(mv, lv, ph)] = {
                'r_pu': R/Zb, 'x_pu': X/Zb,
                'Imax_pu': Sn/Vbt/Ib, 'level': 'trafo',
                'G_core_pu': G_core_pu,        # condutância de ferro [p.u.]
                'P_NL_nom_kw': P_NL_nom_kw,    # perda vazio nominal [kW/fase]
                'P_LL_R_kw': P_LL_R_kw,        # perda carga nominal [kW/fase]
                'theta_TO_R': theta_TO_R,       # elevação óleo nominal [°C]
                'theta_g_R': theta_g_R,         # gradiente nominal [°C]
                'trafo_id': tid, 'mv_bus': mv, 'lv_bus': lv,
                'nph': nph}
            trafos[tid] = {
                'mv_bus': mv, 'lv_bus': lv, 'phases': phs, 'nph': nph,
                'kva_fase': Sn, 'G_core_pu': G_core_pu,
                'P_NL_nom_kw': P_NL_nom_kw, 'P_LL_R_kw': P_LL_R_kw,
                'theta_TO_R': theta_TO_R,    'theta_g_R': theta_g_R}

    # Completar barras ausentes
    for (fr, to, ph) in list(branches.keys()):
        for b in (fr, to):
            if b not in buses:
                buses[b] = {'level': 'mv', 'v_nom_kv': VMT,
                            'phases': [1,2,3], 'tb': 0}
            if (b, ph) not in loads:
                loads[(b, ph)] = {'P_pu': 0., 'Q_pu': 0.}

    print(f"  Barras: {len(buses)}  Ramos: {len(branches)}")
    print(f"  PVs: {len(pv)}  SVRs: {len(svr)}  Slack: {slack}")
    return {'buses': buses, 'loads': loads, 'loads_mt': loads_mt,
            'pv': pv, 'pv_meta': pv_meta,
            'branches': branches, 'svr': svr, 'slack': slack,
            'trafos': trafos}


# ══════════════════════════════════════════════════════════════════════════════
# WARM-START (forward sweep para inicialização)
# ══════════════════════════════════════════════════════════════════════════════

def compute_warm_start(data, hours, irr, load):
    from collections import deque, defaultdict
    buses    = data['buses']
    loads    = data['loads']
    pv       = data['pv']
    branches = data['branches']
    svr      = data['svr']
    slack    = data['slack']

    adj = defaultdict(list)
    for (fr, to, ph) in branches:
        adj[(fr, ph)].append((to, ph, (fr, to, ph)))
    for ph, d in svr.items():
        adj[(d['mv_bus'], ph)].append((d['lv_bus'], ph, ('SVR', ph)))

    def bfs(root, rphs):
        order, visited = [], set()
        q = deque([(root, ph) for ph in rphs])
        for item in q: visited.add(item)
        while q:
            node = q.popleft(); order.append(node)
            for (to, tph, br) in adj.get(node, []):
                if (to, tph) not in visited:
                    visited.add((to, tph)); q.append((to, tph))
        return order

    order  = bfs(slack, buses[slack]['phases'])
    parent = {}
    for (b, ph) in order:
        for (to, tph, br) in adj.get((b, ph), []):
            if (to, tph) not in parent:
                parent[(to, tph)] = ((b, ph), br)

    V_ws  = {}; P_ws  = {}; Q_ws  = {}
    I2_ws = {}; Ppv_ws = {}; tap_ws = {}

    for t in hours:
        alpha = load[t]; irr_t = irr[t]
        tap_t = {ph: svr[ph]['tap_init'] for ph in svr}

        P_net = {}; Q_net = {}
        for (b, ph) in order:
            # Selecionar perfil de carga: MT vs BT (override em _MT_LOAD_PROFILE_MAP)
            if b in _MT_LOAD_PROFILE_MAP:
                _aw = _LOAD_BY_TYPE[_MT_LOAD_PROFILE_MAP[b]][t]
            else:
                level = buses.get(b, {}).get('level', 'mv')
                _aw = LOAD_PROFILE_MT[t] if level == 'mv' else LOAD_PROFILE[t]
            Pd   = loads.get((b, ph), {'P_pu': 0})['P_pu'] * _aw
            Qd   = loads.get((b, ph), {'Q_pu': 0})['Q_pu'] * _aw
            # PV com derate térmico (mesma função usada no modelo)
            Ppv = pv.get((b, ph), {'P_rated_pu': 0})['P_rated_pu'] * PV_EFF_FACTOR[t]
            Ppv_ws[(b, ph, t)] = Ppv
            P_net[(b, ph)] = Pd - Ppv
            Q_net[(b, ph)] = Qd

        P_br = {}; Q_br = {}
        for (b, ph) in reversed(order):
            pf = P_net[(b, ph)]; qf = Q_net[(b, ph)]
            for (to, tph, br) in adj.get((b, ph), []):
                pf += P_br.get((to, ph), 0)
                qf += Q_br.get((to, ph), 0)
            P_br[(b, ph)] = pf; Q_br[(b, ph)] = qf

        V_ws[(slack, 1, t)] = 1.0
        V_ws[(slack, 2, t)] = 1.0
        V_ws[(slack, 3, t)] = 1.0

        for (b, ph) in order:
            if b == slack:
                V_ws[(b, ph, t)] = 1.0; continue
            if (b, ph) not in parent:
                V_ws[(b, ph, t)] = 0.97; continue
            (up, uph), br_key = parent[(b, ph)]
            Vup = V_ws.get((up, uph, t), 1.0)
            if br_key == ('SVR', ph):
                tap_ws[(ph, t)] = tap_t[ph]
                V_ws[(b, ph, t)] = tap_t[ph] * Vup
                P_ws[('SVR_P', ph, t)] = P_br.get((b, ph), 0)
                Q_ws[('SVR_Q', ph, t)] = Q_br.get((b, ph), 0)
            else:
                (fr, to, p) = br_key
                r = branches[br_key]['r_pu']
                x = branches[br_key]['x_pu']
                Pf = P_br.get((b, ph), 0)
                Qf = Q_br.get((b, ph), 0)
                # Forward sweep com aproximação BFM: V_j≈V_i - rP - xQ + 0.5·z²·l_sq
                # l_sq estimado: (P²+Q²)/Vup² para o warm-start
                z2_ws = r**2 + x**2
                l_sq_ws = (Pf**2 + Qf**2) / max(Vup**2, 0.64)
                V_ws[(b, ph, t)] = max(
                    Vup - r*Pf - x*Qf + 0.5*z2_ws*l_sq_ws, 0.80)
                P_ws[(fr, to, p, t)] = Pf
                Q_ws[(fr, to, p, t)] = Qf
                Vi2 = max(Vup**2, 0.64)
                I2_ws[(fr, to, p, t)] = (Pf**2 + Qf**2) / Vi2

        for ph in svr:
            if (ph, t) not in tap_ws:
                tap_ws[(ph, t)] = svr[ph]['tap_init']

    return V_ws, P_ws, Q_ws, I2_ws, Ppv_ws, tap_ws


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO MODELO BFM-SOCP
# ══════════════════════════════════════════════════════════════════════════════

def build_opf(data, hours=HOURS):
    buses    = data['buses']
    branches = data['branches']
    loads    = data['loads']
    loads_mt = data.get('loads_mt', {})
    pv       = data['pv']
    svr      = data['svr']
    slack    = data['slack']

    irr  = {t: IRR_PROFILE[t]  for t in hours}
    load = {t: LOAD_PROFILE[t] for t in hours}

    # ── Conjuntos ────────────────────────────────────────────────────────────
    connected_bph = set()
    for (b, ph) in loads:      connected_bph.add((b, ph))
    for (b, ph) in pv:         connected_bph.add((b, ph))
    for (i, j, ph) in branches:
        connected_bph.add((i, ph)); connected_bph.add((j, ph))
    for ph, d in svr.items():
        connected_bph.add((d['mv_bus'], ph))
        connected_bph.add((d['lv_bus'], ph))

    bph_list  = sorted(connected_bph)
    # V_ref por barra — diferenciado pelo modelo ZIP da carga:
    # • PRODIST Módulo 8 estabelece TR = 1.00 p.u. para todas as barras
    #   regulatoriamente; aqui adotamos uma escolha OPERACIONAL para a F.O.
    #   que difere conforme o modelo ZIP da carga MT.
    # • Cargas com característica de POTÊNCIA CONSTANTE (modelo 1) podem
    #   operar em tensão ligeiramente reduzida (≈0,95 p.u.) sem perda de
    #   atendimento, dado que a potência demandada independe da tensão.
    #   Esta redução libera margem operacional para os reguladores de
    #   tensão e reduz perdas Joule a montante.
    # • Cargas com característica de IMPEDÂNCIA CONSTANTE (modelo 2) ou
    #   CORRENTE CONSTANTE (modelo 5), bem como barras sem carga,
    #   permanecem com V_ref = 1,0167 p.u. (alinhado ao setpoint de
    #   controle local do SVR, 122V/120V), preservando a transferência
    #   nominal de potência em barras com esses perfis.
    # • Os limites adequado/precário PRODIST seguem sendo verificados no
    #   pós-processamento independentemente do V_ref operacional.
    _m1_bph = {(b, ph) for (b, ph), ld in loads_mt.items()
                if ld.get('model', 1) == 1}
    # Para a função objetivo: V_ref = 0,95 p.u. nas barras MT com cargas
    # P-const (modelo 1); V_ref = 1,0167 p.u. nas demais (modelos 2/5
    # ou barras sem carga MT).
    _vref = {(b, ph): (0.95 if (b, ph) in _m1_bph else 1.0167)
             for (b, ph) in bph_list}

    cph_list  = sorted(branches.keys())
    pvph_list = sorted((b, ph) for (b, ph) in pv if (b, ph) in connected_bph)
    svr_phs   = sorted(svr.keys())
    svr_branches = {(svr[ph]['mv_bus'], svr[ph]['lv_bus'], ph) for ph in svr_phs}

    step_    = svr[svr_phs[0]]['tap_step']
    n_pos    = svr[svr_phs[0]]['n_tap']
    tap_min  = svr[svr_phs[0]]['tap_min']
    tap_max  = svr[svr_phs[0]]['tap_max']
    d_delay  = svr[svr_phs[0]]['delay_periods']

    total_P = sum(v['P_pu'] for (b, ph), v in loads.items()) * max(LOAD_PROFILE)
    P_bnd   = max(total_P * 3.0, 2.0)
    Q_bnd   = P_bnd
    _tan_fp = math.sqrt(1.0 - FP_MIN_VV**2) / FP_MIN_VV

    print(f"\n  BFM-SOCP | BPH={len(bph_list)} CPH={len(cph_list)} "
          f"PVPH={len(pvph_list)} SVR={svr_phs} T={len(hours)}")
    print(f"  Tap: {n_pos} pos  step={step_}  N_TAP_MAX={N_TAP_MAX}")

    # ── Barras trifásicas para VUF ───────────────────────────────────────────
    bph_set = set(bph_list)
    trif_mt = sorted({b for (b, ph) in bph_list
                      if buses.get(b, {}).get('level') == 'mv'
                      and all((b, p) in bph_set for p in [1, 2, 3])})
    trif_bt = sorted({b for (b, ph) in bph_list
                      if buses.get(b, {}).get('level') == 'lv'
                      and all((b, p) in bph_set for p in [1, 2, 3])})
    all_trif   = sorted(set(trif_mt) | set(trif_bt))
    unbal_pairs = [(1,2),(1,3),(2,3)]

    # ── Coeficientes de carga V-dependente ──────────────────────────────────
    _aP = {(b,ph): loads.get((b,ph),{}).get('aP', 0.0) for (b,ph) in bph_list}
    _aQ = {(b,ph): loads.get((b,ph),{}).get('aQ', 0.0) for (b,ph) in bph_list}

    # Perfil de carga por (barra, fase, período):
    #   - Barras MT (level='mv', exceto slack): TECHNICHDOBRASIL
    #   - Barras BT (level='lv'):               CURVA_R (residencial)
    #   - Override por barra: _MT_LOAD_PROFILE_MAP (industrial/comercial/residencial)
    def _bus_profile(b):
        # Override explícito por barra (se mapeado)
        if b in _MT_LOAD_PROFILE_MAP:
            return _LOAD_BY_TYPE[_MT_LOAD_PROFILE_MAP[b]]
        # Default por level: MT → TECHNICH, BT → CURVA_R
        level = buses.get(b, {}).get('level', 'mv')
        if level == 'mv': return LOAD_PROFILE_MT
        return LOAD_PROFILE
    _alpha_dict = {
        (b, ph, t): _bus_profile(b)[t]
        for (b, ph) in bph_list for t in hours}

    # ── Warm-start ───────────────────────────────────────────────────────────
    V_ws, P_ws, Q_ws, I2_ws, Ppv_ws, tap_ws = compute_warm_start(
        data, hours, irr, load)

    # ── Modelo Pyomo ─────────────────────────────────────────────────────────
    m = pyo.ConcreteModel(name='BFM_SOCP_VVC_144x10min')

    # Sets
    m.T      = pyo.Set(initialize=hours, ordered=True)
    m.T_VV   = pyo.Set(initialize=[t for t in hours if T_VV_START <= t <= T_VV_END])
    m.T_NVV  = pyo.Set(initialize=[t for t in hours if t < T_VV_START or t > T_VV_END])
    m.BPH    = pyo.Set(initialize=bph_list,  dimen=2)
    m.CPH    = pyo.Set(initialize=cph_list,  dimen=3)
    m.PVPH   = pyo.Set(initialize=pvph_list, dimen=2)
    m.SVRPH  = pyo.Set(initialize=svr_phs)

    # Parâmetros
    m.r     = pyo.Param(m.CPH, initialize={k: v['r_pu']    for k,v in branches.items()})
    m.x     = pyo.Param(m.CPH, initialize={k: v['x_pu']    for k,v in branches.items()})
    m.Imax  = pyo.Param(m.CPH, initialize={k: v['Imax_pu'] for k,v in branches.items()})
    m.Pd0   = pyo.Param(m.BPH, initialize={(b,ph): loads.get((b,ph),{'P_pu':0})['P_pu']
                                            for (b,ph) in bph_list})
    m.Qd0   = pyo.Param(m.BPH, initialize={(b,ph): loads.get((b,ph),{'Q_pu':0})['Q_pu']
                                            for (b,ph) in bph_list})
    m.alpha = pyo.Param(m.T, initialize=load)   # BT fallback
    m.alpha_bph = pyo.Param(
        m.BPH, m.T,
        initialize={(b,ph,t): _alpha_dict.get((b,ph,t), load[t])
                    for (b,ph) in bph_list for t in hours})
    m.irr   = pyo.Param(m.T, initialize=irr)
    m.Prat  = pyo.Param(m.PVPH, initialize={(b,ph): pv[(b,ph)]['P_rated_pu']
                                             for (b,ph) in pvph_list})
    # Pavail considerando temperatura da célula:
    # P_avail(t) = P_STC · irrad(t) · [1 + γ·(T_cell(t)-25)]
    # T_cell(t)  = T_amb(t) + irrad(t)·(NOCT-20)
    # O fator PV_EFF_FACTOR[t] = irrad[t] · derate(t) já encapsula esta relação.
    m.Pavail = pyo.Param(m.PVPH, m.T,
                         initialize={(b,ph,t): pv[(b,ph)]['P_rated_pu']
                                                * PV_EFF_FACTOR[t]
                                     for (b,ph) in pvph_list for t in hours})
    m.Vref   = pyo.Param(m.BPH,
                         initialize={(b,ph): _vref[(b,ph)]
                                     for (b,ph) in bph_list})

    # ── VARIÁVEIS ─────────────────────────────────────────────────────────────

    # Tensão V[b,ph,t] em p.u.
    svr_lv = {svr[ph]['lv_bus'] for ph in svr_phs}
    # _m25 removido — V_ref agora é único (1.0 p.u.) conforme PRODIST

    def v_bounds(m, b, ph, t):
        # Bounds conforme PRODIST Módulo 8 — LIMITE ADEQUADO como teto:
        # - Slack: fixo em 1.0 p.u.
        # - Secundário do SVR (RG60): [0.95, 1.05] — o tap pode estar em
        #   qualquer posição física, mas a tensão resultante respeita o
        #   limite adequado regulatório. Isto evita que o solver use
        #   tap_max=1.10 como artifício para acomodar quedas de tensão
        #   nas pontas, gerando sobretensão na cabeça do alimentador.
        # - Barras MT (1 < V < 69 kV): adequado [0.93, 1.05]
        # - Barras BT (V ≤ 1 kV):       adequado [0.92, 1.05]
        # Bound estrito no limite adequado força o solver a entregar
        # solução em conformidade. Se ficar infactível, é diagnóstico:
        # o sistema não consegue atender PRODIST com a configuração atual.
        if b == slack:                              return (1.0, 1.0)
        if b in svr_lv:                             return (0.95, 1.05)
        if buses.get(b,{}).get('level') == 'mv':    return (0.93, 1.05)
        return (0.92, 1.05)   # BT: adequado PRODIST

    m.V = pyo.Var(m.BPH, m.T,
                  domain=pyo.NonNegativeReals,
                  bounds=v_bounds,
                  initialize=lambda m,b,ph,t: V_ws.get((b,ph,t), 0.97))

    # Fluxos P[i,j,ph,t] e Q[i,j,ph,t]
    m.P = pyo.Var(m.CPH, m.T, domain=pyo.Reals,
                  bounds=(-P_bnd, P_bnd),
                  initialize=lambda m,i,j,ph,t: P_ws.get((i,j,ph,t), 0.0))
    m.Q = pyo.Var(m.CPH, m.T, domain=pyo.Reals,
                  bounds=(-Q_bnd, Q_bnd),
                  initialize=lambda m,i,j,ph,t: Q_ws.get((i,j,ph,t), 0.0))

    # ── l_sq[i,j,ph,t] = |I_ij|²  —  VARIÁVEL CENTRAL DO BFM ────────────────
    # A restrição cônica  l_sq · V_i² ≥ P² + Q²  (cone de Lorentz)
    # é tratada pelo CPLEX como SOCP nativo (exato para redes radiais).
    # Em redes radiais com objetivo de minimização, a relaxação é ativa
    # na solução ótima: Farivar & Low (2013), Teorema 1.
    # Bound generoso: (P_bnd²+Q_bnd²)/V_min²
    # Imax_pu² subdimensionado para o tronco 650→632 no pico PV (~49 vs 27.7)
    # O solver BFM-SOCP impõe l_sq=(P²+Q²)/V² na solução — sem perda de exatidão.
    _lsq_ub = (P_bnd**2 + Q_bnd**2) / 0.64   # V_min=0.8 p.u.
    m.l_sq = pyo.Var(m.CPH, m.T,
                     domain=pyo.NonNegativeReals,
                     bounds=(0.0, _lsq_ub),
                     initialize=lambda m,i,j,ph,t: I2_ws.get((i,j,ph,t), 0.0))

    # PV: potência reativa Qpv e curtailment Pcurt
    m.Qpv  = pyo.Var(m.PVPH, m.T, domain=pyo.Reals,
                     bounds=lambda m,b,ph,t: (
                         -pv[(b,ph)]['P_rated_pu']*_tan_fp,
                          pv[(b,ph)]['P_rated_pu']*_tan_fp),
                     initialize=0.0)
    m.Pcurt = pyo.Var(m.PVPH, m.T, domain=pyo.NonNegativeReals,
                      bounds=lambda m,b,ph,t: (0.0, pv[(b,ph)]['P_rated_pu']),
                      initialize=0.0)

    # Desvio de tensão linearizado |V - V_ref|
    # dv_pos[b,ph,t] >= V[b,ph,t] - Vref[b,ph]
    # dv_neg[b,ph,t] >= Vref[b,ph] - V[b,ph,t]
    _dv_idx = [(b,ph,t) for (b,ph) in bph_list
               for t in hours if b != slack]
    m.dv_pos = pyo.Var(_dv_idx, domain=pyo.NonNegativeReals,
                        bounds=(0, 0.50), initialize=0.)
    m.dv_neg = pyo.Var(_dv_idx, domain=pyo.NonNegativeReals,
                        bounds=(0, 0.50), initialize=0.)

    # SVR: tap discreto e binária de evento
    m.tap_pos = pyo.Var(m.SVRPH, m.T,
                        domain=pyo.NonNegativeIntegers,
                        bounds=(0, n_pos-1),
                        initialize=lambda m,ph,t: int(round(
                            (svr[ph]['tap_init'] - tap_min) / step_)))
    m.u_tap   = pyo.Var(m.SVRPH, m.T, domain=pyo.Binary, initialize=0)

    # SVR: fluxo de potência no regulador
    m.P_svr = pyo.Var(m.SVRPH, m.T, domain=pyo.Reals,
                      bounds=(-P_bnd, P_bnd),
                      initialize=lambda m,ph,t: P_ws.get(('SVR_P',ph,t), 0.0))
    m.Q_svr = pyo.Var(m.SVRPH, m.T, domain=pyo.Reals,
                      bounds=(-Q_bnd, Q_bnd),
                      initialize=lambda m,ph,t: Q_ws.get(('SVR_Q',ph,t), 0.0))

    # Desequilíbrio linearizado: |Vi - Vj|
    _ub_vuf = max(VUF_MAX_MT, VUF_MAX_BT) * math.sqrt(3)
    m.unbal = pyo.Var(
        [(b,ph1,ph2,t) for b in all_trif
         for (ph1,ph2) in unbal_pairs for t in hours],
        domain=pyo.NonNegativeReals,
        bounds=(0, _ub_vuf),
        initialize=0.0)

    # Rampa ΔQpv
    _ramp_bnd = max(pv[(b,ph)]['P_rated_pu']*_tan_fp for (b,ph) in pvph_list) if pvph_list else 0.1
    m.ramp_pos = pyo.Var(m.PVPH, m.T_VV, domain=pyo.NonNegativeReals,
                         bounds=(0, _ramp_bnd*2), initialize=0.0)
    m.ramp_neg = pyo.Var(m.PVPH, m.T_VV, domain=pyo.NonNegativeReals,
                         bounds=(0, _ramp_bnd*2), initialize=0.0)

    # ── RESTRIÇÕES ────────────────────────────────────────────────────────────

    # R1: Tensão fixa no slack
    m.c_slack = pyo.Constraint(
        m.SVRPH, m.T,
        rule=lambda m,ph,t: m.V[slack,ph,t] == 1.0
        if (slack,ph) in connected_bph else pyo.Constraint.Skip)

    # ── R2: Balanço de potência ativa — BFM EXATO ─────────────────────────────
    # P_entrada_líquida = P_ij - r_ij·l_sq_ij  (perdas descontadas no ramo)
    # Referência: Farivar & Low (2013), eq.(2); Baran & Wu (1989)
    def pred(b, ph): return [(i,j,p) for (i,j,p) in branches if j==b and p==ph]
    def succ(b, ph): return [(i,j,p) for (i,j,p) in branches if i==b and p==ph]

    def bal_P_rule(m, b, ph, t):
        if b == slack: return pyo.Constraint.Skip
        ps = pred(b, ph); ss = succ(b, ph)
        svr_in  = [p for p,d in svr.items() if d['lv_bus']==b and p==ph]
        svr_out = [p for p,d in svr.items() if d['mv_bus']==b and p==ph]
        if not ps and not ss and not svr_in and not svr_out and (b,ph) not in pvph_list:
            return pyo.Constraint.Skip
        # BFM: entrada líquida = P_ij - r_ij·l_sq_ij  (perdas no ramo)
        in_P  = sum(m.P[i,j,p,t] - m.r[i,j,p]*m.l_sq[i,j,p,t] for (i,j,p) in ps)
        out_P = sum(m.P[i,j,p,t] for (i,j,p) in ss)
        svr_i = sum(m.P_svr[p,t] for p in svr_in)
        svr_o = sum(m.P_svr[p,t] for p in svr_out)
        Ppv   = (m.Pavail[b,ph,t] - m.Pcurt[b,ph,t]) if (b,ph) in pvph_list else 0.0
        aP_v  = _aP.get((b,ph), 0.0)
        Pd    = (m.Pd0[b,ph]*m.alpha_bph[b,ph,t]*(1.0 + aP_v*(m.V[b,ph,t]-1.0))
                 if aP_v != 0.0 else m.Pd0[b,ph]*m.alpha_bph[b,ph,t])
        return in_P - out_P + Ppv + svr_i - svr_o == Pd
    m.c_bal_P = pyo.Constraint(m.BPH, m.T, rule=bal_P_rule)

    # ── R3: Balanço de potência reativa — BFM EXATO ───────────────────────────
    # Q_entrada_líquida = Q_ij - x_ij·l_sq_ij  (perdas reativas no ramo)
    def bal_Q_rule(m, b, ph, t):
        if b == slack: return pyo.Constraint.Skip
        ps = pred(b, ph); ss = succ(b, ph)
        svr_in  = [p for p,d in svr.items() if d['lv_bus']==b and p==ph]
        svr_out = [p for p,d in svr.items() if d['mv_bus']==b and p==ph]
        if not ps and not ss and not svr_in and not svr_out and (b,ph) not in pvph_list:
            return pyo.Constraint.Skip
        # BFM: entrada líquida reativa = Q_ij - x_ij·l_sq_ij
        in_Q  = sum(m.Q[i,j,p,t] - m.x[i,j,p]*m.l_sq[i,j,p,t] for (i,j,p) in ps)
        out_Q = sum(m.Q[i,j,p,t] for (i,j,p) in ss)
        svr_i = sum(m.Q_svr[p,t] for p in svr_in)
        svr_o = sum(m.Q_svr[p,t] for p in svr_out)
        Qpv   = m.Qpv[b,ph,t] if (b,ph) in pvph_list else 0.0
        aQ_v  = _aQ.get((b,ph), 0.0)
        Qd    = (m.Qd0[b,ph]*m.alpha_bph[b,ph,t]*(1.0 + aQ_v*(m.V[b,ph,t]-1.0))
                 if aQ_v != 0.0 else m.Qd0[b,ph]*m.alpha_bph[b,ph,t])
        return in_Q - out_Q + Qpv + svr_i - svr_o == Qd
    m.c_bal_Q = pyo.Constraint(m.BPH, m.T, rule=bal_Q_rule)

    # ── R4: Queda de tensão — BFM em magnitude V ─────────────────────────────
    # Variável V é magnitude (p.u.), não quadrado v=V².
    # Relação BFM exata em v=V²: v_j = v_i - 2(r·P + x·Q) + (r²+x²)·l_sq
    # Linearizando para V (magnitude) em torno de V≈1 p.u.:
    #   V_j ≈ V_i - (r·P + x·Q) + 0.5·(r²+x²)·l_sq
    # O fator 1 (não 2) em (r·P+x·Q) e 0.5 em (r²+x²)·l_sq
    # resultam da divisão por 2·V_nom≈2 na linearização de V_j².
    # Referência: Baran & Wu (1989), eq.(2)-(3); Jabr (2006).
    def vdrop_rule(m, i, j, ph, t):
        if (i,j,ph) in svr_branches: return pyo.Constraint.Skip
        r_d = branches[(i,j,ph)]['r_pu']
        x_d = branches[(i,j,ph)]['x_pu']
        z2  = r_d**2 + x_d**2
        return (m.V[j,ph,t] == m.V[i,ph,t]
                - r_d*m.P[i,j,ph,t]
                - x_d*m.Q[i,j,ph,t]
                + 0.5*z2*m.l_sq[i,j,ph,t])
    m.c_vdrop = pyo.Constraint(m.CPH, m.T, rule=vdrop_rule)

    # ── R5: Restrição cônica SOCP — cone de Lorentz ───────────────────────────
    # l_sq_ij · V_i² ≥ P_ij² + Q_ij²
    # ── R5: Restrição cônica BFM-SOCP — V² parametrizado pelo warm-start ────
    # Formulação original (bilinear, bloqueada pelo LP writer do Pyomo):
    #   l_sq[ij] · V[i]² >= P[ij]² + Q[ij]²
    #
    # Reformulação com V[i] parametrizado como constante do warm-start:
    #   l_sq[ij] · V_ws²[i] >= P[ij]² + Q[ij]²
    # onde V_ws²[i] = V_ws[i,ph,t]² é um PARÂMETRO (não variável).
    # O LP writer aceita porque:
    #   l_sq · CONSTANTE  → linear em l_sq
    #   P[ij]² + Q[ij]²   → quadrático em P e Q individualmente (sem produto cruzado)
    #   → QP separável: aceito pelo LP writer do Pyomo
    #
    # Validade: exata quando V_ws ≈ V_ótimo. Para planejamento estático com
    # bom warm-start (erro tipicamente < 3%), a solução é praticamente exata.
    # Referência: Jabr (2006) — iteração de ponto fixo para BFM.
    #
    # Adicionar parâmetro V_sq_ws: V_ws[i,ph,t]² por ramo
    # (V upstream do ramo — barra i do ramo (i,j,ph))
    m.V_sq_ws = pyo.Param(
        m.CPH, m.T,
        initialize={(i,j,ph,t): max(V_ws.get((i,ph,t), 0.97)**2, 0.64)
                    for (i,j,ph) in cph_list for t in hours},
        within=pyo.NonNegativeReals)
    def socp_rule(m, i, j, ph, t):
        if (i,j,ph) in svr_branches: return pyo.Constraint.Skip
        # l_sq · V_ws²[upstream] >= P² + Q²
        # V_sq_ws é parâmetro → restrição QP separável (LP writer OK)
        return (m.l_sq[i,j,ph,t] * m.V_sq_ws[i,j,ph,t]
                >= m.P[i,j,ph,t]**2 + m.Q[i,j,ph,t]**2)
    m.c_socp = pyo.Constraint(m.CPH, m.T, rule=socp_rule)
    print(f"  SOCP: {len(cph_list)*len(hours):,} restrições cônicas "
          f"(cone rotacionado — V parametrizado pelo warm-start)")

    # ── R6: Modelo do SVR — linearização A1 ──────────────────────────────────
    # V_lv = tap_min·V_mv + step·tap_pos  (linear em V_mv e tap_pos)
    def svr_rule(m, ph, t):
        mv = svr[ph]['mv_bus']; lv = svr[ph]['lv_bus']
        if (lv,ph) not in connected_bph: return pyo.Constraint.Skip
        return m.V[lv,ph,t] == tap_min*m.V[mv,ph,t] + step_*m.tap_pos[ph,t]
    m.c_svr = pyo.Constraint(m.SVRPH, m.T, rule=svr_rule)

    # ── R7: Curtailment PV ────────────────────────────────────────────────────
    m.c_pcurt_ub = pyo.Constraint(
        m.PVPH, m.T,
        rule=lambda m,b,ph,t: m.Pcurt[b,ph,t] <= m.Pavail[b,ph,t])
    m.c_pcurt_night = pyo.Constraint(
        m.PVPH, m.T,
        rule=lambda m,b,ph,t: (m.Pcurt[b,ph,t] == 0.0)
        if irr[t] < 1e-6 else pyo.Constraint.Skip)

    # ── R8: Curva de capacidade PV — IEEE 1547 Cat B ──────────────────────────
    # |Qpv| ≤ (Pavail - Pcurt) · tan(arccos(FP_min))
    m.c_fp_hi = pyo.Constraint(
        m.PVPH, m.T,
        rule=lambda m,b,ph,t:
        m.Qpv[b,ph,t] <=  (m.Pavail[b,ph,t]-m.Pcurt[b,ph,t])*_tan_fp)
    m.c_fp_lo = pyo.Constraint(
        m.PVPH, m.T,
        rule=lambda m,b,ph,t:
        m.Qpv[b,ph,t] >= -(m.Pavail[b,ph,t]-m.Pcurt[b,ph,t])*_tan_fp)

    # ── R9: Qpv = 0 fora da janela VV ────────────────────────────────────────
    m.c_qpv_nvv = pyo.Constraint(
        m.PVPH, m.T_NVV,
        rule=lambda m,b,ph,t: m.Qpv[b,ph,t] == 0.0)

    # ── R10: Suavização ΔQpv ─────────────────────────────────────────────────
    t_vv_list = sorted([t for t in hours if T_VV_START <= t <= T_VV_END])
    _w_ramp   = {t: (W_RAMP_PICO if IRR_PROFILE[t] > IRR_PICO else W_RAMP_DIA)
                 for t in t_vv_list}
    def ramp_rule(m, b, ph, t):
        prev = m.Qpv[b,ph,t-1] if t-1 in hours else 0.0
        return m.ramp_pos[b,ph,t] - m.ramp_neg[b,ph,t] == m.Qpv[b,ph,t] - prev
    m.c_ramp = pyo.Constraint(
        [(b,ph,t) for (b,ph) in pvph_list for t in t_vv_list],
        rule=ramp_rule)

    # ── R11: Detecção de evento de tap (big-M) ────────────────────────────────
    # CORREÇÃO CRÍTICA: a referência inicial é a POSIÇÃO INICIAL REAL do tap
    # (svr[ph]['tap_init']), não zero. Usar 0 contaria uma operação fantasma
    # no início para qualquer tap_init > tap_min.
    M_tap = n_pos - 1
    # Posição inteira inicial por fase (calculada uma vez para uso nas regras)
    _tap_init_pos = {
        ph: int(round((svr[ph]['tap_init'] - tap_min) / step_))
        for ph in svr_phs
    }
    def _tap_prev(m, ph, t):
        # Posição de tap no período anterior; em t=0 retorna a posição inicial
        if t == hours[0]:
            return _tap_init_pos[ph]
        return m.tap_pos[ph, hours[hours.index(t)-1]]
    m.c_tap_up = pyo.Constraint(
        m.SVRPH, m.T,
        rule=lambda m,ph,t:
        m.tap_pos[ph,t] - _tap_prev(m, ph, t)
        <= M_tap*m.u_tap[ph,t])
    m.c_tap_dn = pyo.Constraint(
        m.SVRPH, m.T,
        rule=lambda m,ph,t:
        _tap_prev(m, ph, t) - m.tap_pos[ph,t]
        <= M_tap*m.u_tap[ph,t])

    # ── R12: Limite de operações de tap ───────────────────────────────────────
    m.c_tap_max = pyo.Constraint(
        m.SVRPH,
        rule=lambda m,ph: sum(m.u_tap[ph,t] for t in hours) <= N_TAP_MAX)

    # ── R13: Delay do SVR ─────────────────────────────────────────────────────
    m.c_tap_delay = pyo.Constraint(
        m.SVRPH, m.T,
        rule=lambda m,ph,t: m.u_tap[ph,t] == 0
        if t < d_delay else pyo.Constraint.Skip)

    # ── R14: Linearização |Vi - Vj| para VUF ─────────────────────────────────
    m.c_unbal_pos = pyo.Constraint(
        [(b,ph1,ph2,t) for b in all_trif for (ph1,ph2) in unbal_pairs for t in hours],
        rule=lambda m,b,ph1,ph2,t: m.unbal[b,ph1,ph2,t] >= m.V[b,ph1,t]-m.V[b,ph2,t])
    m.c_unbal_neg = pyo.Constraint(
        [(b,ph1,ph2,t) for b in all_trif for (ph1,ph2) in unbal_pairs for t in hours],
        rule=lambda m,b,ph1,ph2,t: m.unbal[b,ph1,ph2,t] >= m.V[b,ph2,t]-m.V[b,ph1,t])

    # ── R15: FD% — PRODIST Módulo 8 (SOFT via F.O., não hard constraint) ────
    # A restrição hard de FD% pode tornar o problema infeasível quando o
    # desequilíbrio inerente da rede (cargas delta por fase, topologia assimétrica)
    # excede o limite regulatório antes de qualquer controle.
    # Solução: manter apenas como penalidade soft W_UNBAL·Σ|Vi-Vj| na F.O.
    # A conformidade FD95% é verificada e reportada no pós-processamento.
    # Para ativar como hard: descomentar o bloco abaixo.
    #
    # def _vuf_lim(b):
    #     lv = buses.get(b, {}).get('level', 'mv')
    #     fd = FD_BT_MAX if lv == 'lv' else FD_MT_MAX
    #     return fd / (K_FD_VUF * 100) * math.sqrt(3)
    # m.c_vuf_hard = pyo.Constraint(
    #     [(b,ph1,ph2,t) for b in all_trif for (ph1,ph2) in unbal_pairs for t in hours],
    #     rule=lambda m,b,ph1,ph2,t: m.unbal[b,ph1,ph2,t] <= _vuf_lim(b))

    # ── R16: Desvio |V - V_ref| linearizado ─────────────────────────────────
    # V_ref = 0.95 p.u. para barras MT com carga modelo 1 (P-const)
    # V_ref = 1.0167 p.u. para todas as demais barras (modelos 2/5 ou sem carga MT)
    m.c_dvp = pyo.Constraint(
        _dv_idx,
        rule=lambda m,b,ph,t: m.dv_pos[b,ph,t] >= m.V[b,ph,t] - m.Vref[b,ph])
    m.c_dvn = pyo.Constraint(
        _dv_idx,
        rule=lambda m,b,ph,t: m.dv_neg[b,ph,t] >= m.Vref[b,ph] - m.V[b,ph,t])

    # ── FUNÇÃO OBJETIVO ───────────────────────────────────────────────────────
    # Regulação de tensão pura:
    #   (1) Desequilíbrio entre fases  — W_UNBAL · Σ|Vi-Vj|
    #   (2) Eventos mecânicos de tap   — W_TAP_OPS · Σu_tap
    #   (3) Suavização ΔQpv            — W_RAMP · Σ|ΔQpv|
    # Perdas NÃO estão na F.O.: são contabilizadas fisicamente
    # nas restrições R2, R3, R4 pelo BFM com l_sq.
    # ── FUNÇÃO OBJETIVO ─────────────────────────────────────────────────────
    # Hierarquia de controle (5 termos, alinhada à versão LinDistFlow v4):
    #   (1) W_DV    · Σ|V-V_ref|   âncora de tensão (V_ref diferenciado por barra)
    #   (2) W_UNBAL · Σ|Vi-Vj|     desequilíbrio FD% (PRODIST Módulo 8)
    #   (3) W_TAP   · Σu_tap        eventos mecânicos de tap SVR
    #   (4) W_RAMP  · Σ|ΔQpv|      suavização do despacho reativo
    #   (5) W_CURT  · ΣPcurt       preservar geração PV disponível
    # Perdas NÃO entram na F.O.: contabilizadas nas restrições BFM (r·l_sq).
    m.obj = pyo.Objective(
        expr=(
            # (1) Desvio |V - V_ref| — âncora principal
            # V_ref = 0.95 p.u. nas barras MT com carga modelo 1 (P-const);
            # V_ref = 1.0167 p.u. nas demais (modelos 2/5 ou sem carga MT)
            W_DV * sum(m.dv_pos[b,ph,t] + m.dv_neg[b,ph,t]
                       for (b,ph,t) in _dv_idx)
            # (2) Desequilíbrio entre fases — FD% soft (PRODIST)
            + W_UNBAL * sum(m.unbal[b,ph1,ph2,t]
                            for b in all_trif
                            for (ph1,ph2) in unbal_pairs
                            for t in hours)
            # (3) Eventos mecânicos de tap SVR
            + W_TAP_OPS * sum(m.u_tap[ph,t]
                              for ph in svr_phs for t in hours)
            # (4) Suavização ΔQpv — vida útil dos inversores
            + sum(_w_ramp[t]*(m.ramp_pos[b,ph,t]+m.ramp_neg[b,ph,t])
                  for (b,ph) in pvph_list for t in t_vv_list)
            # (5) Curtailment PV — preservar geração disponível
            # Perdas r·l_sq NÃO estão aqui — apenas nas restrições BFM.
            + W_CURT * sum(m.Pcurt[b,ph,t]
                           for (b,ph) in pvph_list for t in hours)
        ),
        sense=pyo.minimize
    )
    n_vars   = sum(1 for v in m.component_data_objects(pyo.Var, active=True))
    n_constr = sum(1 for c in m.component_data_objects(pyo.Constraint, active=True))
    n_bin    = sum(1 for v in m.component_data_objects(pyo.Var, active=True)
                   if v.is_binary())
    print(f"  Variáveis: {n_vars:,}  ({n_bin} binárias)")
    print(f"  Restrições: {n_constr:,}")
    print(f"  (inclui {len(cph_list)*len(hours):,} cônicas SOCP)")

    return m, {
        'bph': bph_list, 'cph': cph_list, 'pvph': pvph_list,
        'svr_phs': svr_phs, 'hours': hours, 'irr': irr, 'load': load,
        't_vv': (T_VV_START, T_VV_END), 'trif_buses': all_trif,
        'tap_step': step_, 'tap_min': tap_min, 'n_pos': n_pos,
    }, I2_ws


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# GRÁFICOS — BFM-SOCP (adaptado do LinDistFlow v4)
# No BFM-SOCP: m.V[b,ph,t] em p.u. — mesma interface do LinDistFlow.
# =============================================================================

def try_v(m, b, ph, t):
    """Extrai tensão em p.u. do modelo BFM-SOCP."""
    try:
        return pyo.value(m.V[b, ph, t])
    except Exception:
        return 1.0


# =============================================================================
# HELPERS CENTRALIZADOS PARA CÁLCULO DE FD% (PRODIST Módulo 8)
# =============================================================================
# Limitação estrutural reconhecida:
# O modelo BFM-SOCP fornece apenas MAGNITUDES de tensão por fase (sem ângulos
# explícitos). As tensões de linha são reconstruídas assumindo defasagem rígida
# de 120° entre fases. Isto significa que o FD% computado captura apenas o
# componente de desequilíbrio de AMPLITUDE; o componente angular não é
# representado. Esta limitação deve ser declarada explicitamente no paper/tese.
# =============================================================================

def _calc_vll_from_vph(V1, V2, V3):
    """Calcula tensões de linha (V_ab, V_bc, V_ca) a partir das tensões de fase
    assumindo defasagem ideal de 120°. cos(120°) = -1/2.
    V_ab² = V1² + V2² - 2·V1·V2·cos(120°) = V1² + V2² + V1·V2."""
    return (math.sqrt(max(V1**2 + V2**2 + V1*V2, 0.0)),
            math.sqrt(max(V2**2 + V3**2 + V2*V3, 0.0)),
            math.sqrt(max(V3**2 + V1**2 + V3*V1, 0.0)))


def _calc_fd_prodist(Vab, Vbc, Vca):
    """Calcula FD% pela fórmula alternativa do PRODIST Módulo 8
    (Submódulo 8.1, Seção 4.2 — Desequilíbrio de Tensão):

        beta = (Vab^4 + Vbc^4 + Vca^4) / (Vab^2 + Vbc^2 + Vca^2)^2
        FD%  = 100 * sqrt[(1 - sqrt(3 - 6·beta)) / (1 + sqrt(3 - 6·beta))]

    Esta fórmula é equivalente ao método das componentes simétricas
    (V-/V+) em regime senoidal puro, e dispensa a decomposição em
    sequência positiva/negativa.
    """
    S2 = Vab**2 + Vbc**2 + Vca**2
    if S2 < 1e-12:
        return 0.0
    beta = (Vab**4 + Vbc**4 + Vca**4) / S2**2
    arg = max(0.0, min(1.0, 3.0 - 6.0*beta))
    if abs(1.0 - arg) < 1e-14:
        return 0.0
    sqarg = math.sqrt(arg)
    return 100.0 * math.sqrt((1.0 - sqarg) / (1.0 + sqarg))


def calc_fd_percent(m, data, sets_info):
    """Calcula FD% (PRODIST Módulo 8) a partir de m.V — retorna estatísticas ricas.

    Retorna dicionário com:
      fd_per_bus    : {bus: [FD%(t) para t in hours]}
      fd95_per_bus  : {bus: percentil 95 (proxy do FD95% regulatório)}
      fd_max_per_bus: {bus: pico instantâneo}
      t_peak_per_bus: {bus: t do pico (índice de período)}
      fd_all        : np.array com TODAS amostras (bus × tempo) flat
      fd95_system   : percentil 95 sistêmico (proxy)
      fd_max_system : pico instantâneo do sistema
      t_peak_system : período do pico sistêmico
      b_peak_system : barra do pico sistêmico
      pct_time_viol : % de amostras (bus,t) excedendo limite PRODIST
      n_viol        : nº absoluto de amostras (bus,t) em violação
      n_total       : nº total de amostras
      critical_buses: barras cuja FD95% supera o limite PRODIST
    """
    import numpy as np_fd
    buses  = data['buses']
    hours  = sets_info['hours']
    bph    = sets_info['bph']
    bph_set = set(bph)
    all_trif = sorted({b for (b, ph) in bph
                        if all((b, p) in bph_set for p in [1,2,3])})

    fd_per_bus, fd95_per_bus, fd_max_per_bus = {}, {}, {}
    t_peak_per_bus = {}
    critical_buses = []
    n_viol = 0
    n_total = 0
    fd_all_flat = []
    fd_max_system = 0.0
    t_peak_system = hours[0]
    b_peak_system = all_trif[0] if all_trif else None

    for b in all_trif:
        lv = buses.get(b, {}).get('level', 'mv')
        lim = FD_BT_MAX if lv == 'lv' else FD_MT_MAX
        fd_s = []
        t_pk_b = hours[0]
        fd_pk_b = 0.0
        for t in hours:
            V1 = try_v(m, b, 1, t)
            V2 = try_v(m, b, 2, t)
            V3 = try_v(m, b, 3, t)
            fd = _calc_fd_prodist(*_calc_vll_from_vph(V1, V2, V3))
            fd_s.append(fd)
            fd_all_flat.append(fd)
            n_total += 1
            if fd > lim:
                n_viol += 1
            if fd > fd_pk_b:
                fd_pk_b = fd
                t_pk_b = t
            if fd > fd_max_system:
                fd_max_system = fd
                t_peak_system = t
                b_peak_system = b
        fd_per_bus[b] = fd_s
        fd95_b = float(np_fd.percentile(fd_s, 95))
        fd95_per_bus[b] = fd95_b
        fd_max_per_bus[b] = fd_pk_b
        t_peak_per_bus[b] = t_pk_b
        if fd95_b > lim:
            critical_buses.append((b, fd95_b, lim))

    fd_all_arr = np_fd.array(fd_all_flat) if fd_all_flat else np_fd.array([0.0])
    fd95_system = float(np_fd.percentile(fd_all_arr, 95))
    pct_time_viol = (100.0 * n_viol / n_total) if n_total else 0.0

    return {
        'fd_per_bus':     fd_per_bus,
        'fd95_per_bus':   fd95_per_bus,
        'fd_max_per_bus': fd_max_per_bus,
        't_peak_per_bus': t_peak_per_bus,
        'fd_all':         fd_all_arr,
        'fd95_system':    fd95_system,
        'fd_max_system':  fd_max_system,
        't_peak_system':  t_peak_system,
        'b_peak_system':  b_peak_system,
        'pct_time_viol':  pct_time_viol,
        'n_viol':         n_viol,
        'n_total':        n_total,
        'critical_buses': critical_buses,
    }


def plot_results(m, data, sets_info):
    """Gráfico 1: FD% antes/depois | Gráfico 2: curva V-Q | Gráfico 3: tap SVR."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("  pip install matplotlib"); return

    buses  = data['buses']
    pv     = data['pv']
    svr    = data['svr']
    hours  = sets_info['hours']
    irr    = sets_info['irr']
    load   = sets_info['load']
    bph    = sets_info['bph']
    pvph   = sets_info['pvph']
    svr_phs = sets_info['svr_phs']
    t_vv   = sets_info['t_vv']
    step_  = sets_info['tap_step']
    tap_min_ = sets_info['tap_min']
    n_pos_ = sets_info['n_pos']

    bph_set = set(bph)
    trif_buses = sorted({b for (b,ph) in bph
                          if buses.get(b,{}).get('level')=='mv'
                          and all((b,p) in bph_set for p in [1,2,3])})

    # Pré-OPF: forward sweep
    V_pre, _, _, _, _, _ = compute_warm_start(data, hours, irr, load)

    # Cálculo de FD% por amostra (bus, t) usando helpers centralizados
    # Limitação: FD% computado captura apenas desequilíbrio de amplitude
    # (modelo BFM-SOCP não fornece ângulos de tensão explícitos).
    def _fd_pre_sample(b, t):
        V1 = V_pre.get((b, 1, t), 1.0)
        V2 = V_pre.get((b, 2, t), 1.0)
        V3 = V_pre.get((b, 3, t), 1.0)
        return _calc_fd_prodist(*_calc_vll_from_vph(V1, V2, V3))
    def _fd_post_sample(b, t):
        return _calc_fd_prodist(*_calc_vll_from_vph(
            try_v(m, b, 1, t), try_v(m, b, 2, t), try_v(m, b, 3, t)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("BFM-SOCP OPF — Volt-VAR + SVR | 144×10min",
                 fontsize=13, fontweight='bold')

    # ── SUB1: CDF do FD% — proxy do FD95% regulatório (PRODIST Módulo 8) ───
    # Recolhe TODAS amostras (bus, t) antes e depois do OPF.
    fd_pre_all  = np.array([_fd_pre_sample(b, t)
                            for b in trif_buses for t in hours])
    fd_post_all = np.array([_fd_post_sample(b, t)
                            for b in trif_buses for t in hours])

    if fd_pre_all.size and fd_post_all.size:
        # CDF empírica (Função de Distribuição Cumulativa)
        sorted_pre  = np.sort(fd_pre_all)
        sorted_post = np.sort(fd_post_all)
        cdf_pre  = np.arange(1, len(sorted_pre)  + 1) / len(sorted_pre)
        cdf_post = np.arange(1, len(sorted_post) + 1) / len(sorted_post)

        # Indicadores estatísticos
        fd95_pre   = float(np.percentile(fd_pre_all,  95))
        fd95_post  = float(np.percentile(fd_post_all, 95))
        fd_max_pre  = float(np.max(fd_pre_all))
        fd_max_post = float(np.max(fd_post_all))

        # Curvas CDF
        ax1.plot(sorted_pre,  cdf_pre  * 100, '-',
                 color='#E05C3A', lw=2.2, label='Before OPF')
        ax1.plot(sorted_post, cdf_post * 100, '-',
                 color='#2A7EBA', lw=2.2, label='After OPF')

        # Linha horizontal P95 (referência estatística do PRODIST)
        ax1.axhline(95, color='black', lw=1.0, ls=':', alpha=0.6)
        ax1.text(ax1.get_xlim()[1] * 0.02, 95.5, 'P95',
                 fontsize=7, color='black', alpha=0.7)

        # Linha vertical do limite PRODIST MT (2,0%)
        ax1.axvline(FD_MT_MAX, color='gray', lw=1.2, ls='--',
                    label=f'MV limit {FD_MT_MAX:.1f}% PRODIST')

        # Marcadores de FD95% antes/depois sobre as curvas
        ax1.plot(fd95_pre,  95, 'o', color='#E05C3A', ms=10,
                 mec='white', mew=1.5, zorder=5)
        ax1.plot(fd95_post, 95, 's', color='#2A7EBA', ms=10,
                 mec='white', mew=1.5, zorder=5)

        # Caixa de texto com indicadores numéricos
        viol_pre  = 100.0 * np.sum(fd_pre_all  > FD_MT_MAX) / fd_pre_all.size
        viol_post = 100.0 * np.sum(fd_post_all > FD_MT_MAX) / fd_post_all.size
        info_txt = (
            f'FD95%: {fd95_pre:.2f}% → {fd95_post:.2f}%\n'
            f'Pico:  {fd_max_pre:.2f}% → {fd_max_post:.2f}%\n'
            f'%t>lim: {viol_pre:.1f}% → {viol_post:.1f}%'
        )
        ax1.text(0.97, 0.05, info_txt, transform=ax1.transAxes,
                 fontsize=8, ha='right', va='bottom', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='white', edgecolor='gray', alpha=0.9))

        ax1.set_xlim(0, max(fd_max_pre, FD_MT_MAX * 1.5) * 1.05)
        ax1.set_ylim(0, 102)
        ax1.set_xlabel('FD% (PRODIST Module 8)')
        ax1.set_ylabel('Empirical CDF [%]')
        ax1.set_title('Cumulative distribution of FD%\n'
                      '(daily proxy of regulatory FD95%)')
        ax1.legend(fontsize=8, loc='center right')
        ax1.grid(alpha=0.3)

    # ── SUB2: Curva Volt-VAR ───────────────────────────────────────────────
    pvph_list_g = sets_info['pvph']
    t_vv_h = [t for t in hours if t_vv[0]<=t<=t_vv[1]]
    t_peak = max(hours, key=lambda t: irr[t])
    _tan_fp = math.sqrt(1-FP_MIN_VV**2)/FP_MIN_VV
    if pvph_list_g:
        best_bph = max(pvph_list_g,
                       key=lambda bp: abs(pyo.value(m.Qpv[bp[0],bp[1],t_peak])))
        b_vv, ph_vv = best_bph
        V_op=[try_v(m,b_vv,ph_vv,t) for t in t_vv_h]
        Q_op=[pyo.value(m.Qpv[b_vv,ph_vv,t])*SBASE for t in t_vv_h]
        irr_op=[irr[t] for t in t_vv_h]
        _pv_loc = data['pv']
        Qmax_pk = _pv_loc[(b_vv,ph_vv)]['P_rated_pu']*irr[t_peak]*_tan_fp*SBASE if (b_vv,ph_vv) in _pv_loc else 50.
        v_c=np.linspace(0.88,1.12,200)
        def _vvc(v,qm):
            V1,V2,V3,V4=0.92,0.98,1.02,1.08
            if v<=V1: return qm
            if v<=V2: return qm*(V2-v)/(V2-V1)
            if v<=V3: return 0.
            if v<=V4: return -qm*(v-V3)/(V4-V3)
            return -qm
        ax2.plot(v_c,[_vvc(v,Qmax_pk) for v in v_c],'--',color='#BA7517',lw=2,label='IEEE 1547 Cat B')
        norm=mcolors.Normalize(vmin=0,vmax=max(irr_op) if irr_op else 1)
        sc=ax2.scatter(V_op,Q_op,c=irr_op,cmap='YlOrRd',norm=norm,s=60,zorder=5,label='Optimal OPF point')
        plt.colorbar(sc,ax=ax2,label='Irrad.',fraction=0.04)
        ax2.axhline(0,color='gray',lw=.7,ls=':')
        ax2.set_xlabel('Voltage [p.u.]'); ax2.set_ylabel('Qpv [kVAr/phase]')
        ax2.set_title(f'Volt-VAR curve\n({b_vv}, phase {ph_vv})'); ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    # ── SUB3: Tap SVR ──────────────────────────────────────────────────────
    colors_ph={1:'#7F77DD',2:'#E0803A',3:'#1D9E75'}
    ax3.axvspan(t_vv[0]-.5,t_vv[1]+.5,alpha=0.07,color='#378ADD',label='VV window')
    for ph in svr_phs:
        tap_seq=[tap_min_+int(round(pyo.value(m.tap_pos[ph,t])))*step_ for t in hours]
        ax3.step(hours,tap_seq,where='mid',color=colors_ph[ph],lw=2.2,label=f'Phase {ph}')
        prev=tap_min_+int(round((svr[ph]['tap_init']-tap_min_)/step_))*step_
        for i,t in enumerate(hours):
            curr=tap_seq[i]
            if abs(curr-prev)>step_*0.5:
                mk='^' if curr>prev else 'v'
                ax3.scatter(t,curr,marker=mk,s=100,color=colors_ph[ph],zorder=6,edgecolors='white')
            prev=curr
    # Redefine xtk para uso aqui (variável local removida do SUB1 refatorado)
    xtk = [k for k in range(0, N_PERIODS, 12)]
    ax3.set_xticks(xtk); ax3.set_xticklabels([f"{k*DT_MIN//60:02d}h" for k in xtk],fontsize=7,rotation=45)
    ax3.set_xlabel('Period'); ax3.set_ylabel('Tap [p.u.]')
    ax3.set_title('SVR tap — 3 phases\n(▲▼=mechanical operation)'); ax3.legend(fontsize=8); ax3.grid(alpha=.3)

    plt.tight_layout(pad=2.)
    plt.savefig('opf_bfm_voltvar_tap.png',dpi=150,bbox_inches='tight')
    print("  Salvo: opf_bfm_voltvar_tap.png")
    plt.show()


def plot_voltage_mt(m, data, sets_info):
    """Heatmap e evolução temporal de tensão nas barras MT."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("  pip install matplotlib"); return
    from collections import deque, defaultdict

    buses   = data['buses']
    branches= data['branches']
    svr     = data['svr']
    slack   = data['slack']
    hours   = sets_info['hours']
    irr     = sets_info['irr']
    load    = sets_info['load']
    bph     = sets_info['bph']
    svr_phs = sets_info['svr_phs']
    t_vv    = sets_info['t_vv']
    step_   = sets_info['tap_step']
    tap_min_= sets_info['tap_min']

    bph_set = set(bph)
    adj_mt  = defaultdict(list)
    for (fr,to,ph) in branches:
        if buses.get(fr,{}).get('level')=='mv' and buses.get(to,{}).get('level')=='mv':
            if to not in adj_mt[fr]: adj_mt[fr].append(to)
    for ph, d in svr.items():
        if d['lv_bus'] not in adj_mt[d['mv_bus']]: adj_mt[d['mv_bus']].append(d['lv_bus'])

    visited, order_mt = set(), []
    q = deque([slack]); visited.add(slack)
    while q:
        b = q.popleft()
        if buses.get(b, {}).get('level') == 'mv': order_mt.append(b)
        for nb in adj_mt.get(b, []):
            if nb not in visited: visited.add(nb); q.append(nb)
    mt_buses = [b for b in order_mt if any((b, ph) in bph_set for ph in [1, 2, 3])]
    if not mt_buses: print("  Sem barras MT"); return

    V_pre,_,_,_,_,_=compute_warm_start(data,hours,irr,load)
    t_peak=max(hours,key=lambda t:irr[t])
    colors_ph={1:'#7F77DD',2:'#E0803A',3:'#1D9E75'}

    fig,axes=plt.subplots(2,2,figsize=(22,12))
    fig.suptitle("MV voltages — BFM-SOCP | Modified IEEE-13",fontsize=13,fontweight='bold')
    ax1,ax2,ax3,ax4=axes[0,0],axes[0,1],axes[1,0],axes[1,1]

    # Sub1: perfil espacial no pico
    x_pos=list(range(len(mt_buses)))
    for ph in svr_phs:
        va=[V_pre.get((b,ph,t_peak),1.) for b in mt_buses]
        vp=[try_v(m,b,ph,t_peak) for b in mt_buses]
        ax1.plot(x_pos,va,'--',color=colors_ph[ph],lw=1.5,alpha=0.45)
        ax1.plot(x_pos,vp,'-o',color=colors_ph[ph],lw=2,ms=5,label=f'Phase {ph}')
    ax1.axhspan(V_ADEQ_MT_LO,V_ADEQ_MT_HI,alpha=0.07,color='#1D9E75',label='Adequada')
    ax1.axhline(V_ADEQ_MT_LO,color='#1D9E75',lw=.9,ls='--',alpha=.7)
    ax1.axhline(1.0,color='gray',lw=.7,ls=':',alpha=.5)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(mt_buses,rotation=45,ha='right',fontsize=7)
    ax1.set_title(f'Spatial profile — peak t={t_peak}\n(dashed=pre-OPF, solid=post-OPF)')
    ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    # Sub2: evolução temporal de barras selecionadas (fase 1)
    n_sel=min(6,len(mt_buses))
    sel=[mt_buses[i] for i in range(0,len(mt_buses),max(1,len(mt_buses)//n_sel))[:n_sel]]
    cmap_b=plt.cm.plasma
    for idx,b in enumerate(sel):
        vs=[try_v(m,b,1,t) for t in hours]
        ax2.plot(hours,vs,color=cmap_b(idx/max(len(sel)-1,1)),lw=1.8,label=b)
    # mudanças de tap fase 1
    for i,t in enumerate(hours[1:],1):
        p_cur=int(round(pyo.value(m.tap_pos[1,t])))
        p_prv=int(round(pyo.value(m.tap_pos[1,hours[i-1]])))
        if p_cur!=p_prv:
            ax2.axvline(t,color='#BA7517',lw=1.5,ls='--',alpha=.8)
    ax2.axhspan(V_ADEQ_MT_LO,V_ADEQ_MT_HI,alpha=0.06,color='#1D9E75')
    ax2.axhline(1.0,color='gray',lw=.7,ls=':',alpha=.5)
    xtk=[k for k in range(0,N_PERIODS,12)]
    ax2.set_xticks(xtk); ax2.set_xticklabels([f"{k*DT_MIN//60:02d}h" for k in xtk],fontsize=7)
    ax2.set_title('Temporal evolution V — MV buses\n(── = phase-1 tap change)')
    ax2.legend(fontsize=7); ax2.grid(alpha=.3)

    # Sub3: Heatmap V[barra×tempo] fase 1
    V_mat=np.array([[try_v(m,b,1,t) for t in hours] for b in mt_buses])
    vmin_h=max(0.88,V_mat.min()-.003); vmax_h=min(1.12,V_mat.max()+.003)
    vctr=max(vmin_h+1e-4,min(1.0,vmax_h-1e-4))
    norm_h=mcolors.TwoSlopeNorm(vcenter=vctr,vmin=vmin_h,vmax=vmax_h)
    im=ax3.imshow(V_mat,aspect='auto',cmap='RdYlGn',norm=norm_h,
                  extent=[-0.5,len(hours)-.5,len(mt_buses)-.5,-.5])
    for i,t in enumerate(hours[1:],1):
        if int(round(pyo.value(m.tap_pos[1,t])))!=int(round(pyo.value(m.tap_pos[1,hours[i-1]]))):
            ax3.axvline(t,color='white',lw=1.5,ls='--',alpha=.85)
    plt.colorbar(im,ax=ax3,fraction=.03,label='V [p.u.]')
    ax3.set_yticks(range(len(mt_buses))); ax3.set_yticklabels(mt_buses,fontsize=7)
    ax3.set_xticks(xtk); ax3.set_xticklabels([f"{k*DT_MIN//60:02d}h" for k in xtk],fontsize=7)
    ax3.set_title('Heatmap V[bus×time] — Phase 1')

    # Sub4: scatter tap × V[RG60]
    rg60=svr[svr_phs[0]]['lv_bus']
    norm_t=mcolors.Normalize(vmin=0,vmax=len(hours)-1)
    mk=['o','s','^']
    for ph in svr_phs:
        tv=[tap_min_+int(round(pyo.value(m.tap_pos[ph,t])))*step_ for t in hours]
        vr=[try_v(m,rg60,ph,t) for t in hours]
        ax4.scatter(tv,vr,c=list(range(len(hours))),cmap='viridis',norm=norm_t,
                    s=20,alpha=.65,marker=mk[ph-1],label=f'Phase {ph}')
    ax4.axhspan(V_ADEQ_MT_LO,V_ADEQ_MT_HI,alpha=0.07,color='#1D9E75',label='Adequada')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_t,cmap='viridis'),ax=ax4,fraction=.03,label='Period k')
    ax4.set_xlabel('Tap [p.u.]'); ax4.set_ylabel(f'V[{rg60}] [p.u.]')
    ax4.set_title(f'Tap × V[{rg60}] correlation'); ax4.legend(fontsize=8); ax4.grid(alpha=.3)

    plt.tight_layout(pad=2.5)
    plt.savefig('opf_bfm_voltage_mt.png',dpi=150,bbox_inches='tight')
    print("  Salvo: opf_bfm_voltage_mt.png")
    plt.show()


# =============================================================================
# TEMPORAL COMPARISON: BEFORE vs AFTER OPF — Voltage + Tap changes (phase 1)
# =============================================================================
# Produces a 2-panel figure side by side:
#   LEFT  : V(t) of 3-4 critical MV buses BEFORE OPF (local LDC control),
#           with vertical dashed lines marking the baseline tap-change events
#   RIGHT : V(t) of the same buses AFTER OPF, with vertical dashed lines
#           marking the OPF-optimized tap-change events
#
# Designed to be inserted in the Voltage Regulation subsection of the paper.
# Marker in paper_resultados.tex: [INSERT FIG: FIG_TAP_VOLTAGE_COMP]
# =============================================================================

def plot_voltage_temporal_comparison(m, data, sets_info,
                                     save_path='opf_bfm_voltage_temporal_comp.png'):
    """Side-by-side comparison of MV voltage evolution before and after OPF,
    with vertical markers at each tap-change event in each scenario.

    Uses 4 critical MV buses (slack, RG60 post-SVR, two downstream
    representative buses) of phase 1, to keep the figure clean."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import deque, defaultdict
    except ImportError:
        print("  [TEMP_COMP] Install matplotlib first."); return

    buses     = data['buses']
    hours     = sets_info['hours']
    bph_set   = set(sets_info['bph'])
    branches  = data.get('branches', {})
    svr       = data.get('svr', {})
    irr       = sets_info.get('irr')
    load      = sets_info.get('load')

    # ── (1) Recompute pre-OPF voltages via warm-start (same approach used
    #       elsewhere in plot_voltage_mt and plot_voltage_profile_and_vv)
    try:
        V_pre, _, _, _, _, _ = compute_warm_start(data, hours, irr, load)
    except Exception as e:
        print(f"  [TEMP_COMP] Could not compute pre-OPF voltages: {e}")
        return

    # ── (2) Recompute pre-OPF tap trajectory via local LDC simulation
    try:
        tap_traj_pre, _, _ = simulate_svr_local_control(data, hours, irr, load)
    except Exception as e:
        print(f"  [TEMP_COMP] Could not simulate baseline tap trajectory: {e}")
        return

    # ── (3) Select 4 critical MV buses: prefer the canonical ones if present
    mv_buses_all = sorted({b for (b, ph) in bph_set
                           if buses.get(b, {}).get('level', 'mv') == 'mv'})
    preferred = ['650', 'RG60', '671', '675']
    sel = [b for b in preferred if b in mv_buses_all]
    if len(sel) < 4:
        # Fallback: take the first 4 MV buses
        sel = mv_buses_all[:4]
    if not sel:
        print("  [TEMP_COMP] No MV buses found."); return

    # ── (4) Identify tap-change events for phase 1 in each scenario
    def _tap_changes_post(ph):
        changes = []
        prev = None
        for t in hours:
            try:
                cur = int(round(pyo.value(m.tap_pos[ph, t])))
            except Exception:
                continue
            if prev is not None and cur != prev:
                changes.append(t)
            prev = cur
        return changes

    def _tap_changes_pre(ph):
        changes = []
        prev = None
        for t in hours:
            cur = tap_traj_pre.get((ph, t))
            if cur is None:
                continue
            if prev is not None and cur != prev:
                changes.append(t)
            prev = cur
        return changes

    changes_pre  = _tap_changes_pre(1)
    changes_post = _tap_changes_post(1)

    # ── (5) Build the plots
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("MV voltage evolution and tap-change events — "
                 "before vs.\\ after OPF (phase 1)",
                 fontsize=13, fontweight='bold')

    cmap_b = plt.cm.plasma
    n_sel = len(sel)

    # ── Left panel: BEFORE (baseline LDC control)
    for idx, b in enumerate(sel):
        vs = [V_pre.get((b, 1, t), 1.0) for t in hours]
        axL.plot(hours, vs, color=cmap_b(idx / max(n_sel - 1, 1)),
                 lw=1.8, label=b)
    for t_ch in changes_pre:
        axL.axvline(t_ch, color='#BA7517', lw=1.2, ls='--', alpha=0.6)
    try:
        axL.axhspan(V_ADEQ_MT_LO, V_ADEQ_MT_HI, alpha=0.06, color='#1D9E75')
    except NameError:
        axL.axhspan(0.93, 1.05, alpha=0.06, color='#1D9E75')
    axL.axhline(1.0, color='gray', lw=.7, ls=':', alpha=0.5)

    xtk = list(range(0, N_PERIODS, 12))
    axL.set_xticks(xtk)
    axL.set_xticklabels([f"{k * DT_MIN // 60:02d}h" for k in xtk], fontsize=8)
    axL.set_xlabel('Time of day', fontsize=10)
    axL.set_ylabel('Voltage [p.u.]', fontsize=10)
    axL.set_title(f'BEFORE OPF (baseline LDC)\n'
                  f'{len(changes_pre)} tap-change events (phase 1)',
                  fontsize=11)
    axL.legend(fontsize=8, loc='lower right', title='MV bus')
    axL.grid(alpha=0.3)

    # ── Right panel: AFTER (OPF-optimized)
    for idx, b in enumerate(sel):
        vs = [try_v(m, b, 1, t) for t in hours]
        axR.plot(hours, vs, color=cmap_b(idx / max(n_sel - 1, 1)),
                 lw=1.8, label=b)
    for t_ch in changes_post:
        axR.axvline(t_ch, color='#BA7517', lw=1.5, ls='--', alpha=0.85)
    try:
        axR.axhspan(V_ADEQ_MT_LO, V_ADEQ_MT_HI, alpha=0.06, color='#1D9E75')
    except NameError:
        axR.axhspan(0.93, 1.05, alpha=0.06, color='#1D9E75')
    axR.axhline(1.0, color='gray', lw=.7, ls=':', alpha=0.5)

    axR.set_xticks(xtk)
    axR.set_xticklabels([f"{k * DT_MIN // 60:02d}h" for k in xtk], fontsize=8)
    axR.set_xlabel('Time of day', fontsize=10)
    axR.set_title(f'AFTER OPF (proposed framework)\n'
                  f'{len(changes_post)} tap-change events (phase 1)',
                  fontsize=11)
    axR.legend(fontsize=8, loc='lower right', title='MV bus')
    axR.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [TEMP_COMP] Saved: {save_path}")
    print(f"  [TEMP_COMP] Tap changes — before: {len(changes_pre)} | "
          f"after: {len(changes_post)}")
    plt.show()
    return fig


# =============================================================================
# CONSOLIDATED VOLTAGE HEATMAP — STANDALONE FUNCTION
# =============================================================================
# Generates a 2-D heatmap V[b,phi,t] showing all bus-phase pairs of MV and LV
# along the daily horizon. Bus-phases are ordered by voltage level (MV first,
# then LV) and by proximity to the slack bus within each level.
#
# Usage: this function is self-contained and consumes only the Pyomo model,
# the data dictionary, and the sets_info dictionary already in memory after
# a solver run. It does NOT require re-solving the model.
#
#   plot_voltage_heatmap(model, data, sets_info)
#
# Output: writes 'opf_bfm_voltage_heatmap.png' to current working directory.
# =============================================================================

def plot_voltage_heatmap(m, data, sets_info, save_path=None,
                         force_phase=None):
    """Generates a voltage heatmap V[bus, time] for the most critical
    phase (or a user-specified phase), with buses ordered by topological
    distance to the slack bus.

    Selection criterion (default): the most critical phase is the one
    maximizing the spatio-temporal range max(V) - min(V) across all
    bus-time samples — i.e., the phase exhibiting the largest voltage
    variation, hence the most representative for distance-voltage
    analysis.

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel
        Solved BFM-SOCP model with m.V[b, phi, t] populated.
    data : dict
        Network data dictionary (must contain 'buses' and 'branches').
    sets_info : dict
        Sets info (must contain 'hours' and 'bph').
    save_path : str, optional
        Output PNG path. Defaults to 'opf_bfm_voltage_heatmap.png'.
    force_phase : int in {1, 2, 3}, optional
        Override the automatic phase selection.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        from collections import deque, defaultdict
    except ImportError:
        print("  [HEATMAP] Install matplotlib first.")
        return

    buses     = data['buses']
    hours     = sets_info['hours']
    bph_set   = set(sets_info['bph'])
    branches  = data.get('branches', {})
    if save_path is None:
        save_path = 'opf_bfm_voltage_heatmap.png'

    def _level_of(b):
        return buses.get(b, {}).get('level', 'mv')

    # ── (1) Build adjacency graph including transformers (MT→BT bridges)
    adj = defaultdict(list)
    for key in branches:
        # branches keys may be (fr, to, ph) or (fr, to)
        if isinstance(key, tuple) and len(key) >= 2:
            fr, to = key[0], key[1]
            if to not in adj[fr]:
                adj[fr].append(to)
            if fr not in adj[to]:
                adj[to].append(fr)

    # Add transformer edges: each transformer connects its MV bus to LV bus
    trafos = data.get('trafos', {}) or data.get('transformers', {})
    for tid, td in (trafos or {}).items():
        mv_b = td.get('mv_bus') or td.get('hv_bus')
        lv_b = td.get('lv_bus')
        if mv_b and lv_b:
            if lv_b not in adj[mv_b]: adj[mv_b].append(lv_b)
            if mv_b not in adj[lv_b]: adj[lv_b].append(mv_b)

    # ── (2) Identify slack bus
    slack = None
    for b, bd in buses.items():
        if bd.get('slack', False) or b == '650' or bd.get('is_slack', False):
            slack = b
            break
    if slack is None:
        # Fallback: pick first MV bus alphabetically
        mv_list = sorted([b for b in buses if _level_of(b) == 'mv'])
        slack = mv_list[0] if mv_list else next(iter(buses))

    # ── (3) BFS distance from slack
    dist = {slack: 0}
    q = deque([slack])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    # Fallback distance for disconnected buses
    max_d = max(dist.values()) if dist else 0
    for b in buses:
        if b not in dist:
            dist[b] = max_d + 1   # put disconnected buses at the bottom

    # ── (4) Pick the most critical phase (largest spatio-temporal range)
    if force_phase in (1, 2, 3):
        crit_phase = force_phase
        sel_reason = f"forced by user (phase {force_phase})"
    else:
        ranges = {}
        for ph in [1, 2, 3]:
            vals = []
            for (b, p) in bph_set:
                if p != ph: continue
                for t in hours:
                    vals.append(try_v(m, b, ph, t))
            if vals:
                ranges[ph] = max(vals) - min(vals)
        if not ranges:
            print("  [HEATMAP] No phase data available.")
            return
        crit_phase = max(ranges, key=lambda k: ranges[k])
        sel_reason = (f"max spatio-temporal range = "
                      f"{ranges[crit_phase]*1000:.1f} mp.u. "
                      f"(phases: {{1:{ranges.get(1,0)*1000:.1f}, "
                      f"2:{ranges.get(2,0)*1000:.1f}, "
                      f"3:{ranges.get(3,0)*1000:.1f}}} mp.u.)")

    # ── (5) Order buses by (level, distance from slack)
    #       MV buses first (top), then LV; within each level, sort by distance.
    buses_with_phase = sorted({b for (b, ph) in bph_set if ph == crit_phase})
    mv_buses = sorted([b for b in buses_with_phase if _level_of(b) == 'mv'],
                      key=lambda b: (dist.get(b, 999), b))
    lv_buses = sorted([b for b in buses_with_phase if _level_of(b) == 'lv'],
                      key=lambda b: (dist.get(b, 999), b))
    ordered_buses = mv_buses + lv_buses
    n_mv = len(mv_buses)
    n_total = len(ordered_buses)

    if n_total == 0:
        print(f"  [HEATMAP] No buses with phase {crit_phase}.")
        return

    # ── (6) Build the V matrix [bus × time] for the selected phase
    V_mat = np.full((n_total, len(hours)), np.nan, dtype=float)
    for i, b in enumerate(ordered_buses):
        for j, t in enumerate(hours):
            V_mat[i, j] = try_v(m, b, crit_phase, t)

    # ── (7) Color scale centered at 1.0 p.u.
    v_lo, v_hi = 0.90, 1.06
    norm = mcolors.TwoSlopeNorm(vmin=v_lo, vcenter=1.0, vmax=v_hi)
    cmap = plt.get_cmap('RdYlGn_r')

    # ── (8) Plot
    height = max(6, n_total * 0.22)
    fig, ax = plt.subplots(figsize=(13, height))
    im = ax.imshow(V_mat, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest', origin='upper')

    # MV/LV separator
    if 0 < n_mv < n_total:
        ax.axhline(n_mv - 0.5, color='black', lw=1.8, ls='-')
        ax.text(-0.5, n_mv - 0.5, ' MV / LV ',
                ha='right', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black',
                          boxstyle='round,pad=0.3'))

    # X-axis: time labels every 2 h
    n_per_hour = max(1, int(60 / DT_MIN))
    xtk = list(range(0, len(hours), 2 * n_per_hour))
    ax.set_xticks(xtk)
    ax.set_xticklabels([f"{(k * DT_MIN) // 60:02d}h" for k in xtk],
                       fontsize=8, rotation=0)

    # Y-axis: bus labels with topological distance prefix
    ylabels = [f"{b}  (d={dist.get(b, '?')})" for b in ordered_buses]
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel('Time of day', fontsize=11)
    ax.set_ylabel('Bus  (ordered by hops from slack; d = topological distance)',
                  fontsize=10)
    ax.set_title(f'Voltage heatmap $V_b^{{{crit_phase},t}}$ — post-OPF\n'
                 f'Phase {crit_phase} (critical) | MV (top) and LV (bottom) '
                 f'× 144 periods of 10 min',
                 fontsize=12, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Voltage [p.u.]', fontsize=10)
    ref_ticks = [0.90, 0.92, 0.93, 0.95, 1.00, 1.05, 1.06]
    cbar.set_ticks(ref_ticks)
    cbar.ax.tick_params(labelsize=8)

    # Summary box
    v_min_obs = float(np.nanmin(V_mat))
    v_max_obs = float(np.nanmax(V_mat))
    v_mean    = float(np.nanmean(V_mat))
    info_txt = (f'Phase {crit_phase} (criterion: largest range)\n'
                f'V min/max:  [{v_min_obs:.4f}, {v_max_obs:.4f}] p.u.\n'
                f'V mean:      {v_mean:.4f} p.u.\n'
                f'Buses:       {n_total}  ({n_mv} MV + {n_total-n_mv} LV)\n'
                f'Max hops:    {max(dist.get(b,0) for b in ordered_buses)}')
    ax.text(1.05, -0.05, info_txt, transform=ax.transAxes,
            fontsize=8, ha='left', va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [HEATMAP] Saved: {save_path}")
    print(f"  [HEATMAP] Phase selection: {sel_reason}")
    plt.show()
    return fig


def plot_voltage_profile_and_vv(m, data, sets_info):
    """Perfil espacial MT + curva V-Q ajustada por regressão."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  pip install matplotlib"); return
    from collections import deque, defaultdict

    buses   = data['buses']
    branches= data['branches']
    pv      = data['pv']
    svr     = data['svr']
    slack   = data['slack']
    hours   = sets_info['hours']
    irr     = sets_info['irr']
    load    = sets_info['load']
    bph     = sets_info['bph']
    pvph    = sets_info['pvph']
    svr_phs = sets_info['svr_phs']
    t_vv    = sets_info['t_vv']
    step_   = sets_info['tap_step']
    tap_min_= sets_info['tap_min']

    bph_set=set(bph)
    adj_mt=defaultdict(list)
    for (fr,to,ph) in branches:
        if buses.get(fr,{}).get('level')=='mv' and buses.get(to,{}).get('level')=='mv':
            if to not in adj_mt[fr]: adj_mt[fr].append(to)
    for ph,d in svr.items():
        if d['lv_bus'] not in adj_mt[d['mv_bus']]: adj_mt[d['mv_bus']].append(d['lv_bus'])

    visited, order_mt = set(), []
    q = deque([slack]); visited.add(slack)
    while q:
        b = q.popleft()
        if buses.get(b, {}).get('level') == 'mv': order_mt.append(b)
        for nb in adj_mt.get(b, []):
            if nb not in visited: visited.add(nb); q.append(nb)
    mt_buses = [b for b in order_mt if any((b, ph) in bph_set for ph in [1, 2, 3])]

    V_pre,_,_,_,_,_=compute_warm_start(data,hours,irr,load)
    loads_mt_=data.get('loads_mt',{})
    # Mantém coerência com a lógica de V_ref da F.O.: barras com carga
    # MT modelo 1 (P-const) têm V_ref = 0,95; demais (modelos 2/5 ou
    # sem carga) têm V_ref = 1,0167.
    _m1_b={b for (b,ph),ld in loads_mt_.items() if ld.get('model',1) == 1}

    max_dev=0.; crit_bus=mt_buses[0] if mt_buses else slack; crit_ph=1; crit_t=0
    for b in mt_buses:
        TR=0.95 if b in _m1_b else 1.0167
        for ph in [1,2,3]:
            if (b,ph) not in bph_set: continue
            for t in hours:
                dev=abs(V_pre.get((b,ph,t),1.0)-TR)
                if dev>max_dev: max_dev=dev; crit_bus=b; crit_ph=ph; crit_t=t

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
    fig.suptitle("BFM-SOCP — MV profile and fitted V-Q curve",fontsize=13,fontweight='bold')
    colors_ph={1:'#7F77DD',2:'#E0803A',3:'#1D9E75'}
    x_pos=list(range(len(mt_buses)))

    for ph in svr_phs:
        vp=[V_pre.get((b,ph,crit_t),1.) for b in mt_buses]
        vo=[try_v(m,b,ph,crit_t) for b in mt_buses]
        ax1.plot(x_pos,vp,'--',color=colors_ph[ph],lw=1.5,alpha=.45)
        ax1.plot(x_pos,vo,'-o',color=colors_ph[ph],lw=2,ms=5,label=f'Phase {ph}')
    ax1.axhspan(V_ADEQ_MT_LO,V_ADEQ_MT_HI,alpha=.07,color='#1D9E75',label='Adequada MT')
    ax1.axhline(V_ADEQ_MT_LO,color='#1D9E75',lw=.9,ls='--',alpha=.7)
    ax1.axhline(1.0,color='gray',lw=.7,ls=':')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(mt_buses,rotation=45,ha='right',fontsize=7)
    t_hhmm=f"{crit_t*DT_MIN//60:02d}:{crit_t*DT_MIN%60:02d}"
    ax1.set_title(f'MV profile — t={crit_t} ({t_hhmm}) | critical bus: {crit_bus} ph{crit_ph}')
    ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    t_vv_h=[t for t in hours if t_vv[0]<=t<=t_vv[1]]
    _tan_fp=math.sqrt(1-FP_MIN_VV**2)/FP_MIN_VV
    best_bph=(pvph[0] if pvph else None)
    if pvph:
        t_peak=max(hours,key=lambda t:irr[t])
        best_bph=max(pvph,key=lambda bp:abs(pyo.value(m.Qpv[bp[0],bp[1],t_peak])))
    if best_bph:
        b_vv,ph_vv=best_bph
        mask=[irr[t]>0.05 for t in t_vv_h]
        V_s=np.array([try_v(m,b_vv,ph_vv,t) for i,t in enumerate(t_vv_h) if mask[i]])
        Q_s=np.array([pyo.value(m.Qpv[b_vv,ph_vv,t])*SBASE for i,t in enumerate(t_vv_h) if mask[i]])
        irr_s=np.array([irr[t] for i,t in enumerate(t_vv_h) if mask[i]])
        if len(V_s)>2:
            dead_mask=np.abs(Q_s)<0.05*SBASE
            V2f=float(np.percentile(V_s[dead_mask],10)) if dead_mask.sum()>1 else 0.97
            V3f=float(np.percentile(V_s[dead_mask],90)) if dead_mask.sum()>1 else 1.03
            V1f=max(0.88,float(np.percentile(V_s,5)))
            V4f=min(1.12,float(np.percentile(V_s,95)))
            Qmx=float(np.percentile(np.abs(Q_s),95)) if len(Q_s)>0 else 10.
            v_r=np.linspace(min(0.88,V1f-.02),max(1.12,V4f+.02),300)
            def _vvc(v):
                if v<=V1f: return Qmx
                if v<=V2f: return Qmx*(V2f-v)/max(V2f-V1f,1e-6)
                if v<=V3f: return 0.
                if v<=V4f: return -Qmx*(v-V3f)/max(V4f-V3f,1e-6)
                return -Qmx
            sc=ax2.scatter(V_s,Q_s,c=irr_s,cmap='YlOrRd',s=30,alpha=.7,label='OPF points')
            plt.colorbar(sc,ax=ax2,fraction=.03,label='Irrad.')
            ax2.plot(v_r,[_vvc(v) for v in v_r],'-',color='#185FA5',lw=2.5,label='Fitted curve')
            for vbp,lbl in [(V1f,'V1'),(V2f,'V2'),(V3f,'V3'),(V4f,'V4')]:
                ax2.axvline(vbp,color='#BA7517',lw=.9,ls='--',alpha=.6)
                ax2.text(vbp,ax2.get_ylim()[0] if ax2.get_ylim()[0]<0 else -Qmx*.1,
                         lbl,ha='center',fontsize=8,color='#BA7517')
            ax2.axhline(0,color='gray',lw=.7,ls=':')
            ax2.set_title(f'Fitted V-Q curve\n({b_vv}, phase {ph_vv})')
            ax2.set_xlabel('Voltage [p.u.]'); ax2.set_ylabel('Qpv [kVAr/phase]')
            ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    plt.tight_layout(pad=2.5)
    plt.savefig('opf_bfm_voltage_profile_vv.png',dpi=150,bbox_inches='tight')
    print("  Salvo: opf_bfm_voltage_profile_vv.png")
    plt.show()

# ═════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO DO CONTROLE LOCAL DO REGULADOR (BASELINE PRÉ-OPF)
# ═════════════════════════════════════════════════════════════════════════
# O controle local de tap em reguladores Cooper/SEL opera com lógica de
# banda morta + delay (Line Drop Compensation simplificado), sem nenhuma
# otimização. Esta função reproduz esse comportamento para gerar a contagem
# de operações pré-OPF compatível com a simulação do OpenDSS.
#
# Lógica do controle:
#   Para cada período t:
#     1. Medir V_lv(t) com varredura direta usando o tap atual
#     2. Se V_lv > V_ref + BW/2: tap_pos -= 1 (após delay)
#     3. Se V_lv < V_ref - BW/2: tap_pos += 1 (após delay)
#     4. Caso contrário: manter posição
#
# Parâmetros típicos IEEE Cooper:
#   V_ref = 1.0 p.u. (122 V na escala 120 V)
#   BW    = 2.0 V/120 V = 0.0167 p.u. (banda morta total)
#   delay = 15 s = 1 período de 10 min (1.5 períodos arredondados)
# =========================================================================

V_REF_SVR   = 1.0167    # p.u. — referência do controle local (122V/120V)
                          # OpenDSS configura Vreg=122V em base 120V → 1.0167 pu
BW_SVR_PU   = 0.0167    # p.u. — banda morta total (2V/120V)
# Delay de atuação — padrão de campo IEEE Cooper/SEL:
#   15-30 s : ajuste sensível (alimentadores curtos, baixo desbalanceamento)
#   30-60 s : ajuste padrão para distribuição rural/urbana
#   60-120 s: ajuste conservador para alta penetração PV (anti-hunting)
# Referências:
#   - IEEE Std C37.230 (ajustes típicos)
#   - Material Standards 648058.1: faixa configurável 0-120s
#   - Kim et al. (2019): SVR primário com BESS, delays 30-150s
# Cada período = 10 min na simulação; 3 períodos × 10 s da unidade de
# tempo do controle = 30 s, dentro da faixa padrão de campo.
DELAY_PER   = 3         # períodos de espera após detecção (≈30 s)

def simulate_svr_local_control(data, hours, irr_d, load_d):
    """Simula o controle local do regulador período a período.

    Para cada período, executa uma varredura direta da rede com o tap atual,
    mede V no secundário do regulador e aplica a lógica de banda morta para
    decidir a próxima posição.

    Retorna:
        tap_traj: {(ph, t): pos_inteira} — trajetória discreta do tap
        ops_per_phase: {ph: int} — número de operações por fase
        V_lv_traj: {(ph, t): V_lv} — tensão medida no secundário
    """
    from collections import deque, defaultdict
    buses    = data['buses']
    loads    = data['loads']
    pv       = data['pv']
    branches = data['branches']
    svr      = data['svr']
    slack    = data['slack']

    # Construir adjacências (mesma lógica do warm-start)
    adj = defaultdict(list)
    for (fr, to, ph) in branches:
        adj[(fr, ph)].append((to, ph, (fr, to, ph)))
    for ph, d in svr.items():
        adj[(d['mv_bus'], ph)].append((d['lv_bus'], ph, ('SVR', ph)))

    def bfs(root, rphs):
        order, visited = [], set()
        q = deque([(root, ph) for ph in rphs])
        for item in q: visited.add(item)
        while q:
            node = q.popleft(); order.append(node)
            for (to, tph, br) in adj.get(node, []):
                if (to, tph) not in visited:
                    visited.add((to, tph)); q.append((to, tph))
        return order

    order  = bfs(slack, buses[slack]['phases'])
    parent = {}
    for (b, ph) in order:
        for (to, tph, br) in adj.get((b, ph), []):
            if (to, tph) not in parent:
                parent[(to, tph)] = ((b, ph), br)

    svr_phs   = sorted(svr.keys())
    step_     = svr[svr_phs[0]]['tap_step']
    tap_min_  = svr[svr_phs[0]]['tap_min']
    n_pos     = svr[svr_phs[0]]['n_tap']

    # Estado: posição discreta atual e contador de delay por fase
    pos_cur   = {ph: int(round((svr[ph]['tap_init']-tap_min_)/step_))
                 for ph in svr_phs}
    delay_ctr = {ph: 0 for ph in svr_phs}
    ops       = {ph: 0 for ph in svr_phs}
    tap_traj  = {}
    V_lv_traj = {}

    for t in hours:
        # ── 1. Varredura direta com o tap atual ────────────────────────
        # Calcular potências líquidas em cada barra
        P_net = {}; Q_net = {}
        for (b, ph) in order:
            if b in _MT_LOAD_PROFILE_MAP:
                _aw = _LOAD_BY_TYPE[_MT_LOAD_PROFILE_MAP[b]][t]
            else:
                lvl = buses.get(b, {}).get('level', 'mv')
                _aw = LOAD_PROFILE_MT[t] if lvl == 'mv' else LOAD_PROFILE[t]
            Pd = loads.get((b, ph), {'P_pu': 0})['P_pu'] * _aw
            Qd = loads.get((b, ph), {'Q_pu': 0})['Q_pu'] * _aw
            Ppv = pv.get((b, ph), {'P_rated_pu': 0})['P_rated_pu'] * PV_EFF_FACTOR[t]
            P_net[(b, ph)] = Pd - Ppv
            Q_net[(b, ph)] = Qd

        # Backward sweep: somar fluxos
        P_br = {}; Q_br = {}
        for (b, ph) in reversed(order):
            pf = P_net[(b, ph)]; qf = Q_net[(b, ph)]
            for (to, tph, br) in adj.get((b, ph), []):
                pf += P_br.get((to, ph), 0); qf += Q_br.get((to, ph), 0)
            P_br[(b, ph)] = pf; Q_br[(b, ph)] = qf

        # Forward sweep: calcular tensões com o tap_cur atual
        V = {}
        for (b, ph) in order:
            if b == slack:
                V[(b, ph)] = 1.0; continue
            if (b, ph) not in parent:
                V[(b, ph)] = 0.97; continue
            (up, uph), br_key = parent[(b, ph)]
            Vup = V.get((up, uph), 1.0)
            if br_key == ('SVR', ph):
                # Tap atual aplicado
                tau = tap_min_ + pos_cur[ph] * step_
                V[(b, ph)] = tau * Vup
            else:
                r = branches[br_key]['r_pu']
                x = branches[br_key]['x_pu']
                Pf = P_br.get((b, ph), 0); Qf = Q_br.get((b, ph), 0)
                z2 = r**2 + x**2
                l_sq = (Pf**2 + Qf**2) / max(Vup**2, 0.64)
                V[(b, ph)] = max(Vup - r*Pf - x*Qf + 0.5*z2*l_sq, 0.80)

        # ── 2. Lógica de controle do tap (banda morta + delay + LDC) ────
        # LDC: V_medida = V_lv - (R_ldc·I_P + X_ldc·I_Q)
        # I_P e I_Q são as componentes ativa/reativa da corrente pelo SVR,
        # expressas em p.u. da corrente nominal do CT primário.
        # Com fluxo reverso (PV > carga), I muda de sinal → LDC compensa
        # 'pensando' que a tensão na carga subiu → desce tap.
        for ph in svr_phs:
            lv_bus = svr[ph]['lv_bus']
            mv_bus = svr[ph]['mv_bus']
            V_lv   = V.get((lv_bus, ph), 1.0)

            # Corrente passando pelo SVR (em p.u. base MT)
            # P_svr = P_br[(lv_bus,ph)]  Q_svr idem
            # I_complex = (P-jQ)/V*  →  |I|² = (P²+Q²)/V²
            # Para LDC: I_P = P/V (componente ativa), I_Q = Q/V (reativa)
            P_svr = P_br.get((lv_bus, ph), 0.0)
            Q_svr = Q_br.get((lv_bus, ph), 0.0)
            V_for_I = max(V_lv, 0.5)   # evitar divisão por valores baixos
            I_P = P_svr / V_for_I
            I_Q = Q_svr / V_for_I

            # Compensação LDC (parâmetros do regulador desta fase)
            R_ldc = svr[ph].get('R_ldc_pu', 0.0)
            X_ldc = svr[ph].get('X_ldc_pu', 0.0)
            V_drop_ldc = R_ldc * I_P + X_ldc * I_Q
            V_meas     = V_lv - V_drop_ldc   # tensão estimada no centro de carga
            V_lv_traj[(ph, t)] = V_meas

            # Referência e banda morta deste regulador
            V_ref_ph = svr[ph].get('V_ref_lc', V_REF_SVR)
            BW_ph    = svr[ph].get('BW_lc',    BW_SVR_PU)

            # Decisão de tap (banda morta sobre V_meas, não V_lv direto)
            err = V_meas - V_ref_ph
            need_change = False
            delta = 0
            if err > BW_ph/2 and pos_cur[ph] > 0:
                need_change = True; delta = -1   # V alta → descer tap
            elif err < -BW_ph/2 and pos_cur[ph] < n_pos-1:
                need_change = True; delta = +1   # V baixa → subir tap

            if need_change:
                if delay_ctr[ph] >= DELAY_PER:
                    pos_cur[ph] += delta
                    pos_cur[ph] = max(0, min(n_pos-1, pos_cur[ph]))
                    ops[ph] += 1
                    delay_ctr[ph] = 0
                else:
                    delay_ctr[ph] += 1
            else:
                delay_ctr[ph] = 0

            tap_traj[(ph, t)] = pos_cur[ph]

    return tap_traj, ops, V_lv_traj


if __name__ == '__main__':
    import os, subprocess

    # ── Dados ─────────────────────────────────────────────────────────────────
    GITHUB_REPO = 'DarleneJD/ACOPF'
    clone_dir   = '/content/ACOPF'
    if not os.path.isdir(clone_dir):
        subprocess.run(['git', 'clone', '--depth', '1',
                        f'https://github.com/{GITHUB_REPO}.git', clone_dir],
                       check=True)
    BUSES_FILE    = os.path.join(clone_dir, 'buses_1.xlsx')
    BRANCHES_FILE = os.path.join(clone_dir, 'branches_1.xlsx')

    print("="*72)
    print("  BFM-SOCP | IEEE-13 Modificado | CPLEX MISOCP")
    print(f"  SBASE={SBASE:.2f} kVA/f | ZMT={ZMT:.4f}Ω | ZBT3={ZBT3:.4f}Ω")
    print("="*72)

    data = load_data(BUSES_FILE, BRANCHES_FILE)

    # Resumo dos perfis temporais carregados
    print("\nPerfis temporais (144 períodos × 10 min):")
    print(f"  TECHNICHDOBRASIL (MT): min={min(TECHNICHDOBRASIL):.3f} "
          f"max={max(TECHNICHDOBRASIL):.3f} média={sum(TECHNICHDOBRASIL)/144:.3f}")
    print(f"  CURVA_R (BT):         min={min(CURVA_R):.3f} "
          f"max={max(CURVA_R):.3f} média={sum(CURVA_R)/144:.3f}")
    print(f"  IRRAD:                min={min(IRRAD):.3f} "
          f"max={max(IRRAD):.3f} média={sum(IRRAD)/144:.3f}")
    print(f"  TEMP_AMB [°C]:        min={min(TEMP_AMB):.1f} "
          f"max={max(TEMP_AMB):.1f} média={sum(TEMP_AMB)/144:.1f}")
    print(f"  PV_T_CELL [°C]:       min={min(PV_T_CELL):.1f} "
          f"max={max(PV_T_CELL):.1f} (NOCT={NOCT}, γ={GAMMA_PV})")
    _t_peak  = max(range(144), key=lambda k: IRRAD[k])
    _eff_pk  = PV_EFF_FACTOR[_t_peak]
    _derate  = _eff_pk / max(IRRAD[_t_peak], 1e-9)
    print(f"  Pico solar t={_t_peak} ({_t_peak*10//60:02d}:{_t_peak*10%60:02d}): "
          f"irrad={IRRAD[_t_peak]:.3f} T_cell={PV_T_CELL[_t_peak]:.1f}°C "
          f"derate={_derate:.3f} P_eff/P_nom={_eff_pk:.3f}")

    print("\nConstruindo modelo BFM-SOCP...")
    model, sets_info, I2_ws = build_opf(data)

    # ══════════════════════════════════════════════════════════════════════
    # (a) ESTADO INICIAL — antes da otimização
    # ══════════════════════════════════════════════════════════════════════
    svr_pre   = data['svr']
    hours_pre = sets_info['hours']
    step_pre  = sets_info['tap_step']
    tap_min_pre = sets_info['tap_min']
    trafos_pre  = data.get('trafos', {})
    branches_pre = data['branches']
    V_ws_pre, P_ws_pre, Q_ws_pre, _, _, _ = compute_warm_start(
        data, hours_pre, sets_info['irr'], sets_info['load'])

    # (a1) Operações de tap ANTES — simulação do controle local (banda morta+delay)
    # Esta é a baseline compatível com a simulação do OpenDSS:
    # o regulador opera com lógica local (sem otimização), respondendo
    # à variabilidade dos perfis de carga e da geração PV.
    print("\n" + "="*68)
    print("  ESTADO INICIAL (pré-OPF) — Controle Local do Regulador")
    print("="*68)
    # Imprimir parâmetros LDC por fase
    print(f"  Parâmetros do controle local (defaults globais):")
    print(f"    V_ref = {V_REF_SVR:.4f} p.u. | BW = {BW_SVR_PU:.4f} p.u. "
          f"(≈{BW_SVR_PU*120:.1f} V/120V) | delay = {DELAY_PER} períodos")
    print(f"  Parâmetros LDC por fase:")
    for _ph, _sd in sorted(svr_pre.items()):
        _Rldc = _sd.get('R_ldc_pu', 0.0)
        _Xldc = _sd.get('X_ldc_pu', 0.0)
        # Reverter para ohms para visualização
        _Rldc_ohm = _Rldc * ZMT
        _Xldc_ohm = _Xldc * ZMT
        print(f"    Fase {_ph}: R_ldc={_Rldc_ohm:.2f}Ω ({_Rldc:.5f} pu), "
              f"X_ldc={_Xldc_ohm:.2f}Ω ({_Xldc:.5f} pu), "
              f"V_ref={_sd.get('V_ref_lc',1.0):.4f} pu, "
              f"BW={_sd.get('BW_lc',0.0167):.4f} pu")
    print("  Simulando controle local de tap período a período (144 × 10 min)...")
    tap_traj_pre, ops_local, V_lv_traj_pre = simulate_svr_local_control(
        data, hours_pre, sets_info['irr'], sets_info['load'])

    tap_ops_pre = {}
    print(f"  {'Fase':>6} {'pos_init':>10} {'tap_init':>10} "
          f"{'pos_final':>10} {'tap_final':>10} {'ops_local':>10}")
    for ph, sd in svr_pre.items():
        pos_init = int(round((sd['tap_init'] - tap_min_pre) / step_pre))
        pos_final_local = tap_traj_pre.get((ph, hours_pre[-1]), pos_init)
        ops_ph = ops_local.get(ph, 0)
        tap_ops_pre[ph] = {
            'ops':         ops_ph,
            'pos_init':    pos_init,
            'pos_final':   pos_final_local,
            'tap_pu_init': sd['tap_init'],
            'tap_pu_final': tap_min_pre + pos_final_local * step_pre,
            'trajectory':  {t: tap_traj_pre.get((ph,t), pos_init) for t in hours_pre},
        }
        print(f"  {ph:>6} {pos_init:>10} {sd['tap_init']:>10.5f} "
              f"{pos_final_local:>10} "
              f"{tap_min_pre + pos_final_local*step_pre:>10.5f} "
              f"{ops_ph:>10}")
    total_ops_local = sum(ops_local.values())
    print(f"  Total (3 fases): {total_ops_local} operações de tap pré-OPF")
    print(f"  → Comparável à simulação OpenDSS (esperado: 33-39 com 100% PV)")

    # (a2) Perdas de núcleo PRÉ-OPF: P_NL = G_core · V² por trafo, fase, período
    # Usando tensões do warm-start (varredura direta analítica)
    print("\n  Perdas de núcleo pré-OPF (G_core·V²) — por transformador:")
    pnl_pre_total = {}   # {tid: lista por t de {ph: P_NL_kW}}
    for (mv, lv, ph), bd in branches_pre.items():
        if bd.get('level') != 'trafo': continue
        tid = bd.get('trafo_id', f'{mv}-{lv}')
        if tid not in pnl_pre_total: pnl_pre_total[tid] = {}
        G = bd.get('G_core_pu', 0.0)
        for t in hours_pre:
            V_mv = V_ws_pre.get((mv, ph, t), 1.0)
            pnl  = G * V_mv**2 * SBASE    # kW/fase/período
            if t not in pnl_pre_total[tid]:
                pnl_pre_total[tid][t] = {}
            pnl_pre_total[tid][t][ph] = pnl
    # Sumarizar por hora (agregar 6 períodos de 10 min = 1 hora)
    print(f"  {'Trafo':<12} {'P_NL_nom[kW/f]':>16} {'P_NL_pré_médio[kW/f]':>22}")
    for tid, td in trafos_pre.items():
        pnl_flat = [v for t_d in pnl_pre_total.get(tid,{}).values()
                    for v in t_d.values()]
        avg = sum(pnl_flat)/max(len(pnl_flat),1)
        print(f"  {tid:<12} {td['P_NL_nom_kw']:>16.4f} {avg:>22.4f}")


    # ══════════════════════════════════════════════════════════════════════════
    # ESTRATÉGIA DE RESOLUÇÃO EM 2 ESTÁGIOS
    # ══════════════════════════════════════════════════════════════════════════
    # O problema MISOCP com 432 binárias + 19.296 QC não encontra solução
    # inteira no nó raiz em 1800s (IInf=507 na relaxação LP).
    #
    # Estágio 1 — QP contínuo: relaxar tap_pos e u_tap para contínuos
    #   → Resolve em ~2-5 min (sem branch-and-bound)
    #   → Obtém Qpv*, V*, P*, Q* ótimos com tap contínuo
    #   → Arredonda tap_pos para posição discreta mais próxima
    #
    # Estágio 2 — MIQP com tap fixado: fixar tap_pos no valor do estágio 1
    #   → Elimina todas as 432 binárias u_tap + 432 inteiros tap_pos
    #   → Resolve como QP puro (sem MIP) em poucos minutos
    #   → Obtém Qpv*, Pcurt* ótimos com tap arredondado
    # ══════════════════════════════════════════════════════════════════════════
    from pyomo.environ import SolverFactory

    CPLEX_EXE = '/opt/ibm/ILOG/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex'
    svr_phs_s  = sets_info['svr_phs']
    hours_s    = sets_info['hours']
    step_s     = sets_info['tap_step']
    tap_min_s  = sets_info['tap_min']
    n_pos_s    = sets_info['n_pos']

    # ── Estágio 1: Relaxar inteirice — QP contínuo ────────────────────────────
    print("\n[Estágio 1] Relaxando tap_pos e u_tap para contínuos (QP)...")
    # Relaxar tap_pos: inteiro → contínuo [0, n_pos-1]
    for ph in svr_phs_s:
        for t in hours_s:
            model.tap_pos[ph,t].domain = pyo.NonNegativeReals
            model.u_tap[ph,t].domain   = pyo.NonNegativeReals
            model.u_tap[ph,t].setub(1.0)

    opt1 = SolverFactory('cplex', executable=CPLEX_EXE)
    opt1.options['qpmethod']    = 6      # barrier para QP
    opt1.options['timelimit']   = 600    # 10 min para o QP contínuo
    opt1.options['threads']     = 4
    opt1.options['barrier_convergetol'] = 1e-6

    t0 = time.time()
    print("  Resolvendo QP contínuo (relaxação de tap)...")
    res1 = opt1.solve(model, tee=True, load_solutions=False)
    res1.write()
    elapsed1 = time.time() - t0
    tc1 = str(res1.solver.termination_condition)
    # Carregar solução manualmente (QP/LP não carregam automaticamente no Pyomo)
    if len(res1.solution) > 0:
        model.solutions.load_from(res1)
    elif tc1 in ('optimal', 'locallyOptimal', 'feasible'):
        # Tentar re-resolver com load_solutions=True como fallback
        res1b = opt1.solve(model, tee=False)
        tc1 = str(res1b.solver.termination_condition)
    print(f"  Estágio 1: {tc1} | {elapsed1:.1f}s")

    # ── Arredondar tap_pos para posição discreta ──────────────────────────────
    tap_fixed = {}   # {(ph, t): pos_inteira}
    if tc1 in ('optimal', 'feasible', 'locallyOptimal'):
        # Diagnóstico: trajetória contínua de tap por fase
        print("\n  Trajetória contínua tap_pos[ph,t] — E1 (amostra):")
        print(f"  {'t':>4}  {'hora':>5}  " + "  ".join(f"  ph{ph}" for ph in svr_phs_s))
        for t in hours_s[::12][:8]:
            vals = "  ".join(f"{pyo.value(model.tap_pos[ph,t]):6.2f}" for ph in svr_phs_s)
            print(f"  {t:>4}  {t*DT_MIN//60:02d}:{t*DT_MIN%60:02d}  {vals}")
        print("\n  Arredondamento por PERSISTÊNCIA TEMPORAL (independente por fase)...")
        # Reformulação: o tap só muda quando a posição contínua do E1 sustenta
        # a mudança por MIN_PERSIST períodos consecutivos. Esta lógica reflete
        # diretamente o comportamento físico do regulador com delay temporal:
        # mudanças transitórias (< MIN_PERSIST períodos) são filtradas, pois
        # corresponderiam a oscilações que o controle local com banda morta+
        # delay também rejeitaria.
        #
        # Algoritmo (por fase, independente):
        #   1. Arredondar trajetória contínua → discreta por período
        #   2. Para cada candidato a mudança em t*, verificar se a nova
        #      posição persiste por pelo menos MIN_PERSIST períodos
        #   3. Aceitar mudança apenas se persistente; caso contrário, manter
        #      o tap atual
        #
        # MIN_PERSIST é alinhado ao delay do controle local (3 períodos = 30s).
        MIN_PERSIST = DELAY_PER     # mesmo delay do controle local
        N_TAP_MAX_s = sets_info.get('n_tap_max', N_TAP_MAX)

        for ph in svr_phs_s:
            # Estado inicial do tap desta fase (posição discreta)
            init_pos_s = int(round(
                (data['svr'][ph]['tap_init'] - tap_min_s) / step_s))
            init_pos_s = max(0, min(n_pos_s-1, init_pos_s))

            # Passo 1 — arredondar tap_pos contínuo → discreto por período
            pos_disc = [
                int(round(max(0, min(n_pos_s-1,
                    pyo.value(model.tap_pos[ph, t])))))
                for t in hours_s
            ]

            # Passo 2 — aplicar filtragem por persistência
            # cur_pos é o tap efetivamente aplicado; muda apenas se a nova
            # posição persistir por MIN_PERSIST períodos consecutivos
            final_pos = []
            cur_pos = init_pos_s
            n_periods = len(pos_disc)
            for idx_t in range(n_periods):
                proposed = pos_disc[idx_t]
                if proposed != cur_pos:
                    # Verificar se a mudança persiste pelo MIN_PERSIST períodos
                    window_end = min(idx_t + MIN_PERSIST, n_periods)
                    window = pos_disc[idx_t:window_end]
                    # Mudança aceita se a maioria do window confirma proposed
                    # (>= MIN_PERSIST/2 períodos com a nova posição)
                    confirmed = sum(1 for p in window if p == proposed)
                    if confirmed >= MIN_PERSIST:
                        cur_pos = proposed
                final_pos.append(cur_pos)

            # Aplicar teto físico N_TAP_MAX_s: se exceder, selecionar as
            # operações de maior magnitude (proteção contra patológico)
            ops_proposed = sum(1 for i in range(1, len(final_pos))
                               if final_pos[i] != final_pos[i-1])
            if final_pos[0] != init_pos_s:
                ops_proposed += 1

            if ops_proposed > N_TAP_MAX_s:
                # Aplicar fallback greedy top-N (proteção)
                print(f"    [!] SVR ph{ph}: {ops_proposed} ops excede teto "
                      f"{N_TAP_MAX_s}; aplicando top-N de magnitude")
                changes = []
                prev = init_pos_s
                for idx_t, pc in enumerate(final_pos):
                    if pc != prev:
                        changes.append((idx_t, abs(pc - prev), pc))
                    prev = pc
                top_changes = sorted(changes, key=lambda x: -x[1])[:N_TAP_MAX_s]
                sel_set = {c[0] for c in top_changes}
                new_final = []
                cur = init_pos_s
                for idx_t in range(n_periods):
                    if idx_t in sel_set:
                        cur = final_pos[idx_t]
                    new_final.append(cur)
                final_pos = new_final

            # Registrar em tap_fixed para o Estágio 2
            for idx_t, t in enumerate(hours_s):
                tap_fixed[(ph, t)] = final_pos[idx_t]

            # Estatísticas por fase
            ops_real = sum(1 for i in range(1, len(final_pos))
                           if final_pos[i] != final_pos[i-1])
            if final_pos[0] != init_pos_s:
                ops_real += 1
            tap_pu_f = tap_min_s + final_pos[-1] * step_s
            print(f"    SVR ph{ph}: {ops_real} ops (persist≥{MIN_PERSIST}p) | "
                  f"pos_final={final_pos[-1]} ({tap_pu_f:.5f} p.u.)")
    else:
        print("  [!] Estágio 1 não convergiu — usando tap inicial")
        init_pos = int(round((data['svr'][svr_phs_s[0]]['tap_init'] - tap_min_s) / step_s))
        for ph in svr_phs_s:
            for t in hours_s:
                tap_fixed[(ph, t)] = init_pos

    # ── Estágio 2: Fixar tap, resolver QP puro ───────────────────────────────
    print("\n[Estágio 2] Fixando tap arredondado e resolvendo QP puro...")
    for ph in svr_phs_s:
        for t in hours_s:
            pos = tap_fixed[(ph, t)]
            model.tap_pos[ph, t].fix(pos)
            model.u_tap[ph, t].fix(0)
    # Com tap fixo, u_tap é irrelevante — desativar restrições de tap
    model.c_tap_up.deactivate()     # tap_pos[t] - tap_pos[t-1] <= M·u_tap
    model.c_tap_dn.deactivate()     # tap_pos[t-1] - tap_pos[t] <= M·u_tap
    model.c_tap_delay.deactivate()  # u_tap=0 nos primeiros períodos
    model.c_tap_max.deactivate()    # Σu_tap <= N_TAP_MAX

    opt2 = SolverFactory('cplex', executable=CPLEX_EXE)
    opt2.options['qpmethod']    = 6      # barrier para QP convexo
    opt2.options['timelimit']   = 1200   # 20 min
    opt2.options['threads']     = 4
    opt2.options['barrier_convergetol'] = 1e-6

    print("  Resolvendo QP com tap fixado (Qpv*, P*, Q* ótimos)...")
    t1 = time.time()
    results = opt2.solve(model, tee=True, load_solutions=False)
    results.write()
    elapsed2 = time.time() - t1
    tc2 = str(results.solver.termination_condition)
    # Carregar solução do Estágio 2
    if len(results.solution) > 0:
        model.solutions.load_from(results)
    elif tc2 in ('optimal', 'locallyOptimal', 'feasible'):
        results2b = opt2.solve(model, tee=False)
        results = results2b
    print(f"\nTempo total: {time.time()-t0:.1f}s  "
          f"(E1: {elapsed1:.1f}s | E2: {elapsed2:.1f}s)")


    # ══════════════════════════════════════════════════════════════════════
    # DIAGNÓSTICO PÓS-OTIMIZAÇÃO — análise de tensões e folgas
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("  DIAGNÓSTICO DE TENSÃO")
    print("="*68)
    try:
        # Coletar tensões pós-OPF
        buses_diag = data['buses']
        v_mt_list, v_bt_list = [], []
        v_slack = pyo.value(model.V[data['slack'], 1, 0])
        v_rg60_max, v_rg60_min = -float('inf'), float('inf')
        rg60_bus = None
        for ph_d, sd in data['svr'].items():
            rg60_bus = sd['lv_bus']
            for t_d in sets_info['hours']:
                v = pyo.value(model.V[rg60_bus, ph_d, t_d])
                v_rg60_max = max(v_rg60_max, v)
                v_rg60_min = min(v_rg60_min, v)
        for (b_d, ph_d) in sets_info['bph']:
            lvl = buses_diag.get(b_d, {}).get('level', 'mv')
            for t_d in sets_info['hours']:
                try:
                    v = pyo.value(model.V[b_d, ph_d, t_d])
                    if lvl == 'mv':
                        v_mt_list.append(v)
                    else:
                        v_bt_list.append(v)
                except Exception:
                    pass
        import numpy as np_d

        # ───────────────────────────────────────────────────────────────
        # PRODIST Módulo 8 — Indicadores DRP e DRC (proxy diário):
        #   DRP = fração de leituras na faixa PRECÁRIA (limite ≤ 3%)
        #   DRC = fração de leituras na faixa CRÍTICA  (limite ≤ 0,5%)
        # Faixas (V em p.u. da tensão nominal):
        #   MT (4,16 kV):  Adequada [0,93; 1,05] | Precária [0,90; 0,93)∪(1,05; 1,06]
        #                  Crítica  V<0,90 ou V>1,06
        #   BT (0,48 kV):  Adequada [0,92; 1,05] | Precária [0,87; 0,92)∪(1,05; 1,06]
        #                  Crítica  V<0,87 ou V>1,06
        # Limites regulatórios: DRP ≤ 3,0% e DRC ≤ 0,5%.
        # ───────────────────────────────────────────────────────────────
        def _drp_drc(arr, v_adeq_min, v_adeq_max, v_prec_min, v_crit_high):
            """Retorna (DRP%, DRC%, n_prec, n_crit, n_total)."""
            n = arr.size
            if n == 0:
                return 0.0, 0.0, 0, 0, 0
            # CRÍTICA: fora dos limites estendidos
            mask_crit = (arr < v_prec_min) | (arr > v_crit_high)
            # PRECÁRIA: fora da adequada mas dentro da crítica
            mask_adeq = (arr >= v_adeq_min) & (arr <= v_adeq_max)
            mask_prec = (~mask_adeq) & (~mask_crit)
            n_prec = int(mask_prec.sum())
            n_crit = int(mask_crit.sum())
            drp = 100.0 * n_prec / n
            drc = 100.0 * n_crit / n
            return drp, drc, n_prec, n_crit, n

        if v_mt_list:
            arr_mt = np_d.array(v_mt_list)
            # MT: adequada [0,93;1,05] | precária inferior [0,90;0,93) | crítica V<0,90 ou V>1,06
            drp_mt, drc_mt, np_mt, nc_mt, nt_mt = _drp_drc(
                arr_mt, 0.93, 1.05, 0.90, 1.06)
            print(f"  V slack (650):          {v_slack:.4f} p.u.")
            print(f"  V RG60 (saída SVR):     min={v_rg60_min:.4f}  max={v_rg60_max:.4f}")
            print(f"\n  Tensão MT (4,16 kV) — PRODIST Módulo 8:")
            print(f"    Faixas (p.u.):        adequada [0,93;1,05] | precária [0,90;0,93)∪(1,05;1,06] | crítica fora")
            print(f"    V min / max / méd:    {arr_mt.min():.4f} / {arr_mt.max():.4f} / {arr_mt.mean():.4f}")
            print(f"    DRP (precária):       {drp_mt:6.3f} %   ({np_mt}/{nt_mt} amostras)   limite ≤ 3,0%   {'OK' if drp_mt <= 3.0 else 'NÃO CONFORME'}")
            print(f"    DRC (crítica):        {drc_mt:6.3f} %   ({nc_mt}/{nt_mt} amostras)   limite ≤ 0,5%   {'OK' if drc_mt <= 0.5 else 'NÃO CONFORME'}")

        if v_bt_list:
            arr_bt = np_d.array(v_bt_list)
            # BT: adequada [0,92;1,05] | precária inferior [0,87;0,92) | crítica V<0,87 ou V>1,06
            drp_bt, drc_bt, np_bt, nc_bt, nt_bt = _drp_drc(
                arr_bt, 0.92, 1.05, 0.87, 1.06)
            print(f"\n  Tensão BT (0,48 kV) — PRODIST Módulo 8:")
            print(f"    Faixas (p.u.):        adequada [0,92;1,05] | precária [0,87;0,92)∪(1,05;1,06] | crítica fora")
            print(f"    V min / max / méd:    {arr_bt.min():.4f} / {arr_bt.max():.4f} / {arr_bt.mean():.4f}")
            print(f"    DRP (precária):       {drp_bt:6.3f} %   ({np_bt}/{nt_bt} amostras)   limite ≤ 3,0%   {'OK' if drp_bt <= 3.0 else 'NÃO CONFORME'}")
            print(f"    DRC (crítica):        {drc_bt:6.3f} %   ({nc_bt}/{nt_bt} amostras)   limite ≤ 0,5%   {'OK' if drc_bt <= 0.5 else 'NÃO CONFORME'}")

        print(f"\n  Nota: indicadores DRP/DRC reportados como proxy diário; a norma exige janela semanal de 1008 amostras.")
    except Exception as e_diag:
        print(f"  Erro no diagnóstico: {e_diag}")

    # ══════════════════════════════════════════════════════════════════════
    # PÓS-PROCESSAMENTO COMPLETO
    # ══════════════════════════════════════════════════════════════════════
    tc = tc2
    print(f"\nStatus: {results.solver.status} | E2: {tc}")

    if tc in ('optimal', 'feasible', 'locallyOptimal'):
        svr      = data['svr']
        svr_phs  = sets_info['svr_phs']
        step_    = sets_info['tap_step']
        tap_min_ = sets_info['tap_min']
        hours    = sets_info['hours']
        trafos   = data.get('trafos', {})
        branches = data['branches']
        pvph_l   = sets_info['pvph']
        irr_d    = sets_info['irr']

        print("\n" + "="*68)
        print("  RESULTADOS PÓS-OPF")
        print("="*68)

        # ── (0) DESEQUILÍBRIO FD% — PRODIST Módulo 8 ─────────────────────
        # Indicador estatístico FD95% (proxy diário do FD95% regulatório
        # semanal). A média aritmética foi explicitamente abandonada pois
        # mascara picos críticos e não representa a natureza estatística
        # do indicador regulatório (PRODIST exige percentil 95).
        try:
            _fd = calc_fd_percent(model, data, sets_info)
            print("\n  (0) Desequilíbrio FD% — PRODIST Módulo 8 (proxy diário):")
            print(f"      FD95% sistêmico:         {_fd['fd95_system']:6.3f} %  "
                  f"(limite MT {FD_MT_MAX}% | BT {FD_BT_MAX}%)")
            print(f"      Pico instantâneo:        {_fd['fd_max_system']:6.3f} %"
                  f"  em t={_fd['t_peak_system']:>3}"
                  f"  ({(_fd['t_peak_system']*DT_MIN)//60:02d}h"
                  f"{(_fd['t_peak_system']*DT_MIN)%60:02d})"
                  f"  barra={_fd['b_peak_system']}")
            print(f"      Amostras em violação:    {_fd['n_viol']}/{_fd['n_total']}"
                  f"  ({_fd['pct_time_viol']:.2f} % do tempo×barras)")
            if _fd['critical_buses']:
                print("      Barras com FD95% > limite:")
                for b, fd95, lim in sorted(_fd['critical_buses'],
                                           key=lambda x: -x[1])[:5]:
                    print(f"        {b:>8s}  FD95%={fd95:5.2f}% (limite {lim:.1f}%)")
            else:
                print("      Conformidade FD95% em todas as barras analisadas.")
            print("      Limitação: captura apenas desequilíbrio de amplitude"
                  " (BFM-SOCP\n      sem ângulos explícitos).")
        except Exception as _e:
            print(f"  (0) FD%: cálculo indisponível ({_e})")

        # ── (a) OPERAÇÕES DE TAP: controle local (pré) vs. otimizado (pós) ─
        # Pré-OPF: simulação do controle local (banda morta + delay) — baseline
        # Pós-OPF: plano ótimo (limitado a N_TAP_MAX operações por fase)
        print("\n  (a) Operações de tap SVR — controle local vs. otimizado:")
        print(f"  {'Fase':>5} {'pos_i':>6} {'tap_i':>9} {'ops_LOCAL':>10} "
              f"{'pos_f_opt':>10} {'tap_f_opt':>10} {'ops_OPT':>9} {'redução':>10}")
        total_pre  = 0
        total_post = 0
        for ph in svr_phs:
            pos_i = tap_ops_pre[ph]['pos_init']
            tap_i = tap_min_ + pos_i * step_
            ops_b = tap_ops_pre[ph]['ops']     # do controle local
            pos_f = tap_fixed.get((ph, hours[-1]), pos_i)
            tap_f = tap_min_ + pos_f * step_
            seqp  = [tap_fixed.get((ph,t), pos_i) for t in hours]
            ops_a = sum(1 for k in range(1,len(seqp)) if seqp[k]!=seqp[k-1])
            if seqp[0] != pos_i: ops_a += 1
            total_pre  += ops_b
            total_post += ops_a
            # Redução percentual: (ops_pre - ops_pos) / ops_pre × 100
            if ops_b > 0:
                red = f"{(ops_b-ops_a)/ops_b*100:+.1f}%"
            else:
                red = 'n/a'
            print(f"  {ph:>5} {pos_i:>6} {tap_i:>9.5f} {ops_b:>10} "
                  f"{pos_f:>10} {tap_f:>10.5f} {ops_a:>9} {red:>10}")
        # Totais agregados
        if total_pre > 0:
            red_tot = f"{(total_pre-total_post)/total_pre*100:+.1f}%"
        else:
            red_tot = 'n/a'
        print(f"  {'TOTAL':>5} {'':>6} {'':>9} {total_pre:>10} "
              f"{'':>10} {'':>10} {total_post:>9} {red_tot:>10}")
        print(f"\n  Baseline (controle local OpenDSS): 33-39 ops esperadas com 100% PV")
        print(f"  Modelo Pyomo (controle local sim.): {total_pre} ops totais")
        print(f"  Modelo Pyomo (OPF — otimizado):     {total_post} ops totais")

        # ── (a.1) Caracterização qualitativa das operações de tap ───────
        # Com poucos eventos (~5 ops totais no horizonte diário), métricas
        # quantitativas como MTBO (mean time between operations) e
        # histograma horário possuem baixa significância estatística.
        # Reportamos uma CARACTERIZAÇÃO QUALITATIVA: em que períodos
        # ocorrem as operações e qual a coincidência com janelas de
        # carga/PV.
        ops_events = []   # [(t_event, ph, pos_prev, pos_new)]
        for ph in svr_phs:
            pos_i = tap_ops_pre[ph]['pos_init']
            seqp  = [tap_fixed.get((ph,t), pos_i) for t in hours]
            prev = pos_i
            for k, t in enumerate(hours):
                if seqp[k] != prev:
                    ops_events.append((t, ph, prev, seqp[k]))
                    prev = seqp[k]

        if ops_events:
            print(f"\n  (a.1) Caracterização qualitativa das operações de tap:")
            # Distribuição por janela (madrugada / manhã / tarde / noite)
            def _janela(t):
                hora = (t * DT_MIN) // 60
                if   0 <= hora <  6: return 'madrugada (00-06h)'
                elif 6 <= hora < 12: return 'manhã     (06-12h)'
                elif 12 <= hora < 18: return 'tarde     (12-18h)'
                else:                 return 'noite     (18-24h)'
            from collections import Counter
            janelas = Counter(_janela(t) for t, _, _, _ in ops_events)
            print(f"      Distribuição por janela:")
            for jan, cnt in sorted(janelas.items()):
                bar = '█' * cnt
                print(f"        {jan}:  {cnt}  {bar}")
            # Lista cronológica dos eventos
            print(f"      Sequência cronológica dos eventos:")
            for t, ph, prv, nxt in ops_events:
                hora = (t * DT_MIN) // 60
                minu = (t * DT_MIN) % 60
                direcao = '↑' if nxt > prv else '↓'
                tap_prv = tap_min_ + prv * step_
                tap_nxt = tap_min_ + nxt * step_
                print(f"        t={t:>3} ({hora:02d}h{minu:02d}) | fase {ph} | "
                      f"pos {prv:>2}→{nxt:>2} ({tap_prv:.4f}→{tap_nxt:.4f}) {direcao}")
        else:
            print(f"\n  (a.1) Nenhuma operação de tap registrada — sistema operou"
                  f" no tap inicial durante todo o horizonte.")

        # ── (b) PARÂMETROS DOS INVERSORES PV — fator de potência por hora ─
        print("\n  (b) Inversores PV — FP e Qpv por hora (amostra top-5 por Qpv pico):")
        t_peak_pv = max(hours, key=lambda t: irr_d[t])
        top5 = sorted(pvph_l,
                      key=lambda bp: abs(pyo.value(model.Qpv[bp[0],bp[1],t_peak_pv])),
                      reverse=True)[:5]
        # Cabeçalho por hora (00h–23h)
        print(f"  {'Bus':>8} {'Ph':>3} {'3ph':>5} ", end='')
        for h in range(0, 24, 3):
            print(f"  {h:02d}h-FP  {h:02d}h-Q", end='')
        print()
        for (b, ph) in top5:
            is3 = data['pv'][(b,ph)].get('is_3ph', False)
            print(f"  {b:>8} {ph:>3} {'Y' if is3 else 'N':>5} ", end='')
            for h in range(0, 24, 3):
                t_h = h * 6   # período correspondente à hora h
                Pavail = pyo.value(model.Pavail[b,ph,t_h]) * SBASE
                Qpv_kw = pyo.value(model.Qpv[b,ph,t_h]) * SBASE
                Pcurt  = pyo.value(model.Pcurt[b,ph,t_h]) * SBASE
                Pef    = Pavail - Pcurt
                S_op   = math.sqrt(max(Pef**2 + Qpv_kw**2, 1e-9))
                fp     = Pef / S_op if S_op > 0.01 else 1.0
                print(f"  {fp:7.4f} {Qpv_kw:7.2f}", end='')
            print()

        # ── (b.1) CURTAILMENT — % por inversor e sistêmico ───────────────
        # Pcurt(b,ph,t) é o corte de geração ativa em p.u. por fase.
        # Métrica relativa: % = ΣPcurt(t) / ΣPdisp(t) × 100, por inversor.
        # Sistêmico: razão das somas (todos inversores × todos períodos).
        print("\n  (b.1) Curtailment de geração PV (% relativo):")
        Pcurt_total = 0.0
        Pdisp_total = 0.0
        curt_per_inv = []   # [(bus, ph, %curt, kWh_curt, kWh_disp)]
        for (b, ph) in pvph_l:
            kwh_c = sum(pyo.value(model.Pcurt[b,ph,t]) * SBASE
                        for t in hours) * (DT_MIN/60.0)
            kwh_d = sum(pyo.value(model.Pavail[b,ph,t]) * SBASE
                        for t in hours) * (DT_MIN/60.0)
            Pcurt_total += kwh_c
            Pdisp_total += kwh_d
            pct = (100.0 * kwh_c / kwh_d) if kwh_d > 1e-6 else 0.0
            curt_per_inv.append((b, ph, pct, kwh_c, kwh_d))

        pct_sys = (100.0 * Pcurt_total / Pdisp_total) if Pdisp_total > 1e-6 else 0.0
        print(f"      Curtailment sistêmico:     {pct_sys:6.3f} %  "
              f"({Pcurt_total:7.2f} kWh / {Pdisp_total:8.2f} kWh disponíveis)")

        # Top-5 inversores mais cortados
        top_curt = sorted(curt_per_inv, key=lambda x: -x[2])[:5]
        if pct_sys < 0.01:
            print("      Nenhum inversor com curtailment significativo.")
            print("      → A hierarquia da F.O. (W_curt >> W_dv > W_un > W_tap > omega)")
            print("        cumpriu o papel de tratar o curtailment como último recurso.")
        else:
            print("      Top-5 inversores mais cortados:")
            print(f"        {'Bus':>8s} {'Ph':>3s}  {'%curt':>7s}  {'kWh_curt':>10s}  {'kWh_disp':>10s}")
            for b, ph, pct, kc, kd in top_curt:
                print(f"        {b:>8s} {ph:>3d}  {pct:6.3f}%   {kc:10.3f}  {kd:10.3f}")

        # ── (c) PERDAS NO NÚCLEO DOS TRAFOS — evolução temporal ──────────
        # P_NL(b,ph,t) = G_core · V(b,ph,t)²  [kW/fase]
        # Agregado por hora: soma dos 6 períodos de 10 min
        print("\n  (c) Perdas no núcleo pós-OPF P_NL=G·V² [kW/fase] — por hora:")
        pnl_post = {}   # {tid: {t: {ph: P_NL_kW}}}
        for (mv, lv, ph), bd in branches.items():
            if bd.get('level') != 'trafo': continue
            tid = bd.get('trafo_id', f'{mv}-{lv}')
            G   = bd.get('G_core_pu', 0.0)
            if tid not in pnl_post: pnl_post[tid] = {}
            for t in hours:
                V_mv = try_v(model, mv, ph, t)
                pnl  = G * V_mv**2 * SBASE
                if t not in pnl_post[tid]: pnl_post[tid][t] = {}
                pnl_post[tid][t][ph] = pnl

        # Agregar por hora e comparar pré vs pós
        print(f"  {'Trafo':<12} {'P_NL_nom':>10} {'Pré_méd':>10} {'Pós_méd':>10} "
              f"{'Δ%':>8}  Evolução horária (kW/fase/hora)")
        pnl_hourly_all = {}   # para uso no cálculo térmico
        for tid, td in trafos.items():
            td_post = pnl_post.get(tid, {})
            td_pre  = pnl_pre_total.get(tid, {})
            phs_t   = td['phases']
            # Agregar por hora
            hourly_pnl = []   # lista de 24 valores [kW/fase/hora]
            for h in range(24):
                ts = list(range(h*6, min(h*6+6, len(hours))))
                sum_kw = sum(
                    sum(td_post.get(t,{}).get(ph,0.) for ph in phs_t)
                    for t in ts) / max(len(phs_t),1)
                hourly_pnl.append(sum_kw)
            pnl_hourly_all[tid] = hourly_pnl
            # Médias
            flat_post = [v for td_t in td_post.values() for v in td_t.values()]
            flat_pre  = [v for td_t in td_pre.values()  for v in td_t.values()]
            avg_post  = sum(flat_post)/max(len(flat_post),1)
            avg_pre   = sum(flat_pre) /max(len(flat_pre), 1)
            delta_pct = (avg_post-avg_pre)/max(avg_pre,1e-9)*100
            evol = ' '.join(f"{v:.2f}" for v in hourly_pnl[::4])   # a cada 4h
            print(f"  {tid:<12} {td['P_NL_nom_kw']:>10.4f} {avg_pre:>10.4f} "
                  f"{avg_post:>10.4f} {delta_pct:>8.2f}%  [{evol}]")

        # ── (d) TEMPERATURA DOS TRANSFORMADORES — IEEE C57.91 ────────────
        # θ_TO(t) = θ_TO_R · [P_LL(t)+P_NL(t) / P_LL_R+P_NL_R]^0.8
        # θ_g(t)  = θ_g_R  · [P_LL(t) / P_LL_R]^0.8
        # θ_hs(t) = θ_a + θ_TO(t) + θ_g(t)   (θ_a = 25°C padrão)
        THETA_AMB = 25.0   # °C temperatura ambiente
        # ───────────────────────────────────────────────────────────────
        # IMPORTANTE: o pico θ_hs_max é REPORTADO mas NÃO é a métrica
        # principal. O indicador chave do desgaste de isolamento é o
        # ENVELHECIMENTO INTEGRADO ∫F_AA(t)dt, reportado em (e) abaixo.
        # Um pico isolado não causa dano significativo; o que causa
        # desgaste é a permanência prolongada acima de 95°C.
        # ───────────────────────────────────────────────────────────────
        print("\n  (d) Temperatura do hot-spot θ_hs [°C] — IEEE C57.91:")
        print(f"  {'Trafo':<10} {'θ_hs_max':>10} {'θ_hs_med':>10} "
              f"{'t_pico':>8} {'h>95°C':>10} {'h>110°C':>10}")
        theta_hs_all = {}   # {tid: lista 24h}
        for tid, td in trafos.items():
            td_post = pnl_post.get(tid, {})
            phs_t   = td['phases']
            P_LL_R  = td['P_LL_R_kw']
            P_NL_R  = td['P_NL_nom_kw']
            tTO_R   = td['theta_TO_R']
            tg_R    = td['theta_g_R']
            theta_hs_h = []
            for h in range(24):
                ts = list(range(h*6, min(h*6+6, len(hours))))
                # Perdas de carga (Joule) por fase por período → média da hora
                P_LL_h = sum(
                    pyo.value(model.r[mv, lv, ph])
                    * (pyo.value(model.P[mv,lv,ph,t])**2
                       + pyo.value(model.Q[mv,lv,ph,t])**2)
                    / max(try_v(model,mv,ph,t)**2, 0.64) * SBASE
                    for (mv2,lv2,ph) in branches
                    if branches[(mv2,lv2,ph)].get('trafo_id')==tid
                    for t in ts
                    for mv, lv in [(mv2,lv2)]
                ) / max(len(ts)*len(phs_t), 1)
                # Perdas de núcleo média da hora
                P_NL_h = pnl_hourly_all.get(tid, [P_NL_R]*24)[h]
                # IEEE C57.91 — expoente 0.8
                ratio  = (P_LL_h + P_NL_h) / max(P_LL_R + P_NL_R, 1e-9)
                tTO_h  = tTO_R * (ratio ** 0.8)
                ratio_ll = P_LL_h / max(P_LL_R, 1e-9)
                tg_h   = tg_R * (ratio_ll ** 0.8)
                theta_hs_h.append(THETA_AMB + tTO_h + tg_h)
            theta_hs_all[tid] = theta_hs_h
            ths_max = max(theta_hs_h)
            ths_med = sum(theta_hs_h)/24
            t_pico  = theta_hs_h.index(ths_max)
            # Duração acima de limiares (IEEE C57.91 referencia 95°C
            # como início da aceleração de envelhecimento; 110°C como
            # limite nominal)
            h_above_95  = sum(1 for v in theta_hs_h if v > 95.0)
            h_above_110 = sum(1 for v in theta_hs_h if v > 110.0)
            print(f"  {tid:<10} {ths_max:>10.1f} {ths_med:>10.1f} "
                  f"{t_pico:>7d}h {h_above_95:>9d}h {h_above_110:>9d}h")

        # ── (e) VIDA ÚTIL DOS TRANSFORMADORES — Arrhenius ────────────────
        # F_AA(t) = exp[15000*(1/383 - 1/(273+θ_hs))]  (ref. 110°C = 383 K)
        # Perda de vida diária (%) = Σ F_AA(t)·Δt / (20anos × 8760h) × 100
        # Vida nominal: 20 anos @ 110°C → consumo normal = 0.0137%/dia
        # Regra dos 8°C: vida reduzida à metade para cada 8°C acima da ref.
        THETA_REF_K = 383.0   # 110°C em Kelvin (referência IEEE C57.91)
        VIDA_NOM_H  = 20 * 8760.0   # horas de vida nominal
        DT_H        = DT_MIN / 60.0 * 6   # horas por bloco horário (6 períodos)
        print("\n  (e) Vida útil dos transformadores — IEEE C57.91 / Arrhenius:")
        print("      [MÉTRICA PRINCIPAL DO DESGASTE TÉRMICO]")
        print(f"  Referência: θ_hs_ref=110°C | Vida nominal=20 anos | "
              f"Consumo normal≈0.0137%/dia")
        print(f"  {'Trafo':<12} {'θ_hs_max':>10} {'F_AA_max':>10} "
              f"{'Vida_diária%':>14} {'Aceleração':>12} {'Regra 8°C':>12}")
        for tid, ths_h in theta_hs_all.items():
            faa_list = []
            for ths in ths_h:
                ths_k = 273.0 + ths
                faa   = math.exp(15000.0 * (1.0/THETA_REF_K - 1.0/ths_k))
                faa_list.append(faa)
            # Perda de vida diária: integral de F_AA·dt / vida_nominal
            perda_vida = sum(f * DT_H for f in faa_list) / VIDA_NOM_H * 100.0
            aceleracao = perda_vida / 0.0137   # 0.0137% = consumo normal/dia
            faa_max    = max(faa_list)
            ths_max    = max(ths_h)
            # Regra dos 8°C: log2(F_AA_max) ≈ ΔT/8
            excesso_C  = max(ths_max - 110.0, 0.0)
            fator_8    = 2.0 ** (excesso_C / 8.0)
            print(f"  {tid:<12} {ths_max:>10.1f} {faa_max:>10.3f} "
                  f"{perda_vida:>14.5f} {aceleracao:>12.2f}x {fator_8:>12.2f}x")

        # Síntese: envelhecimento integrado do dia agregado
        total_perda = 0.0
        n_trafos = 0
        max_perda = 0.0
        worst_tid = None
        for tid, ths_h in theta_hs_all.items():
            faa_list = [math.exp(15000.0 * (1.0/THETA_REF_K - 1.0/(273.0+v))) for v in ths_h]
            pv = sum(f * DT_H for f in faa_list) / VIDA_NOM_H * 100.0
            total_perda += pv
            n_trafos += 1
            if pv > max_perda:
                max_perda = pv
                worst_tid = tid
        if n_trafos > 0:
            print(f"\n      Envelhecimento integrado (síntese):")
            print(f"        Perda de vida média/trafo:  {total_perda/n_trafos:.5f} %/dia")
            print(f"        Pior trafo:                 {worst_tid}  ({max_perda:.5f} %/dia)")
            print(f"        Referência operação normal: 0,0137 %/dia (= 100% a 110°C)")
            ratio_worst = max_perda / 0.0137 if 0.0137 > 0 else 0
            print(f"        Razão pior/nominal:         {ratio_worst:.3f}×  "
                  f"(<1 = abaixo da operação normal)")

        # ════════════════════════════════════════════════════════════════
        # (f) CLASSIFICAÇÃO DE INVERSORES E INFERÊNCIA DA CURVA VOLT-VAR
        # ════════════════════════════════════════════════════════════════
        # O OPF entrega o ENVELOPE ÓTIMO Q*(t) por inversor. Esta etapa
        # interpreta o despacho ótimo classificando cada inversor em
        # MODOS DE CONTROLE distintos, recomendando ao operador a
        # configuração mais adequada:
        #
        # GRUPO A — FP fixo capacitivo: Q* ≈ +κ·P_disp(t) durante todo o
        #          dia útil. O inversor é mantido na saturação
        #          capacitiva. Recomendação: configurar 'Constant Power
        #          Factor' com FP=0.85 indutivo (capacitivo do lado da
        #          rede).
        #
        # GRUPO B — Volt-VAR ativa: Q* modula entre zonas (capacitivo,
        #          zona morta, e/ou indutivo) ao longo do dia. A tensão
        #          local cruza V_ref. Recomendação: configurar a curva
        #          IEEE 1547 com (V1, V2, V3, V4, Q_max) inferidos.
        #
        # GRUPO C — Inversor inativo: Q* ≈ 0 durante todo o dia (PV
        #          gerando mas sem necessidade de suporte reativo na
        #          barra). Recomendação: configurar 'Unity Power Factor'.
        print("\n  (f) Classificação dos inversores PV e inferência Volt-VAR:")
        import numpy as np_vv

        t_vv = sets_info['t_vv']
        pvph_l = sets_info['pvph']
        _tan_fp_pp = math.sqrt(1.0 - FP_MIN_VV**2) / FP_MIN_VV

        # Para cada inversor, coletar pares (V, Q, P_disp) na janela VV
        # com Pavail > 0 e classificar
        groups = {'A': [], 'B': [], 'C': []}
        inv_data = {}    # {(b,ph): {V_arr, Q_arr, Pdisp_arr, classification}}

        for (b, ph) in pvph_l:
            pairs = []
            for t in hours:
                if not (t_vv[0] <= t <= t_vv[1]):
                    continue
                Pav = pyo.value(model.Pavail[b,ph,t])
                if Pav < 1e-4:
                    continue
                Vt = try_v(model, b, ph, t)
                Qt = pyo.value(model.Qpv[b,ph,t]) * SBASE   # kVAr
                Pt = Pav * SBASE                             # kW
                pairs.append((Vt, Qt, Pt))
            if len(pairs) < 5:
                continue

            V_arr = np_vv.array([p[0] for p in pairs])
            Q_arr = np_vv.array([p[1] for p in pairs])
            P_arr = np_vv.array([p[2] for p in pairs])
            Q_max_inv = float(P_arr.max() * _tan_fp_pp)   # Q máx teórico
            if Q_max_inv < 0.1:
                continue

            # Métricas de classificação
            # - frac_sat_cap: fração do tempo em saturação capacitiva (Q≥0.9·Qmax_t)
            # - frac_dead:   fração na zona morta (|Q|<0.05·Qmax_inv)
            # - frac_ind:    fração em saturação indutiva (Q≤-0.9·Qmax_t)
            # Qmax_t = κ·Pdisp(t) é o limite instantâneo
            Q_lim_t = _tan_fp_pp * P_arr
            mask_cap_sat  = Q_arr >= 0.9 * Q_lim_t
            mask_ind_sat  = Q_arr <= -0.9 * Q_lim_t
            mask_dead     = np_vv.abs(Q_arr) < 0.05 * Q_max_inv
            frac_cap = float(mask_cap_sat.sum() / len(Q_arr))
            frac_dead = float(mask_dead.sum() / len(Q_arr))
            frac_ind = float(mask_ind_sat.sum() / len(Q_arr))

            # Classificação
            if frac_cap >= 0.80 and frac_dead < 0.10:
                # Grupo A: saturado capacitivo (FP fixo)
                classification = 'A'
            elif frac_dead >= 0.60 and abs(Q_arr).max() < 0.10 * Q_max_inv:
                # Grupo C: inativo (Q≈0)
                classification = 'C'
            else:
                # Grupo B: Volt-VAR ativa (modula entre zonas)
                classification = 'B'

            inv_data[(b, ph)] = {
                'V_arr': V_arr, 'Q_arr': Q_arr, 'P_arr': P_arr,
                'Q_max_inv': Q_max_inv,
                'frac_cap': frac_cap, 'frac_dead': frac_dead,
                'frac_ind': frac_ind, 'classification': classification,
            }
            groups[classification].append((b, ph))

        n_total = sum(len(g) for g in groups.values())
        print(f"    Inversores analisados: {n_total} "
              f"(A={len(groups['A'])}, B={len(groups['B'])}, C={len(groups['C'])})")

        # ── GRUPO A: FP fixo capacitivo ────────────────────────────────
        if groups['A']:
            print(f"\n    GRUPO A — FP fixo capacitivo ({len(groups['A'])} inversores):")
            print("      Recomendação: configurar 'Constant Power Factor' com FP = 0.85")
            print("                    (capacitivo do lado da rede; indutivo do lado do inv.)")
            print(f"      {'Bus':>6} {'Ph':>3} {'%cap':>6} {'%dead':>6} {'Q_méd[kVAr]':>13}")
            sample_A = sorted(groups['A'],
                              key=lambda bp: -inv_data[bp]['Q_max_inv'])[:5]
            for (b, ph) in sample_A:
                d = inv_data[(b, ph)]
                q_med = float(d['Q_arr'].mean())
                print(f"      {b:>6} {ph:>3} {d['frac_cap']*100:>5.1f}% "
                      f"{d['frac_dead']*100:>5.1f}% {q_med:>13.2f}")
            if len(groups['A']) > 5:
                print(f"      ... e mais {len(groups['A'])-5} inversores no Grupo A")

        # ── GRUPO C: inativos (FP unitário) ────────────────────────────
        if groups['C']:
            print(f"\n    GRUPO C — Inversores inativos ({len(groups['C'])} inversores):")
            print("      Recomendação: configurar 'Unity Power Factor' (FP = 1.00)")
            print("                    A tensão local não exige suporte reativo.")
            print(f"      {'Bus':>6} {'Ph':>3} {'%dead':>6} {'V_méd':>7}")
            sample_C = groups['C'][:5]
            for (b, ph) in sample_C:
                d = inv_data[(b, ph)]
                v_med = float(d['V_arr'].mean())
                print(f"      {b:>6} {ph:>3} {d['frac_dead']*100:>5.1f}% "
                      f"{v_med:>7.4f}")
            if len(groups['C']) > 5:
                print(f"      ... e mais {len(groups['C'])-5} inversores no Grupo C")

        # ── GRUPO B: Volt-VAR ativa — INFERÊNCIA DA CURVA ──────────────
        if not groups['B']:
            print("\n    GRUPO B — Nenhum inversor com Volt-VAR ativa identificado.")
            print("             Todos os inversores estão saturados ou inativos.")
        else:
            print(f"\n    GRUPO B — Volt-VAR ativa ({len(groups['B'])} inversores):")
            print("      Recomendação: configurar curva Volt-VAR IEEE 1547-2018 Cat.B")
            print("                    com parâmetros inferidos do envelope ótimo.")

            # ──────────────────────────────────────────────────────────
            # AJUSTE COM RESTRIÇÕES DE SEPARAÇÃO MÍNIMA (Opção 2)
            # ──────────────────────────────────────────────────────────
            # Para garantir que a curva inferida seja FISICAMENTE
            # IMPLEMENTÁVEL em um inversor real, aplicamos separações
            # mínimas entre os pontos de inflexão, coerentes com as
            # recomendações da IEEE 1547-2018 (Tabela 8 da norma):
            #
            #   • Rampa capacitiva V2-V1  ≥ 0.020 p.u. (transição suave)
            #   • Zona morta V3-V2        ≥ 0.010 p.u. (estabilidade)
            #   • Rampa indutiva V4-V3    ≥ 0.020 p.u. (transição suave)
            #
            # Estas restrições priorizam a IMPLEMENTABILIDADE da curva
            # em detrimento do ajuste perfeito aos pontos do envelope.
            # O RMSE resultante quantifica a perda de informação ao
            # forçar uma curva paramétrica razoável.
            DV_CAP_MIN  = 0.020  # rampa capacitiva mínima (V2 - V1)
            DV_DEAD_MIN = 0.010  # zona morta mínima (V3 - V2)
            DV_IND_MIN  = 0.020  # rampa indutiva mínima (V4 - V3)

            def _fit_vv(V_arr, Q_arr, Q_max_obs):
                """Ajuste com restrições de separação mínima (Opção 2).

                Estratégia em duas etapas:
                  1) Extrair valores brutos por percentis (como antes)
                  2) Aplicar projeção sequencial para garantir as
                     separações mínimas entre os pontos de inflexão
                """
                if Q_max_obs < 1e-6:
                    return 0.92, 0.98, 1.02, 1.08

                # Zonas operacionais nos pontos observados
                mask_cap = Q_arr > 0.20 * Q_max_obs
                mask_ind = Q_arr < -0.20 * Q_max_obs
                mask_dead = np_vv.abs(Q_arr) < 0.05 * Q_max_obs
                mask_cap_sat = Q_arr > 0.85 * Q_max_obs
                mask_ind_sat = Q_arr < -0.85 * Q_max_obs

                # Etapa 1: valores brutos (mesma lógica da versão anterior)
                V2_raw = (float(np_vv.percentile(V_arr[mask_cap], 90))
                          if mask_cap.sum() > 2 else 0.98)
                V3_raw = (float(np_vv.percentile(V_arr[mask_ind], 10))
                          if mask_ind.sum() > 2 else 1.02)
                V1_raw = (float(np_vv.percentile(V_arr[mask_cap_sat], 80))
                          if mask_cap_sat.sum() > 2 else V2_raw - 0.03)
                V4_raw = (float(np_vv.percentile(V_arr[mask_ind_sat], 20))
                          if mask_ind_sat.sum() > 2 else V3_raw + 0.03)

                # Etapa 2: projeção sequencial para garantir separações
                # Centro de referência: ponto médio entre V2 e V3 (zona morta)
                V_center = 0.5 * (V2_raw + V3_raw)

                # V2 e V3 obtidos preservando a zona morta
                V2 = min(V2_raw, V_center - 0.5 * DV_DEAD_MIN)
                V3 = max(V3_raw, V_center + 0.5 * DV_DEAD_MIN)
                # Garantir V3 - V2 ≥ DV_DEAD_MIN (após truncamento)
                if V3 - V2 < DV_DEAD_MIN:
                    V2 = V_center - 0.5 * DV_DEAD_MIN
                    V3 = V_center + 0.5 * DV_DEAD_MIN

                # V1 e V4 garantindo separação mínima da rampa
                V1 = min(V1_raw, V2 - DV_CAP_MIN)
                V4 = max(V4_raw, V3 + DV_IND_MIN)

                # Bounds adicionais para realismo físico
                # (IEEE 1547 Cat.B permite V1 ≥ 0.88, V4 ≤ 1.10)
                V1 = max(V1, 0.88)
                V4 = min(V4, 1.10)
                # Re-verificar separações após bounds (V2, V3 são prioritários)
                if V2 - V1 < DV_CAP_MIN:
                    V1 = V2 - DV_CAP_MIN
                if V4 - V3 < DV_IND_MIN:
                    V4 = V3 + DV_IND_MIN
                return V1, V2, V3, V4

            def _vv_eval(V, V1, V2, V3, V4, Qmax):
                if V <= V1: return Qmax
                if V <= V2: return Qmax * (V2 - V) / max(V2 - V1, 1e-6)
                if V <= V3: return 0.0
                if V <= V4: return -Qmax * (V - V3) / max(V4 - V3, 1e-6)
                return -Qmax

            print(f"      {'Bus':>6} {'Ph':>3} {'V1':>7} {'V2':>7} {'V3':>7} {'V4':>7} "
                  f"{'Q_max':>10} {'RMSE%':>7}")
            inferred_B = []
            # Ordenar Grupo B pela amplitude de Q (top primeiro)
            sample_B = sorted(groups['B'],
                              key=lambda bp: -(inv_data[bp]['Q_arr'].max() -
                                                inv_data[bp]['Q_arr'].min()))[:10]
            for (b, ph) in sample_B:
                d = inv_data[(b, ph)]
                V_arr, Q_arr = d['V_arr'], d['Q_arr']
                Q_max_obs = float(np_vv.percentile(np_vv.abs(Q_arr), 98))
                V1, V2, V3, V4 = _fit_vv(V_arr, Q_arr, Q_max_obs)
                Q_pred = np_vv.array([_vv_eval(v, V1, V2, V3, V4, Q_max_obs)
                                      for v in V_arr])
                rmse = float(np_vv.sqrt(np_vv.mean((Q_arr - Q_pred)**2)))
                rmse_pct = rmse / max(Q_max_obs, 1e-6) * 100
                inferred_B.append({
                    'bus': b, 'ph': ph,
                    'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4,
                    'Q_max': Q_max_obs, 'rmse_pct': rmse_pct})
                print(f"      {b:>6} {ph:>3}  {V1:>7.4f} {V2:>7.4f} "
                      f"{V3:>7.4f} {V4:>7.4f}  {Q_max_obs:>9.2f}k {rmse_pct:>6.1f}%")
            if len(groups['B']) > 10:
                print(f"      ... e mais {len(groups['B'])-10} inversores no Grupo B")

            # Curva agregada para o Grupo B
            if inferred_B:
                V1_m = float(np_vv.median([c['V1'] for c in inferred_B]))
                V2_m = float(np_vv.median([c['V2'] for c in inferred_B]))
                V3_m = float(np_vv.median([c['V3'] for c in inferred_B]))
                V4_m = float(np_vv.median([c['V4'] for c in inferred_B]))
                print("\n      Curva agregada Volt-VAR (mediana do Grupo B):")
                print(f"        V1={V1_m:.4f}  V2={V2_m:.4f}  "
                      f"V3={V3_m:.4f}  V4={V4_m:.4f}")
                print("      Referência IEEE 1547-2018 Cat.B:")
                print("        V1=0.9200  V2=0.9800  V3=1.0200  V4=1.0800  "
                      "Q_max=0.44·S_nom")

        # Síntese da estratégia recomendada
        print("\n    SÍNTESE — modos de controle recomendados:")
        n_A, n_B, n_C = len(groups['A']), len(groups['B']), len(groups['C'])
        n_tot = max(n_A + n_B + n_C, 1)
        print(f"      • FP fixo capacitivo (Grupo A):    {n_A:>3} "
              f"({100*n_A/n_tot:>4.1f}%)")
        print(f"      • Volt-VAR IEEE 1547 (Grupo B):    {n_B:>3} "
              f"({100*n_B/n_tot:>4.1f}%)")
        print(f"      • FP unitário/inativo (Grupo C):   {n_C:>3} "
              f"({100*n_C/n_tot:>4.1f}%)")

        # ────────────────────────────────────────────────────────────────
        # PERSISTÊNCIA: salvar TODOS os 84 inversores em CSV
        # ────────────────────────────────────────────────────────────────
        try:
            import csv
            csv_path = '/content/inversores_classificacao.csv'
            try:
                import os as _os
                if not _os.path.isdir('/content'):
                    csv_path = 'inversores_classificacao.csv'
            except Exception:
                csv_path = 'inversores_classificacao.csv'

            with open(csv_path, 'w', newline='') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    'bus', 'phase', 'is_3ph', 'kva_inv',
                    'grupo', 'recomendacao',
                    'V_min', 'V_max', 'V_med',
                    'Q_min_kVAr', 'Q_max_kVAr', 'Q_med_kVAr',
                    'P_disp_max_kW', 'Q_max_teorico_kVAr',
                    'frac_cap_sat', 'frac_dead', 'frac_ind_sat',
                    'V1', 'V2', 'V3', 'V4', 'rmse_pct'
                ])
                inferred_lookup = {(c['bus'], c['ph']): c for c in (inferred_B if 'inferred_B' in dir() else [])}
                recommend = {
                    'A': 'FP_fixo_capacitivo_0.85',
                    'B': 'Volt-VAR_IEEE1547_curva_inferida',
                    'C': 'FP_unitario_1.00',
                }
                pv_meta = data.get('pv_meta', {})
                for (b, ph), d in inv_data.items():
                    meta = pv_meta.get(b, {})
                    inferred = inferred_lookup.get((b, ph), {})
                    writer.writerow([
                        b, ph,
                        'Y' if meta.get('type') == '3ph' else 'N',
                        f"{meta.get('kva', 0):.2f}",
                        d['classification'],
                        recommend.get(d['classification'], '-'),
                        f"{float(d['V_arr'].min()):.5f}",
                        f"{float(d['V_arr'].max()):.5f}",
                        f"{float(d['V_arr'].mean()):.5f}",
                        f"{float(d['Q_arr'].min()):.3f}",
                        f"{float(d['Q_arr'].max()):.3f}",
                        f"{float(d['Q_arr'].mean()):.3f}",
                        f"{float(d['P_arr'].max()):.3f}",
                        f"{d['Q_max_inv']:.3f}",
                        f"{d['frac_cap']:.3f}",
                        f"{d['frac_dead']:.3f}",
                        f"{d['frac_ind']:.3f}",
                        f"{inferred.get('V1', ''):>.4f}" if inferred else '',
                        f"{inferred.get('V2', ''):>.4f}" if inferred else '',
                        f"{inferred.get('V3', ''):>.4f}" if inferred else '',
                        f"{inferred.get('V4', ''):>.4f}" if inferred else '',
                        f"{inferred.get('rmse_pct', ''):>.1f}" if inferred else '',
                    ])
            print(f"\n    [CSV] Salvo: {csv_path} ({n_total} inversores)")
        except Exception as e_csv:
            print(f"    [CSV] Aviso: {e_csv}")

        # ────────────────────────────────────────────────────────────────
        # PLOTAGEM: curva agregada Volt-VAR sobre os pontos do Grupo B
        # ────────────────────────────────────────────────────────────────
        if groups['B'] and inferred_B:
            try:
                import matplotlib.pyplot as plt_vv
                import matplotlib.colors as mcolors_vv

                fig_vv, axs_vv = plt_vv.subplots(1, 2, figsize=(16, 6))
                fig_vv.suptitle(
                    'Volt-VAR curve inference — Group B (inverters with active modulation)',
                    fontsize=13, fontweight='bold')

                # SUBPLOT 1: Scatter (V, Q) de todos os inversores do Grupo B
                ax1 = axs_vv[0]
                all_V, all_Q, all_irr = [], [], []
                for (b, ph) in groups['B']:
                    d = inv_data[(b, ph)]
                    all_V.extend(d['V_arr'].tolist())
                    all_Q.extend(d['Q_arr'].tolist())
                    # cor por P_disp (substituto da irradiância individual)
                    Pmax_loc = max(d['P_arr'].max(), 1e-6)
                    all_irr.extend((d['P_arr'] / Pmax_loc).tolist())
                sc = ax1.scatter(all_V, all_Q, c=all_irr, cmap='YlOrRd',
                                 s=30, alpha=0.7, edgecolors='gray', linewidth=0.3,
                                 label='OPF points (envelope)')
                plt_vv.colorbar(sc, ax=ax1, label='P_disp (normalized)')

                # Curva agregada
                V1_m = float(np_vv.median([c['V1'] for c in inferred_B]))
                V2_m = float(np_vv.median([c['V2'] for c in inferred_B]))
                V3_m = float(np_vv.median([c['V3'] for c in inferred_B]))
                V4_m = float(np_vv.median([c['V4'] for c in inferred_B]))
                Qmax_m = float(np_vv.median([c['Q_max'] for c in inferred_B]))
                v_curve = np_vv.linspace(min(0.92, V1_m-0.01),
                                          max(1.08, V4_m+0.01), 400)
                def _vv_curve(v):
                    if v <= V1_m: return Qmax_m
                    if v <= V2_m: return Qmax_m * (V2_m - v) / max(V2_m - V1_m, 1e-6)
                    if v <= V3_m: return 0.0
                    if v <= V4_m: return -Qmax_m * (v - V3_m) / max(V4_m - V3_m, 1e-6)
                    return -Qmax_m
                q_curve = np_vv.array([_vv_curve(v) for v in v_curve])
                ax1.plot(v_curve, q_curve, '-', lw=2.8, color='#185FA5',
                         label=f'Inferred aggregated curve')

                # Curva referência IEEE 1547 (para comparação)
                V1r, V2r, V3r, V4r = 0.92, 0.98, 1.02, 1.08
                def _vv_ref(v):
                    if v <= V1r: return Qmax_m
                    if v <= V2r: return Qmax_m * (V2r - v) / (V2r - V1r)
                    if v <= V3r: return 0.0
                    if v <= V4r: return -Qmax_m * (v - V3r) / (V4r - V3r)
                    return -Qmax_m
                q_ref = np_vv.array([_vv_ref(v) for v in v_curve])
                ax1.plot(v_curve, q_ref, '--', lw=1.8, color='#BA7517',
                         label='IEEE 1547 Cat.B reference')

                # Marcadores dos pontos de inflexão
                for vbp, lbl in [(V1_m, 'V1'), (V2_m, 'V2'),
                                  (V3_m, 'V3'), (V4_m, 'V4')]:
                    ax1.axvline(vbp, color='#185FA5', lw=0.7, ls=':', alpha=0.5)
                    ax1.text(vbp, ax1.get_ylim()[1]*0.92, lbl,
                             ha='center', fontsize=9, color='#185FA5',
                             fontweight='bold')
                ax1.axhline(0, color='gray', lw=0.6, ls=':')
                ax1.set_xlabel('Voltage V [p.u.]')
                ax1.set_ylabel('Q_pv [kVAr/phase]')
                ax1.set_title(f'Group B OPF points (n={len(groups["B"])} inverters)\n'
                              f'Aggregated curve: V1={V1_m:.4f} V2={V2_m:.4f} '
                              f'V3={V3_m:.4f} V4={V4_m:.4f}', fontsize=10)
                ax1.legend(fontsize=8, loc='lower left')
                ax1.grid(alpha=0.3)

                # SUBPLOT 2: Distribuição da classificação (pizza/barras)
                ax2 = axs_vv[1]
                grupos_lbl = ['Group A\n(fixed cap. PF)',
                              'Group B\n(Volt-VAR)',
                              'Group C\n(unity PF)']
                grupos_n = [len(groups['A']), len(groups['B']), len(groups['C'])]
                grupos_color = ['#E05C3A', '#2A7EBA', '#1D9E75']
                bars = ax2.bar(grupos_lbl, grupos_n, color=grupos_color,
                                edgecolor='black', alpha=0.85)
                for bar, n in zip(bars, grupos_n):
                    pct = 100 * n / max(sum(grupos_n), 1)
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.5,
                             f'{n}\n({pct:.1f}%)',
                             ha='center', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Number of inverters')
                ax2.set_title('PV inverter classification by control mode\n'
                              f'Total analyzed: {sum(grupos_n)} inverters',
                              fontsize=10)
                ax2.set_ylim(0, max(grupos_n) * 1.20)
                ax2.grid(alpha=0.3, axis='y')

                plt_vv.tight_layout(pad=2.0)
                plt_vv.savefig('opf_bfm_voltvar_inferida.png',
                                dpi=150, bbox_inches='tight')
                print("\n    [PNG] Salvo: opf_bfm_voltvar_inferida.png")
                plt_vv.show()
            except Exception as e_plot_vv:
                print(f"    [PNG] Aviso: {e_plot_vv}")

        # ── Perdas Joule totais (ramos) ───────────────────────────────────
        # ── Perdas Joule totais (ramos) ───────────────────────────────────
        loss_pu = sum(
            pyo.value(model.r[i,j,ph])
            * (pyo.value(model.P[i,j,ph,t])**2 + pyo.value(model.Q[i,j,ph,t])**2)
            / max(try_v(model, i, ph, t)**2, 0.64)
            for (i,j,ph) in branches for t in hours)
        print(f"\n  Perdas Joule totais r·(P²+Q²)/V²: {loss_pu*SBASE:.2f} kW·período")
        print(f"  Perdas Joule médias/período       : {loss_pu*SBASE/len(hours):.2f} kW")
        print(f"\n  F.O. = {pyo.value(model.obj):.4f}")

        # ── Gráficos de resultados ──────────────────────────────────────
        pv_data = data['pv']
        print("\n  Gerando gráficos...")

        # Plots tradicionais (try unificado — se um falha, os seguintes
        # podem não ser gerados; mas o heatmap fica em try próprio abaixo)
        try:
            plot_results(model, data, sets_info)
            plot_voltage_mt(model, data, sets_info)
            plot_voltage_profile_and_vv(model, data, sets_info)
            plot_voltage_temporal_comparison(model, data, sets_info)
        except Exception as e_plot:
            print(f"  Aviso gráficos (principais): {e_plot}")

        # Heatmap consolidado V[bus-phase × tempo] — try próprio para
        # garantir que seja gerado mesmo se algum dos plots anteriores
        # falhar.
        try:
            plot_voltage_heatmap(model, data, sets_info)
        except Exception as e_heat:
            print(f"  Aviso gráficos (heatmap): {e_heat}")
    else:
        print(f"\n  [!] Solver não convergiu: {tc}")