# -*- coding: utf-8 -*-
"""opf_core.py — nucleo REUTILIZAVEL (dados, OPF livre, classificacao, export)
Extraido e mantido em sincronia com opf_revisado.py; SEM os magics de Colab
e SEM o bloco __main__, para ser importavel como modulo comum aos Codigos 1 e 2.
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
#   W_curt >> W_dv > W_unbal > W_tap_ops > W_Q_USE > W_ramp
# Significado:
#   1) Curtailment é último recurso (peso máximo)
#   2) Desvio |V - V_ref| é a âncora principal de qualidade de tensão
#   3) Desequilíbrio entre fases é secundário
#   4) Operações de tap são preferencialmente evitadas em favor de Qpv,
#      mas autorizadas quando Qpv não basta
#   5) Uso de Qpv é penalizado levemente: o reativo NÃO é mais livre,
#      tem um custo marginal que evita saturar todos os inversores sem
#      necessidade operacional (responde à crítica do despacho "gratuito")
#   6) Suavização de Qpv é fina (não compete com objetivos maiores)
W_DV       = 20.0   # desvio |V - V_ref| — âncora principal (PRIORIDADE 1)
              # CALIBRAÇÃO OPÇÃO C (MDL ordinal): subido de 10 para 20 para
              # zerar empate com W_TAP=10 e satisfazer a hierarquia revista
              # CURT > DV > TAP > UNBAL > Q_USE > RAMP. Justificativa
              # operacional: FD% opera com folga grande sob o limite PRODIST
              # (~0,97% vs 2%), então o custo marginal de mais desequilíbrio
              # é desprezível; já cada operação mecânica de tap tem desgaste
              # documentado (IEEE C57.91). Razões pós-ajuste: W_CURT/W_DV=2,5×,
              # W_DV/W_TAP=2×, W_TAP/W_UNBAL=3,03×, todas com separação clara.
              # V_ref = 0.95 p.u. nas barras MT com carga modelo 1 (P-const);
              # V_ref = 1.0167 p.u. nas demais (modelos 2/5 ou sem carga MT)
W_UNBAL    = 3.3    # desequilíbrio |Vi - Vj| (PRIORIDADE 2)
W_TAP_OPS  = 10.0   # eventos mecânicos de tap (PRIORIDADE 3)
              # Penaliza operações mecânicas; deve permanecer ACIMA de
              # W_Q_USE para que o solver prefira usar Qpv antes de novos taps.
W_Q_USE    = 0.50   # penalização leve de |Qpv| (PRIORIDADE 4)
              # O DESPACHO REATIVO NÃO É MAIS LIVRE: cada kVAr injetado tem
              # custo marginal. Evita que todos os inversores saturem em Q
              # capacitivo sem necessidade. Ajuste: se A=100% persistir,
              # testar 1.00; se surgirem violações/taps demais, reduzir p/ 0.25.
W_RAMP_DIA  = 0.10  # suavização ΔQpv (fora pico solar)
W_RAMP_PICO = 0.01  # suavização ΔQpv (pico solar)
IRR_PICO   = 0.50   # threshold pico solar
W_CURT     = 50.0   # curtailment PV — preservar geração disponível (TOPO)

# ── PERDAS NO NÚCLEO DOS TRAFOS NA F.O. ──────────────────────────────────
# P_NL(b,ph,t) = G_core · V_mv²  (perda de ferro, proporcional a V²).
# Minimizar este termo EMPURRA A TENSÃO PARA BAIXO nas barras primárias dos
# transformadores. Numa rede subtensionada como esta, isso ENTRA EM CONFLITO
# direto com o suporte de tensão (tap em 1,05, Q capacitivo) que o resto do
# objetivo provê. O efeito principal recai sobre a decisão de TAP (Estágio 1):
# o termo desencoraja o boost para 1,05. Esperar: perdas de núcleo menores,
# porém maior desvio de tensão / risco de subtensão. É um trade-off real,
# não um ganho gratuito.
#
# ESCALA: a energia de núcleo é expressa em kW·período (·SBASE). Com V≈1 a
# soma vale ~2300 kW·período, ordem de grandeza comparável à F.O. (~4300).
# Por isso o termo é potente: W_CORE pequeno já pesa. Sugestão de calibração:
#   começar em 0.02–0.05 e observar o impacto em DRP/desvio antes de subir.
# ATIVADO (revisão): W_CORE=0.05 — topo da própria faixa sugerida acima, para
# que o efeito fique visível neste run exploratório (objetivo permanece
# CONVEXO: V² entra com sinal POSITIVO numa minimização — soma de termo
# linear + quadrático convexo é convexo; CPLEX resolve QCQP/MIQCP convexo
# nativamente, é a mesma classe de problema, não há perda de exatidão da
# relaxação SOCP nem necessidade de mudar o solver). Comece o teste assim e
# faça um sweep (0, 0.02, 0.05, 0.1, ...) com N_PERIODS reduzido para achar
# o ponto em que o piso de tensão 0,92 p.u. (BT) passa a ficar ativo.
ENABLE_CORE_LOSS_OBJ = True   # inclui o termo no objetivo (multiplicado por W_CORE)
W_CORE     = 0.05   # peso da energia de perdas no núcleo [por kW·período]
                    # 0.0 = inerte. >0 ativa (ver nota de escala acima).

# ── ACOPLAMENTO MÚTUO ENTRE FASES (ramos MT) ─────────────────────────────
# O OPF é, por padrão, DESACOPLADO por fase (R4 não tem termos cruzados).
# Isto subestima o efeito de FV no tap (visto no OpenDSS: +6 ops; aqui ~+2).
# Esta seção adiciona uma correção de 1ª ordem ao R4 nos ramos MT, sem
# tornar o problema não-linear (mantém R4 uma igualdade LINEAR — apenas mais
# termos com COEFICIENTES FIXOS multiplicando P/Q das OUTRAS fases do MESMO
# ramo). Não afeta a convexidade nem a classe do problema (MISOCP).
#
# DERIVAÇÃO (de V=Z·I, S=V·I*, com a aproximação de rotação nominal
# balanceada V_φ≈|V_φ|·K_φ, K_a=1, K_b=e^{-j120°}, K_c=e^{+j120°}):
#   V_j^φ ≈ V_i^φ − r^φ·P^φ − x^φ·Q^φ − Σ_{ψ≠φ}(M^P_φψ·P^ψ + M^Q_φψ·Q^ψ)
# Para o vizinho "seguinte" (a→b→c→a) com impedância mútua aproximada por
# R_mut≈ρR·r^φ, X_mut≈ρX·x^φ (ρR,ρX = fração mútuo/próprio):
#   M^P = -0,5·R_mut - (√3/2)·X_mut      M^Q = +(√3/2)·R_mut - 0,5·X_mut
# Para o vizinho "anterior":
#   M^P = -0,5·R_mut + (√3/2)·X_mut      M^Q = -(√3/2)·R_mut - 0,5·X_mut
# (faz φ=ψ recair exatamente no termo próprio existente — checagem de
# consistência: M^P_φφ=r^φ, M^Q_φφ=x^φ quando ρ=1, K_φ/K_φ=1).
#
# ρR, ρX: extraídos do linecode mtx601 do 13bus-all.dss (backbone IEEE-13,
# já referenciado nas notas do projeto): R_mut/R_próprio médio ≈0,456;
# X_mut/X_próprio médio ≈0,423. mtx602 dá frações diferentes (R:~0,21,
# X:~0,36) — a impedância mútua REAL varia por linecode/geometria; como
# branches_1.xlsx não preserva a identidade do linecode por ramo, usa-se
# uma fração ÚNICA representativa (mtx601). É uma aproximação de 1ª ordem,
# não a matriz 3×3 exata (Arnold, 2016) — recomenda-se validar contra essa
# referência antes de citar como metodologia final na tese.
# Aplicado SOMENTE em ramos MT (dado real disponível); BT permanece
# desacoplado (sem dado de acoplamento de secundário disponível).
ENABLE_PHASE_COUPLING = True
RHO_R_MUT = 0.456    # R_mútuo / R_próprio (médio, mtx601)
RHO_X_MUT = 0.423    # X_mútuo / X_próprio (médio, mtx601)
_SQRT3_2  = math.sqrt(3) / 2.0
_NEXT_PH  = {1: 2, 2: 3, 3: 1}    # rotação a→b→c→a
_PREV_PH  = {1: 3, 2: 1, 3: 2}

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
        # CORREÇÃO (revisão): a aba 'PV' não tem coluna 'type' — r.get('type',
        # '3ph') sempre caía no default '3ph', e '3' in '3ph' é sempre True,
        # então is3ph era SEMPRE True para qualquer linha (mono ou trifásica).
        # Isso fazia o ramo "trifásico" ler p_pv_1 como referência para TODOS
        # os monofásicos — certo por coincidência quando a fase é 1 (mascarava
        # o bug), mas dava P_rated=0 quando a fase era 2 ou 3 (P011-B114:
        # 16 inversores de 33 kVA tratados como zero desde o load_data).
        # A checagem correta é simplesmente o número de fases.
        is3ph = (nph == 3)
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
        # Line Drop Compensation (LDC) — estimar tensão no centro de carga.
        # CONVENÇÃO IEEE-13/OpenDSS: os settings R e X do regulador são dados em
        # VOLTS na base do TP (120 V), não em ohms primários. A conversão para
        # p.u. é portanto R_pu = R_volts/120, NÃO R_ohm/Z_base (que inflaria o
        # LDC em ~7x e faz o regulador oscilar — 'hunting').
        # Refs: IEEE 13-node test feeder (Kersting); OpenDSS RegControl R,X em V.
        VBASE_TP = 120.0                       # base do transformador de potencial
        # CORRECAO (revisao): a aba 'Reg' usa as colunas r_LDC_V, X_LDC_V,
        # vreg_V e band_V (em VOLTS na base 120 V). Antes liam-se nomes
        # inexistentes (R_ldc/X_ldc/Vreg_pu/BW_pu), caindo nos defaults — em
        # especial V_ref caia em 1.0 em vez de 122/120 = 1.0167 (setpoint real
        # do regulador). Agora le-se o nome correto, com fallback ao antigo.
        R_ldc_volts = _f(r.get('r_LDC_V', r.get('R_ldc')), 3.0)
        X_ldc_volts = _f(r.get('X_LDC_V', r.get('X_ldc')), 9.0)
        R_ldc_pu = R_ldc_volts / VBASE_TP      # ex.: 3/120  = 0.0250 p.u.
        X_ldc_pu = X_ldc_volts / VBASE_TP      # ex.: 9/120  = 0.0750 p.u.
        # Referencia e banda em VOLTS (vreg_V, band_V) -> p.u. na base 120 V;
        # fallback aos nomes _pu antigos. Default fisico: 122 V e 2 V.
        _vreg = _f(r.get('vreg_V'), None)
        v_ref_lc = (_vreg / VBASE_TP) if _vreg is not None \
            else _f(r.get('Vreg_pu'), 122.0 / VBASE_TP)
        _band = _f(r.get('band_V'), None)
        bw_lc = (_band / VBASE_TP) if _band is not None \
            else _f(r.get('BW_pu'), 2.0 / VBASE_TP)
        svr[ph]  = {
            'mv_bus': mv, 'lv_bus': lv,
            'r_pu':   R / ZMT, 'x_pu': X / ZMT,
            'Imax_pu': I / IMT,
            'tap_min': tap_min, 'tap_max': tap_max,
            'tap_step': step,   'tap_init': tap_init,
            'n_tap':   n_tap,   'delay_periods': d_per,
            # Parâmetros LDC (em p.u., convertidos da base 120 V do TP)
            'R_ldc_pu':  R_ldc_pu,
            'X_ldc_pu':  X_ldc_pu,
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
        # CORREÇÃO (revisão): a aba 'Trafos' TEM dados reais de perdas, nas
        # colunas 'Perdas Vazio kW' e 'Perdas Totais kW ' (note o espaço) —
        # nomes diferentes dos que o código procurava ('P_NL_kW'/'P_LL_kW'),
        # então caía sempre no fallback estimado mesmo havendo dado real.
        # Essas colunas são TOTAIS do transformador (não por fase) — dividir
        # por nph. Perdas de carga = Totais − Vazio (ambas a plena carga).
        _PNL_tot  = _f(r.get('Perdas Vazio kW'), None)
        _PTOT_tot = _f(r.get('Perdas Totais kW'), None)
        if _PNL_tot is not None:
            P_NL_nom_kw = _PNL_tot / nph                          # kW/fase (real)
        else:
            P_NL_nom_kw = _f(r.get('P_NL_kW'), Sn * 0.003)        # fallback (~0,3% Sn)
        if _PNL_tot is not None and _PTOT_tot is not None:
            P_LL_R_kw = max(_PTOT_tot - _PNL_tot, 0.0) / nph      # kW/fase (real)
        else:
            P_LL_R_kw = _f(r.get('P_LL_kW'), Sn * (R/Zb))         # fallback (R·I²)
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
    print(f"  Pesos: W_DV={W_DV} W_UNBAL={W_UNBAL} W_TAP={W_TAP_OPS} "
          f"W_Q_USE={W_Q_USE} W_CURT={W_CURT} W_CORE={W_CORE}")

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

    # Uso absoluto de Qpv: linearização L1 para penalização do despacho
    # reativo. Torna o Qpv NÃO LIVRE — cada kVAr passa a ter custo marginal
    # W_Q_USE na F.O., evitando saturação capacitiva generalizada.
    m.q_abs = pyo.Var(m.PVPH, m.T_VV, domain=pyo.NonNegativeReals,
                      bounds=(0, _ramp_bnd), initialize=0.0)

    # ── RESTRIÇÕES ────────────────────────────────────────────────────────────

    # R1: Tensão fixa no slack
    m.c_slack = pyo.Constraint(
        m.SVRPH, m.T,
        rule=lambda m,ph,t: m.V[slack,ph,t] == 1.0
        if (slack,ph) in connected_bph else pyo.Constraint.Skip)

    # ── R2: Balanço de potência ativa — BFM EXATO ─────────────────────────────
    # P_entrada_líquida = P_ij - r_ij·l_sq_ij  (perdas descontadas no ramo)
    # Referência: Farivar & Low (2013), eq.(2); Baran & Wu (1989)
    # Adjacência PRÉ-COMPUTADA uma única vez. Antes, pred/succ varriam TODOS os
    # ~400 ramos a cada chamada, dentro das regras de balanço avaliadas para
    # cada (barra, fase, período) — O(barras·fases·T·ramos) — o que tornava a
    # CONSTRUÇÃO do modelo lentíssima (recalculava a topologia 144× por nada).
    # Agora é lookup O(1).
    from collections import defaultdict as _dd
    _PRED = _dd(list); _SUCC = _dd(list)
    for (_i, _j, _p) in branches:
        _PRED[(_j, _p)].append((_i, _j, _p))
        _SUCC[(_i, _p)].append((_i, _j, _p))
    _SVR_IN = _dd(list); _SVR_OUT = _dd(list)
    for _p, _d in svr.items():
        _SVR_IN[(_d['lv_bus'], _p)].append(_p)
        _SVR_OUT[(_d['mv_bus'], _p)].append(_p)

    def bal_P_rule(m, b, ph, t):
        if b == slack: return pyo.Constraint.Skip
        ps = _PRED.get((b, ph), []); ss = _SUCC.get((b, ph), [])
        svr_in  = _SVR_IN.get((b, ph), []); svr_out = _SVR_OUT.get((b, ph), [])
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
        ps = _PRED.get((b, ph), []); ss = _SUCC.get((b, ph), [])
        svr_in  = _SVR_IN.get((b, ph), []); svr_out = _SVR_OUT.get((b, ph), [])
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
        expr = (m.V[i,ph,t]
                - r_d*m.P[i,j,ph,t]
                - x_d*m.Q[i,j,ph,t]
                + 0.5*z2*m.l_sq[i,j,ph,t])
        # Acoplamento mútuo entre fases (ver derivação na seção de
        # parâmetros). Só em ramos MT ('mv'); termos LINEARES de
        # coeficiente fixo — não altera a classe/convexidade do problema.
        if ENABLE_PHASE_COUPLING and branches[(i,j,ph)].get('level') == 'mv':
            ph_n, ph_p = _NEXT_PH[ph], _PREV_PH[ph]
            R_mut = RHO_R_MUT * r_d; X_mut = RHO_X_MUT * x_d
            if (i,j,ph_n) in branches:
                Mp = -0.5*R_mut - _SQRT3_2*X_mut
                Mq =  _SQRT3_2*R_mut - 0.5*X_mut
                expr -= Mp*m.P[i,j,ph_n,t] + Mq*m.Q[i,j,ph_n,t]
            if (i,j,ph_p) in branches:
                Mp = -0.5*R_mut + _SQRT3_2*X_mut
                Mq = -_SQRT3_2*R_mut - 0.5*X_mut
                expr -= Mp*m.P[i,j,ph_p,t] + Mq*m.Q[i,j,ph_p,t]
        return m.V[j,ph,t] == expr
    m.c_vdrop = pyo.Constraint(m.CPH, m.T, rule=vdrop_rule)
    if ENABLE_PHASE_COUPLING:
        _n_mv_cph = sum(1 for k in branches if branches[k].get('level')=='mv')
        print(f"  [COUPLING] Acoplamento mútuo ativo em {_n_mv_cph} (ramo,fase) "
              f"MT × {len(hours)} períodos | ρR={RHO_R_MUT} ρX={RHO_X_MUT} "
              f"(mtx601) | R4 permanece LINEAR")

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

    # ── R10b: Linearização |Qpv| para penalização do uso reativo ─────────
    # q_abs[b,ph,t] >= +Qpv  e  q_abs >= -Qpv  →  q_abs = |Qpv| no ótimo
    # (porque W_Q_USE > 0 minimiza q_abs). Torna o despacho reativo custoso.
    m.c_qabs_pos = pyo.Constraint(
        m.PVPH, m.T_VV,
        rule=lambda m,b,ph,t: m.q_abs[b,ph,t] >= m.Qpv[b,ph,t])
    m.c_qabs_neg = pyo.Constraint(
        m.PVPH, m.T_VV,
        rule=lambda m,b,ph,t: m.q_abs[b,ph,t] >= -m.Qpv[b,ph,t])

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
    #   (4) W_Q_USE · Σ|Qpv|       uso reativo (não livre)
    #   (5) W_RAMP  · Σ|ΔQpv|      suavização do despacho reativo
    #   (6) W_CURT  · ΣPcurt       preservar geração PV disponível
    #   (7) W_CORE  · Σ G·V²·SBASE perdas no núcleo dos trafos (QUADRÁTICO)
    # Perdas Joule NÃO entram na F.O.: contabilizadas nas restrições BFM (r·l_sq).
    # As perdas de NÚCLEO (V²) entram opcionalmente via W_CORE (ver nota de escala).
    #
    # Termo de perdas no núcleo: lista (mv_bus, ph, G_core_pu) dos trafos.
    # A perda de ferro é avaliada na tensão PRIMÁRIA (lado MV) do trafo.
    _trafo_core = [(i, ph, branches[(i,j,ph)].get('G_core_pu', 0.0))
                   for (i,j,ph) in branches
                   if branches[(i,j,ph)].get('level') == 'trafo'
                   and (i, ph) in bph_set]
    # Inerte (0.0) quando W_CORE=0 → objetivo permanece LINEAR e os resultados
    # atuais ficam idênticos. Quando W_CORE>0, adiciona termo convexo V².
    _core_term = (W_CORE * sum(G * SBASE * m.V[i,ph,t]**2
                               for (i,ph,G) in _trafo_core for t in hours)
                  if (ENABLE_CORE_LOSS_OBJ and W_CORE > 0.0) else 0.0)
    if ENABLE_CORE_LOSS_OBJ and W_CORE > 0.0:
        print(f"  [CORE] Perdas de núcleo no objetivo: W_CORE={W_CORE} "
              f"× {len(_trafo_core)} (trafo,fase) × {len(hours)} períodos "
              f"(termo QUADRÁTICO V² — objetivo passa a QCQP convexo)")
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
            # (4) Uso de Qpv — despacho reativo NÃO LIVRE (custo marginal)
            # Evita saturação capacitiva generalizada dos inversores.
            + W_Q_USE * sum(m.q_abs[b,ph,t]
                            for (b,ph) in pvph_list for t in t_vv_list)
            # (5) Suavização ΔQpv — vida útil dos inversores
            + sum(_w_ramp[t]*(m.ramp_pos[b,ph,t]+m.ramp_neg[b,ph,t])
                  for (b,ph) in pvph_list for t in t_vv_list)
            # (6) Curtailment PV — preservar geração disponível
            # Perdas r·l_sq NÃO estão aqui — apenas nas restrições BFM.
            + W_CURT * sum(m.Pcurt[b,ph,t]
                           for (b,ph) in pvph_list for t in hours)
            # (7) Perdas no núcleo dos trafos G·V²·SBASE [kW·período]
            # Inerte quando W_CORE=0; quadrático convexo quando ativo.
            # ATENÇÃO: empurra a tensão para baixo, conflita com o suporte.
            + _core_term
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


VV_CURVE_MODE = 'FITTED'  # 'FITTED' (ajuste por percentis, atual) ou
                           # 'IEEE_DEFAULT_CATB' (pontos fixos da norma,
                           # sem otimização/ajuste por dados)

# Pontos-padrão da curva Volt-VAR, Categoria B, IEEE 1547-2018 (Tabela 8):
#   V1 = VRef - 0.08 VN, V2 = VRef - 0.03 VN,
#   V3 = VRef + 0.03 VN, V4 = VRef + 0.08 VN  (VRef = VN = 1,0 pu)
#   Q1 = +100% da capacidade reativa declarada (injeção plena)
#   Q4 = -100% da capacidade reativa declarada (absorção plena)
IEEE1547_CATB_V1 = 0.92
IEEE1547_CATB_V2 = 0.97
IEEE1547_CATB_V3 = 1.03
IEEE1547_CATB_V4 = 1.08


def _fit_vv_curve(V_arr, Q_arr, Q_max_obs):
    """Ajuste com restrições de separação mínima (IEEE 1547 Tabela 8).

    Estratégia em duas etapas:
      1) Extrair valores brutos por percentis DENTRO das zonas onde Q
         realmente está capacitivo/morto/indutivo (não do V bruto).
      2) Aplicar projeção sequencial para garantir as separações mínimas
         entre os pontos de inflexão.
    """
    import numpy as _np_vv
    DV_CAP_MIN  = 0.020  # rampa capacitiva mínima (V2 - V1)
    DV_DEAD_MIN = 0.010  # zona morta mínima (V3 - V2)
    DV_IND_MIN  = 0.020  # rampa indutiva mínima (V4 - V3)

    if Q_max_obs < 1e-6:
        return 0.92, 0.98, 1.02, 1.08

    mask_cap     = Q_arr > 0.20 * Q_max_obs
    mask_ind     = Q_arr < -0.20 * Q_max_obs
    mask_cap_sat = Q_arr > 0.85 * Q_max_obs
    mask_ind_sat = Q_arr < -0.85 * Q_max_obs

    V2_raw = (float(_np_vv.percentile(V_arr[mask_cap], 90))
              if mask_cap.sum() > 2 else 0.98)
    V3_raw = (float(_np_vv.percentile(V_arr[mask_ind], 10))
              if mask_ind.sum() > 2 else 1.02)
    V1_raw = (float(_np_vv.percentile(V_arr[mask_cap_sat], 80))
              if mask_cap_sat.sum() > 2 else V2_raw - 0.03)
    V4_raw = (float(_np_vv.percentile(V_arr[mask_ind_sat], 20))
              if mask_ind_sat.sum() > 2 else V3_raw + 0.03)

    V_center = 0.5 * (V2_raw + V3_raw)
    V2 = min(V2_raw, V_center - 0.5 * DV_DEAD_MIN)
    V3 = max(V3_raw, V_center + 0.5 * DV_DEAD_MIN)
    if V3 - V2 < DV_DEAD_MIN:
        V2 = V_center - 0.5 * DV_DEAD_MIN
        V3 = V_center + 0.5 * DV_DEAD_MIN

    V1 = min(V1_raw, V2 - DV_CAP_MIN)
    V4 = max(V4_raw, V3 + DV_IND_MIN)
    V1 = max(V1, 0.88)
    V4 = min(V4, 1.10)
    if V2 - V1 < DV_CAP_MIN:
        V1 = V2 - DV_CAP_MIN
    if V4 - V3 < DV_IND_MIN:
        V4 = V3 + DV_IND_MIN
    return V1, V2, V3, V4


def _vv_curve_eval(V, V1, V2, V3, V4, Qmax):
    """Avalia a curva Volt-VAR piecewise em V, dados os 4 pontos e Qmax."""
    if V <= V1: return Qmax
    if V <= V2: return Qmax * (V2 - V) / max(V2 - V1, 1e-6)
    if V <= V3: return 0.0
    if V <= V4: return -Qmax * (V - V3) / max(V4 - V3, 1e-6)
    return -Qmax


def classify_inverters_from_opf(model, data, sets_info, vv_curve_mode=None):
    """
    Classifica os inversores (A/B/C) a partir do despacho ótimo do OPF e infere
    as curvas Volt-VAR do Grupo B. Versão standalone da lógica da seção (f),
    para uso ANTES do relatório (permite carregar a Via 2 como cenário base).

    vv_curve_mode: None usa o default do módulo (VV_CURVE_MODE). Valores
    aceitos:
      - 'FITTED': ajusta V1..V4 por inversor a partir dos percentis do
        despacho ótimo (_fit_vv_curve), como antes.
      - 'IEEE_DEFAULT_CATB': usa os pontos fixos da Tabela 8 do IEEE
        1547-2018 (Categoria B) para TODOS os inversores do Grupo B, sem
        nenhum ajuste por dados. O rmse_pct ainda é calculado, comparando
        o despacho ótimo observado contra essa curva fixa — serve para
        medir o quanto a curva padrão "deixa na mesa" em relação ao
        envelope ótimo, não para calibrar a curva.

    Retorna (groups, inferred_B, inv_data) com a mesma semântica da seção (f).
    """
    mode = vv_curve_mode or VV_CURVE_MODE
    if mode not in ('FITTED', 'IEEE_DEFAULT_CATB'):
        raise ValueError(f"vv_curve_mode inválido: {mode!r}")
    import numpy as _np
    pvph_l = sets_info['pvph']
    hours = sets_info['hours']
    t_vv = sets_info['t_vv']
    kappa = math.sqrt(1.0 - FP_MIN_VV**2) / FP_MIN_VV

    groups = {'A': [], 'B': [], 'C': []}
    inv_data = {}
    for (b, ph) in pvph_l:
        pairs = []
        for t in hours:
            if not (t_vv[0] <= t <= t_vv[1]):
                continue
            Pav = pyo.value(model.Pavail[b, ph, t])
            if Pav < 1e-4:
                continue
            Vt = try_v(model, b, ph, t)
            Qt = pyo.value(model.Qpv[b, ph, t]) * SBASE
            Pt = Pav * SBASE
            pairs.append((Vt, Qt, Pt))
        if len(pairs) < 5:
            continue
        V_arr = _np.array([p[0] for p in pairs])
        Q_arr = _np.array([p[1] for p in pairs])
        P_arr = _np.array([p[2] for p in pairs])
        Q_max_inv = float(P_arr.max() * kappa)
        if Q_max_inv < 0.1:
            continue
        Q_lim_t = kappa * P_arr
        frac_cap = float((Q_arr >= 0.9 * Q_lim_t).sum() / len(Q_arr))
        frac_dead = float((_np.abs(Q_arr) < 0.05 * Q_max_inv).sum() / len(Q_arr))
        frac_ind = float((Q_arr <= -0.9 * Q_lim_t).sum() / len(Q_arr))
        if frac_cap >= 0.80 and frac_dead < 0.10:
            cls = 'A'
        elif frac_dead >= 0.60 and abs(Q_arr).max() < 0.10 * Q_max_inv:
            cls = 'C'
        else:
            cls = 'B'
        inv_data[(b, ph)] = {
            'V_arr': V_arr, 'Q_arr': Q_arr, 'P_arr': P_arr,
            'Q_max_inv': Q_max_inv, 'frac_cap': frac_cap,
            'frac_dead': frac_dead, 'frac_ind': frac_ind,
            'classification': cls,
        }
        groups[cls].append((b, ph))

    # Inferir curvas Volt-VAR para o Grupo B — usa a MESMA função canônica
    # (_fit_vv_curve) que a seção (f)/CSV usam, para que a classificação, o
    # export .dss e o CSV sejam sempre consistentes entre si.
    inferred_B = []
    for (b, ph) in groups['B']:
        d = inv_data[(b, ph)]
        V_arr = d['V_arr']; Q_arr = d['Q_arr']
        Q_max_obs = float(_np.percentile(_np.abs(Q_arr), 98))
        if mode == 'IEEE_DEFAULT_CATB':
            V1, V2, V3, V4 = (IEEE1547_CATB_V1, IEEE1547_CATB_V2,
                              IEEE1547_CATB_V3, IEEE1547_CATB_V4)
        else:
            V1, V2, V3, V4 = _fit_vv_curve(V_arr, Q_arr, Q_max_obs)
        Q_pred = _np.array([_vv_curve_eval(v, V1, V2, V3, V4, Q_max_obs)
                            for v in V_arr])
        rmse = float(_np.sqrt(_np.mean((Q_arr - Q_pred)**2)))
        rmse_pct = rmse / max(Q_max_obs, 1e-6) * 100
        inferred_B.append({
            'bus': b, 'ph': ph, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4,
            'Q_max': Q_max_obs / SBASE, 'rmse_pct': rmse_pct,
            'vv_curve_mode': mode,
        })
    return groups, inferred_B, inv_data


_TPL_FIXO_DSS = (
    "New PVSystem.{name} phases={ph} bus1={bus1} kV={kv} kVA={kva} "
    "Pmpp={pmpp} irrad = 0.98 temperature = 25 PF = {pf} "
    "%cutin = 0.1 %cutout = 0.1 effcurve = Myeff P-TCurve = MyPvsT "
    "Daily = MyIrrad TDaily = Temp")
_TPL_VV_DSS = (
    "New PVSystem.{name} phases={ph} bus1={bus1}\n"
    "~ kV={kv} kVA={kva} Pmpp={pmpp} PF=1 irrad=0.98 temperature=25\n"
    "~ effcurve=MyEff P-TCurve=MyPvsT daily=MyIrrad Tdaily=Temp\n"
    "~ %cutin=0.1 %cutout=0.1 kvarMax={kvarmax:.3f} kvarMaxAbs={kvarmaxabs:.3f}")


def _pv_name_dss(bus):
    """Convencao do DSS: barra B011 -> PVSystem PV011."""
    b = str(bus)
    return ("PV" + b[1:]) if b[:1].upper() == "B" else ("PV_" + b)


def _device_group_dss(bus, phases, groups):
    """Classe do dispositivo agregando fases. Prioridade B > A > C."""
    classes = set()
    for cls in ("A", "B", "C"):
        for (b, ph) in groups.get(cls, []):
            if b == bus and ph in phases:
                classes.add(cls)
    for cls in ("B", "A", "C"):
        if cls in classes:
            return cls
    return None


def _kvar_envelope_dss(bus, phases, inv_data):
    """(kvarMax capacitivo, kvarMaxAbs indutivo) do dispositivo, somando o
    envelope de despacho do OPF nas fases (pico de Q>0 e de |Q<0|)."""
    cap = ind = 0.0
    for ph in phases:
        d = inv_data.get((bus, ph))
        if not d:
            continue
        Q = d['Q_arr']
        cap += float(max(Q.max(), 0.0))
        ind += float(max((-Q).max(), 0.0))
    return cap, ind


def export_pvsystems_to_dss(data, groups, inv_data, inferred_B,
                            out_path='pvsystems_opf.dss',
                            pmpp_override=None, kv=0.48, emit_invcontrol=True):
    """Escreve o .dss com TODOS os PVSystems classificados + (opcional) o
    InvControl do grupo B com a curva Volt-VAR agregada inferida.
    Percorre data['pv_meta'] inteiro: nenhum DER fica de fora (nao-classificados
    saem em PF=1). Retorna a lista de nomes do grupo B (DERList do InvControl)."""
    pv_meta = data['pv_meta']
    linhas = ["! PVSystems gerados a partir da solucao do OPF (A/B/C)",
              "! A: FP=0.85 capacitivo | C: FP=1 | B: FP=1 + kvarMax/kvarMaxAbs", ""]
    b_names = []
    cont = {'A': 0, 'B': 0, 'C': 0, '?': 0}
    nc_list = []
    for bus, meta in sorted(pv_meta.items()):
        phases = meta['phases']
        nph = len(phases)
        name = _pv_name_dss(bus)
        kva = meta['kva']
        if pmpp_override is not None:
            pmpp = pmpp_override
        else:
            pmpp = sum(data['pv'].get((bus, ph), {}).get('P_rated_pu', 0.0) * SBASE
                       for ph in phases)
        bus1 = bus if nph == 3 else f"{bus}.{phases[0]}"
        g = _device_group_dss(bus, phases, groups)
        if g == 'B':
            cap, ind = _kvar_envelope_dss(bus, phases, inv_data)
            if cap < 0.05 and ind < 0.05:      # envelope ~0 -> capacidade plena
                cap = ind = sum(inv_data.get((bus, ph), {}).get('Q_max_inv', 0.0)
                                for ph in phases)
            linhas.append(_TPL_VV_DSS.format(name=name, ph=nph, bus1=bus1, kv=kv,
                                             kva=kva, pmpp=round(pmpp, 1),
                                             kvarmax=cap, kvarmaxabs=ind))
            b_names.append(f"PVSystem.{name}")
            cont['B'] += 1
        else:
            pf = 0.85 if g == 'A' else 1.0
            cont[g if g in cont else '?'] += 1
            if g is None:
                nc_list.append(bus)
            linhas.append(_TPL_FIXO_DSS.format(name=name, ph=nph, bus1=bus1,
                                               kv=kv, kva=kva,
                                               pmpp=round(pmpp, 1), pf=pf))
        linhas.append("")
    if emit_invcontrol and b_names and inferred_B:
        import statistics as _st
        V1 = _st.median(d['V1'] for d in inferred_B)
        V2 = _st.median(d['V2'] for d in inferred_B)
        V3 = _st.median(d['V3'] for d in inferred_B)
        V4 = _st.median(d['V4'] for d in inferred_B)
        linhas += [
            "! ---- Volt-VAR do grupo B (curva agregada inferida) ----",
            f"New XYCurve.vvc npts=4 Yarray=[1,0,0,-1] "
            f"Xarray=[{V1:.4f},{V2:.4f},{V3:.4f},{V4:.4f}]",
            "New InvControl.icB mode=VOLTVAR voltage_curvex_ref=rated "
            "vvc_curve1=vvc RefReactivePower=VARMAX",
            "~ DERList=[" + ",".join(b_names) + "]", ""]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas))
    total_dev = len(pv_meta)
    emit = sum(cont.values())
    print(f"  [export OpenDSS] {out_path}")
    print(f"    dispositivos em pv_meta: {total_dev} | emitidos: {emit} "
          f"(A={cont['A']} B={cont['B']} C={cont['C']} n/c={cont['?']})")
    print(f"    InvControl B com {len(b_names)} DERs")
    if emit != total_dev:
        print(f"    [AVISO] {total_dev - emit} dispositivo(s) NAO emitido(s)!")
    if nc_list:
        print(f"    [n/c -> PF=1] sem classificacao no OPF: {nc_list}")
    return b_names

