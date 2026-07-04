# -*- coding: utf-8 -*-
"""
04_diagnose_infeasibility.py
=============================================================================
DIAGNÓSTICO DE INFEASIBILIDADE — refinador de conflitos do CPLEX
=============================================================================
Quando o Estágio 2 prova infeasible (não timeout — prova real, como nas
últimas rodadas: banda=93 períodos, ±1/±2/±3, todas provadas infeasible em
9-11s), a pergunta que falta responder é ONDE. Este módulo não tenta mais
bandas às cegas — usa o refinador de conflitos do próprio CPLEX (IIS —
Irreducible Infeasible Subsystem) pra listar o conjunto MÍNIMO de restrições
que, juntas, não têm solução. Isso costuma apontar direto pra barra/fase/
período que está no limite.

IMPORTANTE — leia antes de rodar:
  A sintaxe exata do modo interativo do CPLEX pode variar um pouco entre
  versões. Este script assume a sequência padrão (read/optimize/conflict/
  display conflict all), documentada há muitas versões do CPLEX, mas se a
  sua instalação (2212) usar uma variação, o retorno bruto do CPLEX (salvo
  em conflict_output_raw.txt) vai mostrar o que aconteceu, mesmo que o
  parsing automático abaixo não capture tudo perfeitamente. Não confie
  cegamente no resumo — o arquivo raw é a fonte da verdade.

Uso típico (depois que 01_opf_extract_policy.py já deixou o modelo num
estado infeasible — chame logo após um _try_stage2 que provou infeasible):

    diag = load_module_from_file('diagnose_infeasibility',
                                 CODE_DIR / '04_diagnose_infeasibility.py')
    diag.diagnose_infeasibility_cplex_conflict(
        model, out_dir='/content/opf_outputs/diag_conflito',
        cplex_exe=os.environ.get('CPLEX_EXE'))
"""
import os
import subprocess
import re
import time


def export_model_lp(model, out_dir, filename='infeasible_model.lp'):
    """Exporta o modelo (no estado atual) para LP com nomes SIMBÓLICOS —
    essencial para o conflito do CPLEX vir com nomes legíveis
    (ex.: 'vdrop_rule[B114,2,123]') em vez de 'c00123481'."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    model.write(path, io_options={'symbolic_solver_labels': True})
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [LP] {path} ({size_mb:.1f} MB)")
    return path


def _build_cplex_script(lp_path, script_path):
    """Sequência padrão do modo interativo do CPLEX para refinamento de
    conflito. Ver aviso no cabeçalho do arquivo sobre variação de sintaxe
    entre versões."""
    script = (
        f"read {lp_path}\n"
        f"optimize\n"
        f"conflict\n"
        f"display conflict all\n"
        f"quit\n"
    )
    with open(script_path, 'w') as f:
        f.write(script)
    return script_path


def run_cplex_conflict_refiner(lp_path, out_dir, cplex_exe='cplex',
                               timelimit=300):
    """Roda o CPLEX em modo interativo via stdin (forma mais portável entre
    versões) com o script de conflito. Salva TUDO que o CPLEX imprimiu em
    conflict_output_raw.txt, independentemente do parsing funcionar."""
    script_path = os.path.join(out_dir, 'conflict_script.txt')
    _build_cplex_script(lp_path, script_path)
    raw_out_path = os.path.join(out_dir, 'conflict_output_raw.txt')

    print(f"  [CPLEX] Rodando refinador de conflito (pode levar até "
         f"{timelimit}s)...")
    t0 = time.time()
    try:
        with open(script_path, 'r') as script_f:
            proc = subprocess.run(
                [cplex_exe], stdin=script_f, capture_output=True,
                text=True, timeout=timelimit)
        raw_output = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    except subprocess.TimeoutExpired as e:
        raw_output = (f"[TIMEOUT após {timelimit}s]\n"
                     f"stdout parcial:\n{e.stdout or ''}\n"
                     f"stderr parcial:\n{e.stderr or ''}")
    except FileNotFoundError:
        raw_output = (f"[ERRO] Executável '{cplex_exe}' não encontrado. "
                     f"Confirme CPLEX_EXE.")
    except Exception as e:
        raw_output = f"[ERRO INESPERADO] {type(e).__name__}: {e}"

    with open(raw_out_path, 'w') as f:
        f.write(raw_output)
    print(f"  [CPLEX] Concluído em {time.time()-t0:.1f}s. "
         f"Saída bruta salva em: {raw_out_path}")
    return raw_output, raw_out_path


def parse_conflict_members(raw_output):
    """Extrai nomes de restrições/variáveis do bloco 'display conflict all'.
    Best-effort — o formato exato varia por versão; o arquivo raw continua
    sendo a fonte confiável se isto não capturar tudo.

    IMPORTANTE (confirmado testando export_model_lp contra o modelo real):
    o escritor LP do Pyomo usa PARÊNTESES, não colchetes, e separa os
    índices por underscore (ex.: 'dv_pos(_611__3_0)' para
    dv_pos[bus=611, ph=3, t=0], não 'dv_pos[611,3,0]'). Barras que começam
    com dígito ganham underscore de escape nas duas pontas. Ajustado para
    esse formato real em vez do bracket-style que eu tinha assumido antes
    de testar."""
    members = []
    pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\(([^)]+)\)')
    for line in raw_output.splitlines():
        for m in pattern.finditer(line):
            cons_name, idx_raw = m.group(1), m.group(2)
            # desfaz o escape: tira underscores duplos/nas pontas, separa
            parts = [p for p in idx_raw.strip('_').split('_') if p]
            members.append((cons_name, tuple(parts)))

    members = sorted(set(members))
    grouped = {}
    for cons_name, parts in members:
        bus = parts[0] if parts else None
        grouped.setdefault(cons_name, []).append((bus, parts))

    return members, grouped


def diagnose_infeasibility_cplex_conflict(model, out_dir, cplex_exe=None,
                                          timelimit=300):
    """Função principal: exporta LP, roda o refinador, resume o resultado.
    Retorna dict com os caminhos dos arquivos e o resumo parseado (quando
    possível)."""
    cplex_exe = cplex_exe or os.environ.get('CPLEX_EXE', 'cplex')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  DIAGNÓSTICO — refinador de conflitos do CPLEX (IIS)")
    print("=" * 78)

    lp_path = export_model_lp(model, out_dir)
    raw_output, raw_path = run_cplex_conflict_refiner(
        lp_path, out_dir, cplex_exe=cplex_exe, timelimit=timelimit)

    members, grouped = parse_conflict_members(raw_output)

    if members:
        print(f"\n  {len(members)} elementos identificados no conflito "
             f"(agrupados por tipo de restrição):")
        for cons_name, entries in sorted(grouped.items(),
                                        key=lambda kv: -len(kv[1])):
            barras = sorted(set(e[0] for e in entries if e[0]))
            print(f"    {cons_name}: {len(entries)} ocorrência(s) — "
                 f"barras envolvidas: {', '.join(barras[:10])}"
                 f"{' ...' if len(barras) > 10 else ''}")
    else:
        print(f"\n  [AVISO] Não consegui extrair membros do conflito "
             f"automaticamente. ISSO NÃO SIGNIFICA QUE FALHOU — abra "
             f"{raw_path} e leia o bloco após 'display conflict all' "
             f"manualmente; o formato pode diferir do esperado por este "
             f"parser.")

    print(f"\n  Arquivos gerados em {out_dir}:")
    print(f"    - {os.path.basename(lp_path)} (modelo exportado)")
    print(f"    - conflict_script.txt (comandos enviados ao CPLEX)")
    print(f"    - conflict_output_raw.txt (saída completa e sem filtro)")
    print("=" * 78)

    return {'lp_path': lp_path, 'raw_output_path': raw_path,
           'members': members, 'grouped': grouped}
