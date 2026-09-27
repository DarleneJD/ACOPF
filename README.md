# ACOPF — gerenciamento de tensão em redes MT/BT

Este repositório reúne dois fluxos encadeados:

1. seleção de dias meteorológicos representativos de 2019;
2. execução do modelo OPF MV/LV autônomo V4.9.

Os notebooks são os pontos de entrada principais no Google Colab. O primeiro produz o JSON consumido pelo segundo.

## Estrutura

```text
ACOPF/
├── data/
│   ├── meteorology/
│   │   ├── profiles_10min/
│   │   │   ├── perfis_10min_2019_01.csv
│   │   │   ├── ...
│   │   │   └── perfis_10min_2019_12.csv
│   │   ├── dicionario_8_dias_representativos.json
│   │   └── manifesto_perfis_10min.json
│   └── network/
│       ├── branches_1.xlsx
│       └── buses_1.xlsx
├── notebooks/
│   ├── DiasRepresentativos.ipynb
│   └── OPF_MVLV_autonomo_v4_9.ipynb
├── src/
│   └── acopf.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Ordem de execução

### 1. Gerar os dias representativos

[Abrir `DiasRepresentativos.ipynb` no Google Colab](https://colab.research.google.com/github/DarleneJD/ACOPF/blob/main/notebooks/DiasRepresentativos.ipynb)

O notebook:

- lê os doze perfis mensais de 2019;
- valida o calendário e a grade de 144 períodos de 10 minutos por dia;
- agrupa separadamente os dias de verão, outono, inverno e primavera;
- utiliza dois clusters por estação;
- seleciona o dia observado mais próximo de cada centroide;
- produz oito dias representativos e suas frequências;
- exporta o JSON usado pelo modelo OPF.

Somente a parte posterior a `#Corrigido` do notebook original foi preservada conceitualmente. O conteúdo foi reorganizado em etapas explicadas por células Markdown, com configuração, validação, agrupamento, exportação e interpretação das saídas.

Saída principal:

```text
dicionario_8_dias_representativos.json
```

Após conferir os resultados, esse arquivo pode substituir:

```text
data/meteorology/dicionario_8_dias_representativos.json
```

### 2. Executar o modelo OPF

[Abrir `OPF_MVLV_autonomo_v4_9.ipynb` no Google Colab](https://colab.research.google.com/github/DarleneJD/ACOPF/blob/main/notebooks/OPF_MVLV_autonomo_v4_9.ipynb)

O notebook V4.9 incorpora seu próprio motor e não importa `src/acopf.py`. Ele utiliza:

- `data/network/buses_1.xlsx`;
- `data/network/branches_1.xlsx`;
- `data/meteorology/dicionario_8_dias_representativos.json`;
- `data/meteorology/profiles_10min/perfis_10min_2019_02.csv` na campanha multidia de fevereiro.

O bloco `CFG`, no início do notebook, controla o dia-fonte, o diretório de saída, os limites de tempo do solver e a ativação das campanhas.

## Correção do calendário

Os arquivos do antigo branch meteorológico usavam 2001 como calendário auxiliar, embora os dados correspondessem a 2019. Nesta organização:

- os arquivos foram renomeados para `perfis_10min_2019_MM.csv`;
- todos os timestamps usam 2019;
- a coluna `source_year_index` registra 2019;
- o manifesto declara `calendar_year: 2019`;
- ambos os notebooks trabalham diretamente com 2019, sem conversão de ano-índice.

O ano de 2019 não é bissexto, assim como 2001; portanto, a alteração preserva a quantidade de dias e a correspondência entre mês, dia, hora e amostra.

## CPLEX no Colab

O notebook OPF espera, por padrão, o instalador Linux do CPLEX em:

```text
/content/drive/MyDrive/Solvers/cplex_studio2212.linux_x86_64.bin
```

O instalador não é distribuído neste repositório. É necessário possuir licença válida e disponibilizar o arquivo no próprio Google Drive. A execução da célula de instalação pressupõe a aceitação dos termos aplicáveis da IBM.

## Dependências

Os notebooks instalam automaticamente os pacotes Python ausentes. Para execução local:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução Python independente

`src/acopf.py` é uma alternativa independente baseada no antigo `1.py`. Ela usa as planilhas de `data/network/` e grava os resultados em `outputs/`.

```bash
python src/acopf.py --cplex-exe /caminho/para/cplex
```

Esse arquivo não é carregado pelos notebooks e foi mantido apenas como segunda forma implementada de execução. Nenhum dos demais módulos Python históricos é necessário.

## Dados e resultados

Resultados gerados não são versionados. Os notebooks gravam suas campanhas no Google Drive, enquanto a execução local utiliza `outputs/`.

Os materiais recebidos não incluem arquivo de licença nem instrução formal de citação. Antes da redistribuição, inclua a licença aplicável ao código e aos dados, além da referência bibliográfica do trabalho associado.
