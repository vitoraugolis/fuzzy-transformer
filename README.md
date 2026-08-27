# FT-IC — Fuzzy Transformer for Industrial Control

Arquitetura de transformer para **controle inteligente de malhas industriais**:
tokenização fuzzy dos dados de processo, self-attention para o contexto e
camadas **ANFIS** no lugar do MLP para o conhecimento de engenharia. A saída não
é só um número — é a ação de controle **filtrada por um envelope de
contingência** mais as **orientações às equipes**, com as regras que as
sustentam.

> Uma válvula satura em um ponto incorreto, muito antes do alarme de
> temperatura. O sistema filtra a ação para a faixa admissível, mantém a malha
> dentro do que resta de curso e aciona a manutenção — dizendo qual válvula e
> por quê.

Projeto de doutorado, organizado em **Crawl → Walk → Run**:
[`docs/02-roadmap-crawl-walk-run.md`](docs/02-roadmap-crawl-walk-run.md).

## Começando

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # 43 testes
python scripts/run_crawl.py --smoke         # pipeline completo no simulador (~2 min)

python scripts/setup_case_study.py          # extrai case_study.zip
python scripts/run_u200.py --no-train       # ingere o acervo real e reporta
```

## A arquitetura em uma tela

```
{T-102: [101.563, 102.642, ...], V-097: [0.765, ...]}
        │
   fuzzificação      101.563 → 50% mediano, 30% alto, ...
        │
   embedding         h = E_id[tag] + Σ w_p·E_state[slot_p] + E_lag + valor
        │
   ┌────────────────────────────────┐
   │ self-attention  (contexto)     │  × N
   │ ANFIS           (conhecimento) │
   └────────────────────────────────┘
        │
   ┌──────────┬──────────┬───────────┬────────────┐
   │ Δu       │ envelope │ orientações│ diagnóstico│
   └──────────┴──────────┴───────────┴────────────┘
```

Cada token é **uma tag em um instante**, e carrega quem ela é (embedding de ID)
e como está (combinação convexa dos embeddings dos termos linguísticos). A
camada ANFIS é um sistema Takagi-Sugeno com **banco de regras aprendido e
esparso** — `K` regras que escolhem quais termos usar em cada eixo, em vez da
grade combinatória `R^A` do ANFIS clássico.

## Estudo de caso

O acervo U-200/R-201 (`case_study.zip`) é um benchmark multissilo: o PID mantém
100% do produto em especificação enquanto a válvula FV-201 vai de 32% a 94,7% de
curso — a unidade queima toda a reserva de rejeição de perturbação **sem que
nenhuma métrica de erro registre nada**. Depois, 3 K de variação na carga custam
21 h fora de controle e 138× o custo operacional.

A causa raiz não está em nenhum documento isolado: exige cruzar manutenção,
engenharia/fabricante e operação. `fuzzytf.integration.FTICController` implementa
o contrato `BaseController` do próprio caso, e
[`docs/08-case-study.md`](docs/08-case-study.md) descreve a ingestão.

## Estrutura

```
docs/          visão geral, arquitetura, roadmap, protocolo, questões em aberto
src/fuzzytf/
  fuzzy/       partições, variáveis linguísticas, fuzzificador
  model/       embedding, atenção, camada ANFIS, blocos, cabeças
  data/        simulador, dataset, documentos, adaptador U-200
  train/       perdas multitarefa e laço de treino
  eval/        métricas e rollout em malha fechada
  interpret/   leitura das regras aprendidas
  integration/ plugue no harness do estudo de caso
experiments/   configurações versionadas
scripts/       run_crawl.py · run_u200.py · setup_case_study.py · inspect_case_study.py
tests/         43 testes
```

## Estado

| Estágio | Situação |
|---|---|
| Crawl C1–C3 (fuzzy, bloco, treino) | implementado e testado |
| Crawl C4–C5 (malha fechada, regras) | infraestrutura pronta; ablações pendentes |
| Walk W1 (ingestão do U-200) | implementado — 11 tags, 115 slots, 33 documentos em 8 silos |
| Walk W2–W5 | em aberto |
| Run | em aberto |

Achados já registrados (e o que custaram) estão em
[`docs/07-perguntas-de-pesquisa.md`](docs/07-perguntas-de-pesquisa.md).

## Documentação

| | |
|---|---|
| [00 — Visão geral](docs/00-visao-geral.md) | problema, tese, contribuições |
| [01 — Arquitetura](docs/01-arquitetura.md) | especificação do modelo |
| [02 — Roadmap](docs/02-roadmap-crawl-walk-run.md) | Crawl, Walk, Run e os portões |
| [03 — Tokenização](docs/03-tokenizacao-e-fuzzificacao.md) | fuzzificação e vocabulário |
| [04 — Camada ANFIS](docs/04-camada-anfis.md) | por que não um MLP |
| [05 — Conhecimento](docs/05-conhecimento-e-dados.md) | WOs, POs, HAZOP no treino |
| [06 — Protocolo](docs/06-protocolo-experimental.md) | baselines, ablações, métricas |
| [07 — Questões](docs/07-perguntas-de-pesquisa.md) | QP-1..QP-12 e achados |
| [08 — Estudo de caso](docs/08-case-study.md) | U-200 e a integração |
| [09 — Segurança](docs/09-seguranca-e-governanca.md) | camadas de proteção e governança |

## Licença

MIT — ver [LICENSE](LICENSE).
