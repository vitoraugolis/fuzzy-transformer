# Protocolo experimental

## 1. Regras que não se negociam

**Divisão por episódio, nunca por amostra.** Janelas vizinhas compartilham
quase todo o conteúdo; dividir por amostra vaza o alvo entre treino e validação
e produz métricas excelentes e sem valor. `split_episodes` faz a divisão certa.

**Divisão temporal no dado real.** No case study, validação e teste vêm de
períodos *posteriores* aos de treino. Aleatorizar no tempo é a segunda forma
mais comum de vazamento em séries industriais.

**Nenhum documento do futuro.** Ver `docs/05` §1.

**Referência trivial em toda tabela.** Todo resultado de ação vem acompanhado
de `skill_over_zero = 1 − MAE/MAE_zero`. Nas três primeiras execuções deste
projeto, o MAE absoluto parecia razoável (0,087) e era exatamente o do preditor
"não mexer" — sem a referência ao lado, o bug teria passado.

**Semente fixa e configuração versionada.** Cada experimento é um JSON em
`experiments/`; o `report.json` guarda a configuração usada junto dos
resultados. Resultado sem configuração ao lado não entra no relatório.

## 2. Linhas de base

| Baseline | Por que existe |
|---|---|
| Δu = 0 ("não mexer") | referência trivial; um modelo que não a supera não aprendeu nada |
| PI sintonizado | é o que a planta tem hoje |
| PI + supervisório (professor) | teto da imitação no Crawl |
| ANFIS puro (sem atenção) | isola a contribuição da atenção |
| Transformer com MLP (`mixer=mlp`) | isola a contribuição da camada fuzzy |
| Gradient boosting sobre janela achatada | referência tabular forte, sem estrutura sequencial |

A comparação com `mixer=mlp` só é honesta com **contagem de parâmetros
pareada** — ajuste `mlp_ratio` até igualar `n_parameters()`.

## 3. Métricas

**Malha aberta** (`eval/metrics.py`): `delta_mae`, `delta_rmse`, `delta_r2`,
e `skill_over_zero = 1 − MAE/MAE_zero`. É `skill_over_zero` que se reporta;
MAE absoluto sozinho engana quando o alvo tem pouca variância.

**Envelope:** `violation_rate` (fração de amostras em que o modelo autorizaria
ação fora da faixa admissível — métrica de segurança, tem prioridade sobre
desempenho), `mean_width_ratio` (conservadorismo) e `band_mae`.

**Orientações:** F1 macro e micro, `exact_match`, e — mais importante — precisão
por rótulo nos rótulos raros. Média macro esconde que o rótulo que importa
("acionar manutenção") é justamente o mais raro.

**Antecipação:** `detection_lead_time` devolve `lead_over_alarm`, o número de
amostras entre a detecção do modelo e o alarme convencional. É a métrica que
sustenta a motivação do projeto. Agregue com `np.nanmean` — episódios sem
detecção ou sem alarme retornam NaN de propósito, e substituí-los por zero
falsearia a média.

**Malha fechada:** IAE, ISE, erro máximo, esforço de controle (Σ|Δu|), número de
reversões (oscilação), fração do tempo acima do alarme.

## 4. Ablações do Crawl

| # | Ablação | Hipótese |
|---|---|---|
| A1 | `mixer=mlp`, params pareados | ANFIS ≥ MLP em interpretabilidade; desempenho a medir |
| A2 | `use_value_channel=false` | perda pequena se a partição fuzzy for adequada |
| A3 | `use_tag_bias=false` | prior de topologia acelera convergência |
| A4 | `n_mfs` ∈ {2,3,5} | 3 é suficiente para eixos latentes |
| A5 | `n_rules` ∈ {8,32,128} | ganho satura; regras excedentes morrem |
| A6 | sem `trend` | a dimensão de tendência é decisiva para detecção precoce |
| A7 | sem auto-supervisão | pré-treino ajuda no regime de poucos rótulos |
| A8 | `hard_rules=true` | regras nítidas custam pouco desempenho |

Cada ablação roda com 3 sementes. Diferença menor que o desvio entre sementes
não é resultado — e deve ser reportada como "sem diferença detectável", não
omitida.

## 5. Avaliação de interpretabilidade

Quantitativo: fração de regras com antecedente nítido (confiança > 0,5 em ao
menos dois eixos); fração de regras ativas (força média acima do limiar);
concentração da atribuição por tag (uma regra que dispara sobre tudo não explica
nada).

Qualitativo, com procedimento fixo: imprimir as 10 regras mais ativas
(`interpret/top_rules`), apresentá-las a um engenheiro de processo **sem** dizer
qual falha estava ativa, e pedir que ele nomeie o cenário. Registrar acertos,
erros e comentários por escrito. É um protocolo simples e sujeito a viés, mas é
mensurável e repetível — e muito melhor que a alternativa usual, que é o autor
declarar o próprio modelo interpretável.

## 6. Relatório de estágio

Cada estágio termina com um documento contendo: configuração, tabela de
baselines e ablações com desvio entre sementes, gráficos de malha fechada, as
regras extraídas, os portões que fecharam e **os que não fecharam**, e as
questões de pesquisa atualizadas. Portão que não fecha não é fracasso do
estágio: é resultado, desde que registrado.
