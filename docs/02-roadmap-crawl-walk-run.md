# Roadmap — Crawl, Walk, Run

O princípio de organização é simples: **cada estágio existe para eliminar uma
classe de risco, e termina com um portão mensurável.** Não se passa de estágio
por cronograma, e sim por evidência. Se um portão não fecha, ou o experimento
volta, ou a hipótese muda — e a mudança fica registrada em
`docs/07-perguntas-de-pesquisa.md`.

| Estágio | Pergunta que responde | Risco que elimina |
|---|---|---|
| **Crawl** | A arquitetura aprende e fecha malha? | risco de arquitetura |
| **Walk** | O conhecimento documental agrega? | risco de representação de conhecimento |
| **Run** | Isso sobrevive à planta real? | risco operacional e de escala |

---

## CRAWL — a arquitetura funciona?

**Ambiente:** simulador da malha TIC-102/V-097 (`src/fuzzytf/data/simulator.py`),
com verdade-fundamental de falha, envelope e orientações. Nada de dado real
ainda: aqui se depura arquitetura, e para isso é preciso saber a resposta certa.

### C1 — Núcleo fuzzy e tokenização ✅ implementado
Partições de Ruspini por tag, dimensões `level`/`trend`/`error`, vocabulário de
slots serializável com *fingerprint*, fuzzificação esparsa top-p.
**Portão:** `fuzzifier(101.563)` produz a leitura linguística esperada e
`VariableBook` sobrevive a *round-trip* JSON. → `tests/test_fuzzy.py`

### C2 — Bloco atenção + ANFIS ✅ implementado
Atenção com viés relacional; camada ANFIS com banco de regras esparso,
estabilizada em log-espaço; ablação com MLP na mesma assinatura.
**Portão:** gradiente flui para todos os parâmetros; forças de disparo somam 1;
o modelo com `mixer=mlp` treina no mesmo laço. → `tests/test_model.py`

### C3 — Imitação do professor ✅ implementado
Treino multitarefa (ação, envelope, orientações, diagnóstico, previsão fuzzy,
estado mascarado) sobre episódios simulados.
**Portão:** `delta_mae` em validação abaixo do erro do preditor trivial
(Δu = 0) por margem clara, com o *split* feito **por episódio**.
→ `scripts/run_crawl.py`

### C4 — Malha fechada ✗ não fechou (ver `docs/10-relatorio-crawl-01.md`)
O modelo substitui o PI no simulador (`eval/rollout.py`).
**Portão:** IAE do FT-IC ≤ 1,2 × IAE do PI nos episódios sem falha, **e**
IAE estritamente menor nos episódios com `valve_travel_limit`, **e** nenhuma
oscilação sustentada (número de reversões de sinal comparável ao do PI).
**Resultado:** IAE 2,54 contra 0,237 do PI (10,7×). Causa isolada: viés médio de
−0,040% de curso por amostra que, com ação incremental sobre planta integradora,
**acumula** — 24% de curso de deriva em 600 amostras. Correlação com a ação do
professor é 0,760, então não é ruído; é viés.

### C4a — Corrigir o viés acumulado *(novo, decorrente de C4)*
Três frentes em ordem de custo: âncora absoluta de `u`; perda sobre o comando
acumulado em horizonte H; ajuste em malha fechada tipo DAgger para tratar o
desvio de distribuição.
**Portão:** o mesmo de C4.

### C5 — Antecipação e leitura de regras ✗ não fechou
**Portão duplo:**
1. `lead_over_alarm` mediana > 0 nos episódios de saturação de válvula — o
   modelo sinaliza antes do alarme de temperatura;
2. as 10 regras mais ativas são legíveis (`interpret/top_rules`) e pelo menos
   metade delas é atribuível a uma tag dominante coerente com a falha.

**Resultado:** a extração funciona e os antecedentes são nítidos (confiança 0,52
a 0,72), mas a atribuição por tag é difusa — 26%/19%/19% entre três tags, quase
a uniforme entre cinco, e nenhuma das dez regras mais ativas tem tag dominante.
É a QP-7 se manifestando: nada no treino obriga os eixos latentes a se alinharem
com grandezas físicas. A metade da antecipação não foi medida, porque a métrica
só faz sentido sobre rollouts válidos — e C4 não fechou.

O portão 2 é qualitativo por natureza. O procedimento de avaliação está em
`docs/06-protocolo-experimental.md` §5; o julgamento é registrado por escrito,
com as regras impressas, para poder ser contestado depois.

### Ablações obrigatórias do Crawl
| Ablação | Pergunta |
|---|---|
| `mixer=mlp` com nº de parâmetros pareado | o ANFIS agrega desempenho ou só interpretabilidade? |
| `use_value_channel=false` | quanto se perde só com a descrição fuzzy? |
| `use_tag_bias=false` | o prior de topologia importa? |
| `n_mfs` ∈ {2,3,5}, `n_rules` ∈ {8,32,128} | sensibilidade à granularidade fuzzy |
| sem `w_masked_state`/`w_forecast` | a auto-supervisão ajuda com pouco rótulo? |

**Saída do Crawl:** um relatório com as cinco ablações, o gráfico de malha
fechada e a lista de regras aprendidas. Se `mixer=mlp` vencer em tudo, isso
precisa ser dito — e a justificativa do ANFIS passa a ser exclusivamente de
auditabilidade, o que muda a tese.

---

## WALK — o conhecimento documental agrega?

**Ambiente:** case study real (`docs/08-case-study.md`), com o simulador ainda
disponível para testes controlados.

### W1 — Ingestão do case study
Adaptador já escrito (`data/case_study.py`); validador em
`scripts/inspect_case_study.py`. Falta: alinhamento temporal entre histórico de
processo e eventos, tratamento de amostragem irregular e de *dead bands* do
historiador (compressão por *swinging door* deforma a dimensão `trend` — é
preciso medir esse efeito antes de confiar nela).
**Portão:** `inspect --load` roda limpo; vocabulário fuzzy construído por
quantis com partição sensata para cada tag, revisado por um especialista.

### W2 — Rótulos por supervisão distante
WOs e relatórios de turno rotulam retroativamente as janelas: uma WO mecânica
na V-097 em `t` torna positivas as janelas em `[t-Δ, t]` para
"acionar manutenção da válvula".
**Portão:** taxa de concordância aceitável entre rótulo automático e uma amostra
de 200 janelas rotuladas à mão. O valor-alvo se define depois de ver a primeira
amostra — fixar um número antes de conhecer o desbalanceamento seria arbitrário.
**Risco principal:** rótulo por WO é *tardio* (a WO é aberta depois que o
problema ficou óbvio). Mitigação: rotular a janela **anterior** à abertura e
tratar Δ como hiperparâmetro medido, não suposto.

### W3 — Recuperação documental no contexto
Recuperador BM25-lite (`data/documents.py`) seleciona documentos pelo estado
fuzzy corrente e pelas tags ativas; os trechos entram como tokens de contexto.
**Portão:** ganho estatisticamente significativo em F1 das orientações contra o
mesmo modelo sem contexto, com o mesmo orçamento de parâmetros.

### W4 — Aterramento de regras por HAZOP
Linhas de HAZOP viram regras candidatas (`hazop_to_rules`), usadas para
inicializar e regularizar o banco de regras.
**Portão:** ≥ 30% das regras aterradas permanecem ativas após o treino (força
de disparo média acima do percentil 50 do banco), sem perda de desempenho na
ação. Regras que morrem são material de análise: ou o HAZOP não se manifesta nos
dados, ou o mecanismo de aterramento é fraco.

### W5 — Pré-treino auto-supervisionado em escala
Estado mascarado + previsão fuzzy sobre todo o histórico disponível, sem
rótulo; ajuste fino só nas janelas com supervisão.
**Portão:** com 10% dos rótulos, o modelo pré-treinado alcança ≥ 90% do
desempenho do modelo treinado do zero com 100% dos rótulos.

**Saída do Walk:** o modelo reproduz, no case study, o caso motivador completo —
detecta a saturação anômala, estreita o envelope, mantém a malha e emite a
orientação correta com a tag correta.

---

## RUN — isso sobrevive à planta real?

### R1 — Escala de sequência
Dezenas a centenas de tags × janelas longas. Atenção fatorada (tag × tempo),
agrupamento por malha, compressão temporal de instantes antigos. Ver QP-6.
**Portão:** latência de inferência abaixo do período de amostragem da malha
mais rápida do escopo, com margem de 10×.

### R2 — Além da imitação
O professor limita o aluno: imitar um PI produz, no melhor caso, um PI caro.
Duas frentes, nesta ordem: (a) RL *offline* com restrição de política
(o histórico é a única fonte segura de exploração); (b) MPC como professor onde
houver modelo. O envelope aprendido no Crawl/Walk vira restrição dura da
otimização.
**Portão:** ganho em malha fechada sobre o professor em cenários de falha, sem
piora nos cenários normais.

### R3 — Operação assistida (*shadow mode*)
O modelo roda em paralelo à malha real sem atuar: registra ação sugerida,
envelope e orientações; a comparação com o que o operador de fato fez é o
conjunto de avaliação mais honesto que existe.
**Portão:** três meses de registro; concordância medida; zero incidente
atribuível ao sistema (que não atua) e orientações consideradas úteis pela
equipe em revisão estruturada.

### R4 — Humano no laço e governança
Orientações vão para o fluxo de trabalho existente (sistema de WO, passagem de
turno), sempre com a explicação anexa: regras disparadas, tags atribuídas,
janela de evidência. Ver `docs/09-seguranca-e-governanca.md`.

### R5 — Produção do doutorado
Capítulos e artigos: (1) tokenização fuzzy de processo; (2) camada ANFIS
esparsa e escalável; (3) aterramento documental de regras; (4) estudo de caso
completo com avaliação em *shadow mode*.

---

## Cronograma indicativo

| Trimestre | Foco | Marco |
|---|---|---|
| T1 | C1–C3 | pipeline ponta a ponta treinando |
| T2 | C4–C5 + ablações | relatório do Crawl |
| T3 | W1–W2 | case study ingerido e rotulado |
| T4 | W3–W4 | contexto e aterramento avaliados |
| T5 | W5 + R1 | pré-treino e escala |
| T6–T7 | R2 | além da imitação |
| T8+ | R3–R5 | shadow mode, tese, artigos |

Os trimestres são de esforço, não de calendário: W1 costuma consumir mais tempo
do que o previsto, porque dado industrial real chega sujo, com amostragem
irregular e sem alinhamento entre historiador e sistema de manutenção.

## Riscos e planos B

| Risco | Sinal de alerta | Plano B |
|---|---|---|
| ANFIS não supera MLP | ablação C4 empatada | manter ANFIS pela auditabilidade; declarar isso explicitamente e mover a contribuição para tokenização + aterramento |
| Regras aprendidas ilegíveis | entropia alta apesar do regularizador | `hard_rules=True` com Gumbel; menos eixos por regra; extração post-hoc por poda |
| Rótulos de WO fracos demais | concordância baixa em W2 | reduzir escopo para diagnóstico (falha) + envelope, deixando orientação como *ranking* e não classificação |
| Sequência longa demais | latência acima do período de amostragem | atenção fatorada; janela hierárquica; seleção de tags por malha |
| Case study pequeno | poucos episódios de falha | transferência: pré-treino no simulador calibrado com os parâmetros do case study, ajuste fino no real |
| Modelo bom em métrica, inútil em planta | operadores ignoram as orientações | R3 antes de qualquer atuação; revisão estruturada com a equipe como critério de aceitação |

## Estado atual

Fechados: C1, C2, C3 (`skill_over_zero` = +0,350 contra o preditor trivial) e
W1 (ingestão completa do acervo U-200). Não fecharam: C4 e C5 — o relatório está
em `docs/10-relatorio-crawl-01.md`, com o diagnóstico e o encaminhamento.

O próximo passo é **C4a**, não as ablações: não faz sentido medir a contribuição
do ANFIS contra um MLP enquanto a política ainda não fecha malha.
