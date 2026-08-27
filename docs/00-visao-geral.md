# Visão geral

## O problema

Uma malha de controle industrial decide com uma fração ínfima da informação que
a planta possui. Um PID vê erro, integral e derivada. Um MPC vê um modelo
linearizado e restrições. Nenhum dos dois vê que a V-097 teve três ordens de
trabalho por travamento de haste nos últimos seis meses, que a OP em curso pede
uma taxa 12% acima do projeto, que o HAZOP do nó 4 prevê exatamente a sequência
que está começando, ou que o relatório do turno anterior registrou "válvula
respondendo lenta".

O operador experiente vê. E é justamente por isso que, quando a coisa aperta,
ele passa a malha para manual.

## A tese

É possível construir um modelo que decida a ação de controle **e** as
providências organizacionais a partir da mesma representação interna, se essa
representação for:

1. **linguística**, e não puramente numérica — porque é em termos linguísticos
   ("válvula saturando", "temperatura subindo rápido") que o conhecimento de
   engenharia está codificado nos documentos e na cabeça das pessoas;
2. **contextual**, com atenção entre tags e instantes — porque o diagnóstico
   nasce da relação entre grandezas, não de nenhuma delas isolada;
3. **estruturada em regras**, e não em um MLP opaco — porque o conhecimento a
   ser armazenado já vem em forma de regra (HAZOP, matriz de causa-e-efeito,
   procedimento operacional) e porque a decisão precisa ser auditável.

Daí a arquitetura proposta: **tokenização fuzzy** dos dados de processo,
**self-attention** para o contexto, e **camadas ANFIS** no lugar do MLP para o
conhecimento. O nome de trabalho é **FT-IC** (*Fuzzy Transformer for Industrial
Control*).

## O caso motivador

> Uma válvula satura em um ponto incorreto. A saturação é detectável muito
> antes que a temperatura chegue ao alarme. O sistema, em vez de continuar
> pedindo abertura que a válvula não entrega, **filtra a ação de controle para
> a faixa de contingência** e **aciona a manutenção** para intervir no problema
> mecânico.

Esse caso é o critério de aceitação do projeto inteiro. Ele exige, simultaneamente:

| Exigência | Onde é atendida |
|---|---|
| Perceber saturação anômala antes do alarme | tokens de estado + atenção entre V-097 e T-102 |
| Saber que a faixa útil mudou | cabeça de envelope (`band_lo`, `band_hi`) |
| Continuar controlando dentro do que resta | recorte da ação defuzzificada |
| Saber *quem* chamar e *sobre o quê* | cabeça de orientações + ponteiro de atribuição |
| Justificar a decisão | regras disparadas + mapas de atenção |

## Contribuições pretendidas

1. Um esquema de tokenização que trata dado de processo como sequência de
   símbolos com semântica de engenharia (ID + estado fuzzy), e não como janela
   numérica achatada.
2. Uma camada neuro-fuzzy escalável em profundidade — banco de regras aprendido
   com seleção esparsa, em vez da grade combinatória do ANFIS clássico.
3. Um protocolo de treinamento que incorpora documentação (WO, PO, HAZOP,
   manuais, relatórios de turno) em três níveis: recuperação, aterramento de
   regras e supervisão distante.
4. Uma avaliação que mede o que importa em planta: desempenho em malha fechada,
   antecipação em relação ao alarme e taxa de violação do envelope — não apenas
   erro de regressão.

## Como este repositório está organizado

| Caminho | Conteúdo |
|---|---|
| `docs/01-arquitetura.md` | especificação do modelo |
| `docs/02-roadmap-crawl-walk-run.md` | **o plano de investigação** |
| `docs/03..05` | tokenização, camada ANFIS, conhecimento documental |
| `docs/06-protocolo-experimental.md` | baselines, ablações, métricas |
| `docs/07-perguntas-de-pesquisa.md` | o que ainda está em aberto (QP-1..QP-12) |
| `docs/08-case-study.md` | contrato de dados do estudo de caso |
| `docs/09-seguranca-e-governanca.md` | o que impede o modelo de fazer besteira |
| `src/fuzzytf/` | implementação |
| `experiments/` | configurações versionadas de cada experimento |
| `scripts/` | pontos de entrada executáveis |

## Glossário

**Tag** — identificador de um ponto de medição ou atuação (T-102, V-097).
**Token de ID** — vetor que representa a identidade da tag.
**Token de estado** — vetor que representa *em que estado linguístico* a tag está.
**Slot** — par (tag, dimensão, termo); a unidade do vocabulário de estado.
**Envelope / banda de contingência** — faixa de ação considerada admissível
dadas as condições atuais dos equipamentos.
**Orientação (*advisory*)** — providência recomendada a uma equipe.
**Professor** — controlador de referência cuja ação o modelo imita no Crawl.
