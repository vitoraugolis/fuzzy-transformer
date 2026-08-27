# Arquitetura FT-IC

## Fluxo completo

```
janela {tag: [x_{k-n} ... x_k]}
        │
        ▼
┌───────────────────────┐
│ FUZZIFICAÇÃO          │  por tag, por dimensão (level / trend / error)
│ valor → graus         │  101.563 → {mediano 0.50, alto 0.30, ...}
└───────────────────────┘
        │  slots + pesos
        ▼
┌───────────────────────┐
│ EMBEDDING             │  h = E_id[tag] + Σ_p w_p·E_state[slot_p]
│                       │      + E_lag[k-t] + v·W_val
└───────────────────────┘
        │  (B, S, d)          S = n_tags × janela + tokens especiais
        ▼
┌───────────────────────┐  ┐
│ SELF-ATTENTION        │  │
│ softmax(QKᵀ/√d_k)V    │  │
│ + viés relacional P&ID│  │   × N blocos
└───────────────────────┘  │
        │                  │
┌───────────────────────┐  │
│ ANFIS                 │  │  banco de K regras TSK
│ 5 camadas fuzzy       │  │
└───────────────────────┘  ┘
        │
        ▼
┌──────────┬──────────┬──────────┬──────────┐
│ controle │ envelope │orientação│diagnóstico│
│   Δu     │ [lo, hi] │ multi-lbl│  falha    │
└──────────┴──────────┴──────────┴──────────┘
```

Implementação: `src/fuzzytf/model/model.py` (`FTIC`).

## Sequência de tokens

Para `M` tags e janela `W = n+1`, a sequência de processo tem `S = M·W` tokens.
Antes dela entram os tokens especiais, e depois dela (opcionalmente) os tokens
de contexto documental:

```
[CLS] [ACT_1] ... [ACT_a] | T-102@k T-102@k-1 ... | P-204@k ... | ctx_1 ... ctx_c
└── leitura global ─┘      └────── processo (layout tag_major) ──┘ └ documentos ┘
```

* `[CLS]` alimenta as cabeças de orientação e diagnóstico;
* `[ACT_i]` é a consulta de cada variável manipulada; seu estado final vai para
  a cabeça de controle;
* tokens de contexto são embeddings de documentos recuperados (estágio Walk).

Cada token especial tem seu próprio índice na tabela de tags, de modo que o
viés relacional da atenção também os cobre.

## Embedding (`model/embedding.py`)

```
h_t = E_id[tag(t)] + Σ_{p} w_{t,p} · E_state[slot_{t,p}] + E_lag[lag(t)] + [v_t, valid_t]·W_val
```

* `E_state` tem uma linha por slot `(tag, dimensão, termo)` — "alto em T-102" e
  "alto em P-204" são vetores diferentes, como exigido na formulação original.
* A soma ponderada é uma **combinação convexa**: a partição fuzzy é normalizada,
  então o token de estado vive no fecho convexo dos embeddings dos termos. É o
  que dá sentido geométrico à leitura "30% alto, 50% mediano".
* `W_val` reinjeta o valor numérico normalizado. É opcional (`use_value_channel`)
  porque saber quanto se perde sem ele é uma questão de pesquisa (QP-3).

## Atenção (`model/attention.py`)

Atenção multi-cabeça padrão, com dois acréscimos de domínio:

**Viés relacional por par de tags.** Um termo aprendido `B[h, tag_i, tag_j]`
somado aos *scores*, inicializável pela adjacência do P&ID
(`set_topology_prior`). Duas tags da mesma malha começam com afinidade positiva;
o modelo pode reforçar ou desfazer isso durante o treino. É prior estrutural
barato: `n_heads × M²` parâmetros.

**Máscara de validade.** Tokens de instrumento em falha (NaN no histórico) não
recebem atenção, mas continuam presentes — o modelo enxerga *que há um buraco*,
o que é diferente de não haver dado.

A janela é observada por inteiro (`causal=False`): não há geração
autorregressiva sobre o passado, o instante `k` é sempre o mais recente. Ver
QP-5 sobre quando a máscara causal passa a fazer sentido (geração de sequências
de ações futuras).

## Camada ANFIS (`model/anfis.py`)

Detalhada em `docs/04-camada-anfis.md`. Em resumo, por cabeça:

| Camada | Operação | Saída |
|---|---|---|
| 1. fuzzificação latente | `z = LN(W_a h)`, MFs gaussianas | `μ[a,r]` |
| 2. disparo | `w_k = Π_a Σ_r p[k,a,r]·μ[a,r]` | `(B,S,K)` |
| 3. normalização | `w̄ = softmax_k(log w_k / τ)` | `(B,S,K)` |
| 4. consequente | TSK-1 de posto baixo | `(B,S,K,d_h)` |
| 5. agregação | `Σ_k w̄_k · y_k` | `(B,S,d_h)` |

O produto sobre eixos é feito em log-espaço com remoção do máximo por eixo;
sem isso, `A=8` fatores pequenos causam *underflow* em float32.

## Cabeças (`model/heads.py`)

**Controle.** Distribuição sobre termos linguísticos de ação
(`reduzir_muito` … `aumentar_muito`), defuzzificada por centroide com centros
aprendidos, e depois recortada no envelope. A saída é **incremental** (Δu, forma
velocidade), normalizada por `delta_scale`: é como as malhas recebem comando na
prática, evita *bump* na transferência automático/manual e mantém a escala de
saída independente do ponto de operação.

**Envelope.** Prevê `[lo, hi]` como centro `tanh` ± largura `softplus`. O
recorte final é `clamp`, que mantém gradiente nos limites — eles são treináveis
com supervisão de contingência. Limites duros da instrumentação entram por
`hard_limits` e nunca podem ser alargados pelo modelo.

**Orientações.** Multi-rótulo com ponteiro: uma consulta por rótulo atende sobre
os tokens da janela, e o token de maior peso responde "sobre qual tag". É o que
transforma "acionar manutenção" em "acionar manutenção **da V-097**".

**Estado (auto-supervisão).** Projeta o token de volta ao vocabulário de slots,
com pesos amarrados a `E_state` (*weight tying*). Serve à previsão de `k+1` e à
reconstrução de tokens mascarados.

**Diagnóstico.** Classificação da falha ativa a partir de `[CLS]`. Auxiliar, mas
é dela que sai o escore usado na métrica de antecipação ao alarme.

## Dimensionamento

Valores de referência do estágio Crawl e do alvo pretendido:

| Hiperparâmetro | Crawl | Alvo (Run) | Observação |
|---|---|---|---|
| `d_model` | 128 | 512–1024 | QP-2 |
| `n_blocks` | 3 | 8–16 | pares atenção+ANFIS |
| `window` | 24 | 64–256 | limitado por `S = M·W` |
| tags `M` | 4 | 50–500 | atenção fica O(M²W²) — ver QP-6 |
| `n_rules K` | 32 | 256–1024 | por cabeça |
| `n_axes A` | 6 | 8–16 | antecedentes por regra |
| `n_mfs R` | 3 | 3–5 | termos por eixo latente |
| parâmetros | ~0,4 M | 50–500 M | |

O gargalo de escala **não** é a camada ANFIS (custo `O(S·H·K·A·R)`), e sim o
comprimento da sequência: 500 tags × 128 instantes = 64 000 tokens. As saídas
possíveis estão em QP-6 (atenção fatorada tag×tempo, agrupamento por malha,
compressão temporal).
