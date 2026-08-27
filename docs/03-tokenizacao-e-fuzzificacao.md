# Tokenização e fuzzificação

## O que é um token aqui

Em um modelo de linguagem, um token é um pedaço de texto. Aqui, um token é
**uma tag em um instante**, e ele carrega duas informações independentes:

* **quem é** — a identidade da tag (token de ID);
* **como está** — o estado linguístico (token de estado).

As duas são somadas no mesmo vetor. A separação importa: permite que o modelo
generalize "válvula saturada" entre válvulas diferentes (mesmo termo, tags
diferentes) e, ao mesmo tempo, saiba que a V-097 tem histórico próprio.

## Dimensões linguísticas

Uma tag não tem só "nível". Um valor de 101,5 °C significa coisas diferentes se
está subindo rápido ou estável. Por isso cada tag pode ter até três dimensões:

| Dimensão | Sinal fuzzificado | Termos padrão |
|---|---|---|
| `level` | `x_t` | muito_baixo, baixo, mediano, alto, muito_alto |
| `trend` | `x_t − x_{t−1}` | caindo_rapido, caindo, estavel, subindo, subindo_rapido |
| `error` | `x_t − SP_t` | idem `level`, centrado em zero |

O token de estado é a soma das contribuições de todas as dimensões. Isso evita a
explosão combinatória de um vocabulário conjunto: `5 × 5 = 25` termos compostos
viram `5 + 5 = 10` slots somados.

**Escolha aberta (QP-1):** somar as dimensões pressupõe que elas se combinam
aditivamente no espaço de embedding. A alternativa é concatenar sub-espaços
dedicados por dimensão. A soma é mais barata e mais flexível; a concatenação
garante que uma dimensão não "apague" a outra. Ainda não medimos a diferença.

## Da partição uniforme à partição de engenharia

`ruspini_partition` gera uma partição triangular uniforme — é ponto de partida,
não resposta. Em planta, os termos precisam estar ancorados em números que o
operador reconhece:

```python
partition_from_breakpoints(
    breakpoints=[70, 90, 100, 108, 115],   # ← LL, L, normal, H (alarme), HH
    terms=("muito_baixo", "baixo", "mediano", "alto", "muito_alto"),
)
```

Com isso, "T-102 está alta" passa a significar literalmente "está na região do
alarme H", e a regra aprendida pelo modelo torna-se comparável à regra escrita
no HAZOP. Quando não há limites documentados, `load_variable_book(...,
quantile_partition=True)` ancora os termos nos quantis observados — quase sempre
melhor que a faixa do instrumento, que costuma ser muito mais larga que a faixa
realmente operada.

**Três estratégias, a serem comparadas (QP-2):**

1. **fixa por engenharia** — máxima interpretabilidade, exige trabalho manual;
2. **por quantis** — automática, adaptada aos dados, semanticamente mais frouxa;
3. **aprendida** — MFs gaussianas com centro e largura treináveis; melhor ajuste,
   mas os termos deixam de significar o que o nome diz, e a auditabilidade cai.

A hipótese de trabalho é um híbrido: centros fixos por engenharia, larguras
aprendidas. Ainda não testado.

## Esparsidade (top-p)

Com partição de Ruspini triangular, um valor pertence a no máximo **dois**
termos. Guardar o vetor denso de pertinências é desperdício. O fuzzificador
mantém apenas os `top_p` maiores graus por dimensão, renormalizados:

```
101.563 → slots [(T-102,level,mediano), (T-102,level,alto)]
          pesos [0.50, 0.30] → normalizado [0.625, 0.375]
```

Isso torna o custo do embedding `O(S · P · d)` com `P ≈ 3`, independentemente do
tamanho do vocabulário de estado — que pode chegar a milhares de slots numa
planta inteira.

## Dados faltantes

`valid=False` marca o token cujo dado original era NaN. O valor é preenchido por
*hold* para não propagar NaN, mas a flag entra no canal numérico e a máscara de
atenção exclui o token como chave. O modelo enxerga o buraco em vez de um valor
inventado — necessário para que ele aprenda a diagnosticar instrumento em falha
em vez de acreditar nele.

## Layout da sequência

`tag_major` agrupa cada tag com sua própria história; `time_major` agrupa por
instante. Matematicamente é indiferente (a atenção é equivariante a permutação e
a posição entra por embedding), mas muda a leitura dos mapas de atenção e o
padrão de acesso à memória. Passa a importar se for adotada atenção fatorada
(QP-6), em que os eixos deixam de ser intercambiáveis.

## Riscos conhecidos

**Compressão do historiador.** Historiadores industriais comprimem por *swinging
door*: o dado gravado não é o dado amostrado. A dimensão `trend`, calculada por
diferença, é a primeira vítima. Antes de confiar nela no case study, é preciso
medir o efeito da reconstrução (W1).

**Amostragem irregular.** O modelo assume passo constante via `E_lag`. Séries
com passo variável exigem reamostragem — e reamostrar antes de fuzzificar ou
depois dá resultados diferentes. Reamostrar primeiro é o padrão adotado, e a
alternativa fica registrada como questão em aberto.

**Perda de resolução.** Fuzzificar descarta informação: 101,4 e 101,6 podem cair
nos mesmos dois termos com pesos quase idênticos. O canal numérico residual
existe para compensar; medir quanto ele compensa é QP-3.
