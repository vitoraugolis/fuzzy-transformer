# A camada ANFIS

## Por que não um MLP

O MLP de um transformer é onde o conhecimento fica armazenado. A pergunta deste
projeto é: **que forma esse conhecimento tem, no domínio industrial?**

A resposta é conhecida e está escrita nos documentos da planta: forma de regra.
HAZOP é uma tabela de regras. Matriz de causa-e-efeito é uma tabela de regras.
Procedimento operacional é uma sequência de regras. Um MLP consegue aprender
qualquer uma delas, mas não permite *escrevê-las*, *lê-las* nem *auditá-las* —
e sem isso não há como aterrar documentação no modelo, nem como justificar uma
ação de controle para uma equipe de operação.

A camada ANFIS mantém a capacidade de aproximação (é um aproximador universal,
como o MLP) e acrescenta uma estrutura que se lê como regra.

## O problema do ANFIS clássico

O ANFIS de Jang é uma grade completa: `A` entradas com `R` MFs cada geram `R^A`
regras. Com `A=8` e `R=3` são 6 561 regras por camada — e `A` precisaria ser da
ordem de `d_model` para operar sobre embeddings. Inviável.

## A solução adotada: banco de regras esparso e aprendido

Em vez da grade completa, um banco de `K` regras, e cada regra **aprende quais
termos usa** em cada eixo:

```
p[k, a, ·] = softmax(rule_logits[k, a, ·])      # distribuição sobre as R MFs do eixo a
w_k        = Π_a ( Σ_r p[k,a,r] · μ_{a,r}(z_a) )  # força de disparo (t-norma produto)
w̄          = softmax_k( log w_k / τ )
y          = Σ_k w̄_k · ( C_k z + b_k )           # consequente TSK-1
```

Três consequências:

1. **`K` é hiperparâmetro, não consequência de `R^A`.** A capacidade é escolhida.
2. **Quando `p[k,a,·]` é quase uniforme, o eixo `a` é indiferente para a regra
   `k`.** Isso é a versão diferenciável de uma regra que não menciona uma
   variável — exatamente como as regras de engenharia reais, que citam duas ou
   três grandezas, não todas.
3. **Com `hard_rules=True`, `p` vira one-hot** (Gumbel-softmax no treino, argmax
   na inferência) e a regra passa a ser literalmente enumerável.

O regularizador `w_rule_entropy` empurra `p` para escolhas nítidas. Ele não
melhora desempenho — melhora **legibilidade**, e essa é uma métrica de primeira
classe neste projeto.

## Estabilidade numérica

O produto sobre `A` eixos de valores em (0,1] causa *underflow* rápido em
float32. A implementação subtrai o máximo por eixo antes de exponenciar e só
depois toma o logaritmo:

```python
mmax     = log_mu.amax(dim=-1, keepdim=True)
mu_shift = (log_mu - mmax).exp()
mixed    = einsum("hkar,bshar->bshka", p_rule, mu_shift)
per_axis = mixed.clamp_min(1e-12).log() + mmax
log_w    = per_axis.sum(-1)
```

Sem isso, `A=8` já produz `NaN` no gradiente em poucas centenas de passos.

## Consequentes de posto baixo

TSK-1 pediria uma matriz `d × d` por regra: com `K=256` e `d=512`, 67 M
parâmetros por camada. A implementação fatora em uma **base compartilhada** e
mapas por regra:

```
t   = U_h ᵀ h                    U_h ∈ R^{d × rank}     (compartilhada entre as regras)
y_k = V_k t + b_k                V_k ∈ R^{d_h × rank}   (por regra)
```

Isto é, as regras compartilham *o que olham* do embedding e diferem em *o que
fazem* com isso. É uma aproximação — a formulação plena daria a cada regra sua
própria projeção de entrada — e está registrada como QP-4. A intuição que a
justifica é que o antecedente já seleciona o regime; o consequente precisa
sobretudo especificar a resposta.

## Multi-cabeça fuzzy

Assim como a atenção tem cabeças, a camada tem `H` bancos de regras
independentes sobre subespaços de `d_model`. A leitura natural é que cada cabeça
se especializa em um aspecto (dinâmica térmica, saúde do atuador, condição de
carga), mas isso é hipótese a verificar — nada na formulação força a
especialização. Verificá-la é parte de C5.

## O que sai para interpretação

`AnfisTrace` expõe, por passagem:

* `firing` `(B,S,H,K)` — qual regra explicou cada token;
* `log_firing` — antes da normalização, útil para detectar regras mortas;
* `axes` `(B,S,H,A)` — os valores nos eixos antecedentes.

`interpret/rules.py` transforma isso em texto: antecedente de cada regra, tags
que mais a disparam, e regras dominantes de cada decisão.

## Limitação honesta

Os eixos antecedentes são projeções aprendidas de `h`, **não** variáveis de
processo. "eixo3 é alto" não é, por si só, uma afirmação de engenharia. A ponte
entre eixo latente e grandeza física é feita empiricamente (quais tokens
maximizam cada eixo) e é o elo mais fraco da cadeia de interpretabilidade —
QP-7. Uma alternativa a explorar: forçar a primeira camada ANFIS a operar sobre
eixos *ancorados*, projeções restritas aos slots fuzzy de entrada, tornando seus
antecedentes literalmente afirmações sobre tags.
