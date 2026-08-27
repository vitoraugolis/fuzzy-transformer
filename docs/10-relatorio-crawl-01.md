# Relatório — Crawl-01 (imitação do professor)

Configuração: `experiments/crawl_01_pid_imitation/config.json` ·
Execução: `python scripts/run_crawl.py --episodes 40 --steps 600` ·
Saídas: `runs/crawl-01/` (checkpoint, histórico, regras, explicação)

| | |
|---|---|
| Dados | 40 episódios × 600 amostras, divisão por episódio (30 treino / 10 validação) |
| Modelo | d=128, 3 blocos, ANFIS K=32 · A=6 · R=3 · H=4, janela 24, 434 599 parâmetros |
| Treino | 15 épocas, 2 100 passos, 32 min em 4 núcleos de CPU |

## Resultados em malha aberta

| Métrica | Valor | Leitura |
|---|---|---|
| `delta_mae` | 0,0564 | erro da ação, em unidades normalizadas |
| `delta_mae_zero_baseline` | 0,0868 | preditor trivial "não mexer" |
| **`skill_over_zero`** | **+0,350** | ganho sobre o trivial |
| `delta_r2` | 0,631 | |
| `violation_rate` | 0,302 | fração em que o envelope autorizaria ação proibida |
| `mean_width_ratio` | 1,03 | envelope com a largura certa, mas mal posicionado |
| `f1_macro` (orientações) | 0,476 | |
| `fault_accuracy` | 0,645 | 5 classes |

**Portão C3 — fechado.** O modelo supera o preditor trivial por margem clara
(35%), com divisão por episódio. Comparação de referência: uma sonda linear
sobre as features fuzzy atinge R² = 0,988 no mesmo alvo — ou seja, a informação
está na representação e o modelo ainda usa só parte dela. O teto não é a
tokenização.

## Resultados em malha fechada

Quatro cenários (sem falha, limitação de curso, atrito, incrustação), mesma
semente para todas as políticas.

| Política | IAE | Esforço | Reversões | Fração acima do alarme |
|---|---|---|---|---|
| PI (referência) | **0,237** | 1,38 | 98,8 | 0,000 |
| FT-IC | 2,544 | 0,46 | 28,5 | 0,017 |
| FT-IC sem envelope | 2,544 | 0,46 | 28,5 | 0,017 |

**Portão C4 — não fechou.** O critério era IAE ≤ 1,2 × o do PI nos episódios sem
falha; o resultado é 10,7×.

### Diagnóstico

O erro **não** é ruído: a correlação entre ação prevista e ação do professor é
0,760. O problema é **viés**.

```
viés médio (previsto − alvo) = −0,00404  ⇒  −0,040% de curso por amostra
```

Com ação incremental (forma velocidade) sobre uma planta integradora, um viés
constante não se cancela — ele **acumula**. Em 600 amostras, −0,040% por amostra
são 24% de curso de deriva, o que basta para explicar integralmente a diferença
de IAE. Os outros números confirmam: esforço de controle 3× menor e reversões
3,5× menores que as do PI — a política é lenta demais e escorrega, não oscila.

A identidade entre "FT-IC" e "FT-IC sem envelope" mostra que o envelope nunca
chegou a recortar a ação nesses rollouts: o problema está na ação, não no filtro.

### Encaminhamento (novo item C4a)

Em ordem de custo crescente:

1. **Âncora absoluta.** Prever também a posição absoluta `u` e corrigir
   lentamente o comando incremental na direção dela. É o análogo do *reset* de
   um controlador com bias — trata o viés diretamente, sem mexer na arquitetura.
2. **Perda sobre o comando acumulado.** Somar à perda o erro de `u` reconstruído
   ao longo de um horizonte de H passos, e não só o de `Δu` a cada passo. Penaliza
   viés explicitamente, que a perda pontual não vê.
3. **Ajuste em malha fechada (DAgger).** Rodar a política, coletar os estados que
   ela mesma visita e rotular com o professor. É o que corrige o desvio de
   distribuição, que aqui é a causa segunda: o modelo é treinado nos estados do
   PI e avaliado nos estados que ele próprio produz.

## Interpretabilidade (portão C5)

As regras são extraídas e legíveis. Exemplo real de `runs/crawl-01/rules.txt`:

```
bloco 1 cabeça 2 regra # 19 | força 0.1048
    SE eixo2 é medio[0.59] E eixo3 é baixo[0.64] E eixo5 é alto[0.52]
    dispara sobre: T-102 26%, V-097 19%, F-301 19%
```

**Portão C5 — não fechou, em ambas as metades.**

* *Antecipação:* não medida nesta execução, porque a política em malha fechada
  ainda não é utilizável — a métrica só faz sentido sobre rollouts válidos.
* *Legibilidade:* o antecedente é nítido (confianças de 0,52 a 0,72 nos eixos que
  entram na regra) e a regra cita duas ou três grandezas, como uma regra de
  engenharia. Mas a **atribuição é difusa**: 26%/19%/19% entre três tags é quase
  a distribuição uniforme entre cinco. Nenhuma das dez regras mais ativas tem
  tag dominante. É a manifestação empírica da QP-7: os eixos são projeções
  aprendidas, e nada no treino as obriga a se alinhar com grandezas físicas.

A explicação por decisão (`runs/crawl-01/explicacao.txt`) sai completa — estado
linguístico, ação com sua distribuição sobre termos, envelope, diagnóstico,
orientações e regras dominantes por bloco. A infraestrutura de auditoria está
pronta; o que falta é o conteúdo ser ancorado.

## Conclusão

Dos três portões desta rodada, **um fechou** (C3) e **dois não** (C4, C5).

O que a rodada estabeleceu de positivo: a arquitetura treina de forma estável,
supera o preditor trivial, aprende o envelope com a largura certa, e a cadeia de
interpretabilidade funciona ponta a ponta. O que ela estabeleceu de negativo é
mais útil: o gargalo em malha fechada é **viés acumulado da ação incremental**,
não capacidade do modelo — e a interpretabilidade esbarra na ancoragem dos
eixos, não na extração das regras.

As ablações do estágio (A1–A8, ver `docs/06`) ainda não foram executadas. Rodá-las
antes de C4a seria inverter a ordem: não faz sentido medir a contribuição do
ANFIS contra um MLP enquanto a política ainda não fecha malha.
