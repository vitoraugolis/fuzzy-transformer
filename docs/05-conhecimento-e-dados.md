# Conhecimento além do dado de processo

O ponto mais difícil da proposta não é a arquitetura — é **como treinar o modelo
com WOs, POs, relatórios de turno, manuais, HAZOPs e matrizes de informação.**
Estes artefatos não são séries temporais, não são rótulos e, na maior parte,
nem sequer estão alinhados no tempo com o processo. Este documento define a
estratégia adotada.

## Três mecanismos, três papéis

| Mecanismo | O que injeta | Onde entra | Estágio |
|---|---|---|---|
| **Recuperação** | contexto pontual | tokens de contexto na sequência | Walk |
| **Aterramento de regras** | estrutura causal | banco de regras da camada ANFIS | Walk/Run |
| **Supervisão distante** | rótulos | alvos das cabeças de orientação | Walk |

A separação é deliberada. Um único mecanismo — jogar tudo em um recuperador —
desperdiça o que cada tipo de documento tem de específico: um HAZOP é uma
*regra*, uma WO é um *rótulo com data*, um manual é *contexto de referência*.

## 1. Recuperação (documentos como contexto)

A consulta é montada a partir do estado corrente: as tags ativas e sua descrição
linguística ("V-097 muito_alto, T-102 subindo_rapido"). O recuperador
(`data/documents.py::KeywordRetriever`) devolve os `k` documentos mais
relevantes, com bônus para os que citam explicitamente as tags envolvidas. Os
trechos são codificados e entram como tokens de contexto, visíveis à atenção.

O codificador atual é um *placeholder* determinístico por hashing. Ele existe
para que o caminho de contexto seja exercitável hoje; a escolha do codificador
real (treinado junto, congelado, ou um modelo de linguagem externo) é QP-8.

**Filtro temporal obrigatório.** Só podem ser recuperados documentos com data
anterior ao instante `k`. Um relatório de turno escrito *depois* do evento
descreve o evento — usá-lo é vazamento, e o vazamento aqui é especialmente
insidioso porque produz resultados excelentes e completamente falsos.

## 2. Aterramento de regras (HAZOP → banco de regras)

Uma linha de HAZOP tem exatamente a forma de uma regra fuzzy:

```
Desvio: MAIS temperatura no vaso V-201
Causa: falha de fechamento da V-097
Consequência: abertura da PSV, parada
Recomendação: manutenção preditiva na V-097
```

`hazop_to_rules` converte isso em:

```
SE (T-102 é level:alto) E (V-097 é level:alto)
ENTÃO acionar_manutencao_valvula
```

Essas regras candidatas são usadas de duas formas:

* **inicialização** — os `rule_logits` das primeiras regras do banco são
  ajustados para favorecer as MFs correspondentes ao antecedente;
* **regularização** — um termo de perda que penaliza o afastamento entre a regra
  aprendida e a regra documentada, com peso decrescente ao longo do treino.

A extração é conservadora por decisão: só casa desvios e tags reconhecidos, e
descarta o resto. É melhor aterrar poucas regras corretas do que muitas
duvidosas — uma regra errada aterrada é pior que nenhuma, porque carrega a
autoridade do documento.

**Métrica de sucesso (portão W4):** ao menos 30% das regras aterradas continuam
ativas após o treino. As que morrem são material de análise, não fracasso: ou o
desvio previsto no HAZOP não se manifesta no histórico, ou o mecanismo de
aterramento é fraco demais — e distinguir os dois casos é resultado de pesquisa.

## 3. Supervisão distante (WOs como rótulos)

Uma WO mecânica na V-097 aberta em `t` implica que **antes** de `t` havia um
problema mecânico observável na V-097. Logo, as janelas em `[t − Δ, t]` são
exemplos positivos de "acionar manutenção da válvula".

Três decisões delicadas:

**O horizonte Δ.** WO é rótulo tardio: ela é aberta quando o problema ficou
óbvio, não quando ficou detectável. Δ muito curto perde o período interessante;
Δ muito longo rotula operação normal como falha. Δ é hiperparâmetro **medido**
(varredura com validação em episódios de falha conhecida), nunca suposto.

**O rótulo negativo.** Ausência de WO não é ausência de problema — pode ser
apenas ausência de registro. O tratamento correto é *positive-unlabeled*, não
classificação binária ingênua. Simplificação aceita no Walk: tratar como
negativo apenas períodos com histórico de manutenção documentado e sem
ocorrência.

**POs como contexto, não rótulo.** Uma ordem de produção muda o regime alvo
(taxa, produto, especificação), o que altera o que é "normal". Ela entra como
contexto e, quando houver estrutura suficiente, como condicionamento explícito
do envelope — operar 12% acima do projeto legitimamente estreita a faixa
admissível.

## Alinhamento temporal

O elo mais frágil de toda a cadeia. Historiador de processo, sistema de
manutenção e relatórios de turno têm relógios, fusos e granularidades
diferentes; WOs registram a data de abertura, não a da ocorrência; relatórios de
turno referem-se ao turno inteiro, não a um instante.

Convenções adotadas:

1. tudo é convertido para UTC na ingestão;
2. eventos sem hora são ancorados no início do turno correspondente, e recebem
   uma marca de incerteza temporal;
3. nenhum documento com data posterior a `k` entra no contexto de `k`;
4. a granularidade do rótulo nunca é mais fina que a da fonte — um evento de
   turno rotula o turno, não a amostra.

## Ordem de implementação

O caminho crítico é rótulo, não modelo. A sequência recomendada é: (1) ingerir e
alinhar, (2) rotular por supervisão distante e **auditar à mão uma amostra**,
(3) só então ligar recuperação e aterramento. Inverter essa ordem leva a passar
semanas ajustando um recuperador sobre rótulos que ninguém verificou.
