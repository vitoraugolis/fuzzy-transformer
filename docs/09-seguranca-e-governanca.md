# Segurança e governança

Um modelo que escreve na malha é um elemento do sistema de controle. Este
documento define o que o impede de causar dano, e é pré-requisito de qualquer
atuação — não um anexo do fim.

## Camadas de proteção

O FT-IC **não** é uma camada de proteção. Ele opera na camada de controle
regulatório; o intertravamento (SIS) permanece independente, e nada aqui o
substitui, o modifica ou o contorna. No U-200, o TAHH-201 é do SIS: o modelo não
o vê como restrição negociável.

Ordem de precedência, do mais forte ao mais fraco:

1. **SIS / intertravamento** — independente, fora do alcance do modelo;
2. **Limites duros de instrumentação** — `hard_limits` na cabeça de controle;
   o modelo pode estreitar, nunca alargar;
3. **Envelope de contingência** — a faixa que o modelo prevê;
4. **Ação de controle** — sempre recortada pelas camadas acima.

`ControlHead.forward` implementa essa precedência: `hard_limits` sobrepõe-se ao
envelope previsto, e a ação é recortada no resultado.

## Modos de falha do próprio modelo

| Falha | Consequência | Mitigação |
|---|---|---|
| Envelope largo demais | autoriza ação proibida | perda assimétrica (2× no excesso); `violation_rate` como métrica de primeira classe |
| Entrada fora da distribuição | ação arbitrária | detecção por energia de disparo das regras: se nenhuma regra dispara com força, o estado é desconhecido → cair para o controlador de retaguarda |
| Instrumento em falha | modelo confia em valor falso | `valid=False`, máscara na atenção, tarefa de estado mascarado no pré-treino |
| Deriva do processo no tempo | modelo silenciosamente obsoleto | monitorar distribuição das forças de disparo; regra ativa que morre é sinal de deriva |
| Orientação errada em massa | equipe perde confiança e passa a ignorar | limiar por rótulo calibrado em precisão, não em F1; orientação sempre com evidência anexa |

O caminho de retaguarda é parte do desenho, não uma concessão:
`FTICController` aceita um `fallback` (o PID existente) e o usa enquanto a
janela não está cheia. A mesma via serve para reversão em operação.

## Explicação obrigatória

Nenhuma saída sai sozinha. Toda ação e toda orientação carregam:

* as regras disparadas e seus antecedentes (`interpret/rules.py`);
* a atribuição por tag (o ponteiro da cabeça de orientações);
* a janela de evidência que sustentou a decisão;
* o envelope aplicado e por quê.

Isso não é enfeite de interpretabilidade: é o que permite ao operador
**discordar com fundamento**, e é o registro que sobra para a investigação de um
evento.

## Progressão de autoridade

Nenhum salto direto de simulador para atuação:

1. **Offline** — avaliação sobre histórico; o modelo não vê a planta.
2. **Shadow mode** — roda em paralelo, registra, não atua. Mínimo três meses.
3. **Assistido** — sugere ao operador, que aceita ou rejeita; a taxa de rejeição
   é métrica de acompanhamento.
4. **Supervisionado** — atua dentro de um envelope estreito, com reversão
   automática para o PID a qualquer violação.
5. **Autônomo na faixa** — só depois de tudo acima, e sempre com faixa limitada.

## Vazamento como questão de segurança

Um modelo validado com vazamento parece seguro e não é. As três formas que
importam aqui: divisão aleatória de janelas sobrepostas; divisão aleatória no
tempo; e documento posterior ao instante entrando no contexto. No U-200 há uma
quarta: `internal/` contém o *ground truth* — usá-lo como entrada invalida tudo.

## Registro e auditoria

Cada versão em operação guarda: o `fingerprint` do vocabulário fuzzy, o
`config.json` do experimento, o checkpoint, o conjunto de avaliação e o
relatório do estágio. Vocabulário e checkpoint são inseparáveis — mudar a ordem
das tags invalida o modelo, e é por isso que o `fingerprint` existe.
