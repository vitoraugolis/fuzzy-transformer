# Estudo de caso — U-200 / R-201 (QUIMIVALE)

O acervo está versionado como `case_study.zip` na raiz do repositório. Para usar:

```bash
python scripts/setup_case_study.py       # extrai para case_study/
python scripts/run_u200.py --no-train    # ingere e reporta, sem treinar
python scripts/run_u200.py               # ingere, treina e avalia
```

O diretório extraído está em `.gitignore`: a fonte é sempre o zip.

## Por que este caso

O caso é um benchmark multissilo construído sobre o CSTR adimensional de
Lee, Jin & So (2025), *Algorithms* 18(7), 442. Durante 4,7 dias de campanha o
PID mantém a temperatura no set-point com **100% do produto em especificação** e
IAE por hora de 0,0535. No mesmo período, a abertura da FV-201 sobe de ~32% para
**94,7%**: a unidade consome toda a sua reserva de rejeição de perturbação **sem
que nenhuma métrica de erro registre nada**. Quando chega uma variação de apenas
3,1 K na temperatura da carga, a válvula satura e a unidade perde o controle da
temperatura por 21 horas.

| | A — baseline | B — com incidente | razão |
|---|---|---|---|
| IAE | 3,35 | 32,02 | 9,6× |
| Custo operacional | 0,024 | 3,292 | **138×** |
| Esforço de controle | 31,91 | 27,26 | 0,85× (B parece *melhor*) |

Isto é, literalmente, o caso motivador deste projeto — e com uma exigência a
mais: **a causa raiz não está escrita em nenhum documento isolado.** Ela exige o
cruzamento de pelo menos três silos.

## A planta

Reator CSTR encamisado de 12 m³, reação exotérmica A → B (resinas alquídicas e
poliéster). Temperatura de operação 229,2 °C, envelope 224,0–235,0 °C. O elemento
final é a **FV-201**, válvula de três vias que desvia o loop de água temperada
por E-201. Alarmes: TAH-201 em 235,2 °C, TAHH-201 (SIS) em 244,4 °C, **ZAH-201
em 80% de curso**.

A variável manipulada é a *abertura da válvula*, não a temperatura da camisa: a
saturação é um limite **físico**, o que coloca a capacidade do equipamento — e
não o controlador — como variável limitante.

## O acervo, por silo

| Silo | Pasta | Conteúdo |
|---|---|---|
| Processo/PIMS | `data/` | historian 5 em 5 min, alarmes, log de eventos, LIMS |
| Produção | `production/` | POs, COA, plano de campanhas |
| Manutenção | `maintenance/` | 7 ordens de trabalho |
| Operação | `operations/` | 7 relatórios de turno |
| Processo | `process/` | descrição, PFD, P&ID, procedimento operacional |
| Engenharia | `engineering/` | design basis, folhas de dados, memória de cálculo, filosofia de controle |
| Fabricante | `vendor/` | documentação da válvula e do atuador |
| Segurança | `safety/` | HAZOP |
| Especialista | `specialist/` | notas técnicas e memorandos |
| — | `investigation_package/` | o que a equipe de investigação receberia |
| — | `internal/` | ground truth e matriz de informação — **não usar no treino** |

`internal/` é material de avaliação. Usá-lo como entrada é vazamento e invalida
qualquer resultado.

## Como o FT-IC consome o acervo

Implementado em `src/fuzzytf/data/u200.py`:

**Vocabulário fuzzy ancorado.** Os centros de `TT201_PV_C` são o piso do
envelope, a região normal, o set-point, a região alta e o TAH-201 — "alto"
significa literalmente "na região do alarme". Os de `ZT201_pct` incluem o
ZAH-201 (80%), de modo que "curso alto" é o mesmo conceito da filosofia de
alarmes da unidade. 11 tags, 115 slots de estado.

**Episódios.** `episodes_from_case` monta o historian do incidente **e** o
baseline. Sem o baseline o modelo só vê a unidade degradada e não tem
contra-exemplo de operação com reserva de curso saudável.

**Alvos.** Ação = saída real do TIC-201 (imitação do que a planta fez);
envelope = regra declarada a partir do ZAH-201; orientações = supervisão
distante por alarmes e eventos; diagnóstico = estado `ANORMAL` declarado pela
própria unidade.

**Documentos.** `load_documents` extrai o texto dos 33 PDFs (via `pypdf`) e os
marca por silo; `KeywordRetriever` os recupera por estado fuzzy e tag.

**Divisão temporal.** O trecho final da série (o incidente) é validação. Divisão
aleatória seria vazamento.

## Achado da ingestão

**As ordens de trabalho que explicam o incidente são anteriores à janela do
historian.** A revisão geral da FV-201 (WO-3874) é de 29/07; o registro começa em
08/08. Supervisão distante por evento *dentro* da janela — a receita usual — não
produz **nenhum** rótulo de manutenção aqui.

O que sobra dentro da janela é o log de alarmes, e é dele que sai o sinal de
curso alto (ZAH-201 em 10/08 e 11/08 — e, em 11/08, o alarme é *suprimido no
console a pedido do turno*, o que o silencia justamente quando passa a importar).
A história de manutenção anterior precisa entrar por outro caminho: um canal de
**condição de equipamento** (`equipment_context`), e não como rótulo. É a QP-11,
e é provavelmente a questão mais importante da lista.

## Interface com o harness do caso

O caso define `BaseController` (`reset`/`update`) e reserva `ProposedController`
para a estratégia a desenvolver. `fuzzytf.integration.FTICController` implementa
esse contrato: recebe o que o DCS mede, tokeniza a janela e devolve a abertura
comandada de FV-201.

```python
from fuzzytf.integration import FTICController, load_case_module

case = load_case_module("case_study")            # importa tools/u200_case.py
ctrl = FTICController(model, book)
resultado = case.run_case(ctrl, case.CEN_INCIDENTE)
print(case.metrics(resultado))
```

Duas notas: `observation` **não** traz o `gain_trim` da válvula — inferir a
perda de capacidade é o problema, não a entrada. E `TT202_C`/`FT204_m3h` não
estão na observação do DCS; entram como NaN, marcadas `valid=False`, para que o
modelo enxergue o buraco em vez de um valor inventado.

## Critério de aceitação

Definido pelo próprio caso, e adotado como portão do estágio Walk:

> Um controlador só é superior neste benchmark se reduzir `tempo_saturado_h`
> **e** `producao_perdida_pct` **sem** aumentar `tempo_acima_TAH_h`.

Melhora de IAE isolada não conta. `esforco_controle` é enganoso aqui, porque o
cenário degradado o *reduz* — a válvula fica travada.

A esse critério o projeto acrescenta dois seus: a orientação correta
(manutenção na FV-201) precisa ser emitida **antes** da saturação, e a regra que
a sustenta precisa ser legível.
