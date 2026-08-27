# Questões de pesquisa em aberto

Cada item tem: a pergunta, por que ela importa, como será respondida, e o
estado. São elas que definem o que ainda é *pesquisa* neste projeto — o resto é
engenharia.

---

**QP-1 — Como combinar as dimensões linguísticas de uma tag?**
Hoje as dimensões (`level`, `trend`, `error`) são **somadas** no espaço de
embedding. A alternativa é concatenar sub-espaços dedicados. A soma é mais
barata e flexível; a concatenação garante que uma dimensão não apague a outra.
*Como responder:* ablação direta no Crawl, medindo `skill_over_zero` e a
detecção precoce. *Estado:* aberta.

**QP-2 — De onde vêm as funções de pertinência?**
Três estratégias: fixa por engenharia (limites e alarmes), por quantis do
histórico, ou aprendida. A hipótese de trabalho é o híbrido — centros fixos,
larguras aprendidas.
*Já sabemos:* partição uniforme na faixa do instrumento é ruim o bastante para
impedir o aprendizado. No simulador, trocá-la por centros ancorados na faixa de
operação foi condição necessária para o modelo sair do preditor trivial.
*Estado:* parcialmente respondida.

**QP-3 — Quanto se perde sem o canal numérico?**
Se a partição fuzzy for adequada, o valor cru deveria ser redundante — com
partição triangular normalizada, o valor é recuperável dos dois graus de
pertinência. Se não for, a fuzzificação está descartando informação útil.
*Como responder:* ablação A2. *Estado:* aberta.

**QP-4 — Consequentes TSK: base compartilhada é suficiente?**
A implementação fatora o consequente em uma base de entrada compartilhada entre
as regras e um mapa de saída por regra. A formulação plena daria a cada regra
sua própria projeção de entrada, a um custo proibitivo.
*Como responder:* comparar em escala reduzida (`d_model` pequeno) onde a versão
plena cabe. *Estado:* aberta.

**QP-5 — A atenção deve ser causal?**
Hoje a janela é observada por inteiro. Máscara causal passa a fazer sentido se o
modelo for gerar *sequências* de ações futuras (horizonte de controle, como em
MPC) em vez de uma ação por instante.
*Estado:* aberta, ligada ao estágio Run (R2).

**QP-6 — Como escalar o comprimento da sequência?**
`S = M × W`. Com 500 tags e janela 128 são 64 000 tokens — inviável para atenção
densa. Candidatos: fatoração tag×tempo (atenção separada em cada eixo),
agrupamento por malha com atenção hierárquica, compressão temporal (instantes
antigos em resolução menor).
*Como responder:* medir custo e perda de desempenho de cada esquema no
simulador com M inflado artificialmente. *Estado:* aberta, crítica para R1.

**QP-7 — Como ancorar os eixos antecedentes em grandezas físicas?**
Os eixos do ANFIS são projeções aprendidas: "eixo3 é alto" não é, por si só, uma
afirmação de engenharia. É o elo mais fraco da cadeia de interpretabilidade.
Candidato: restringir a primeira camada ANFIS a eixos *ancorados* — projeções
sobre os próprios slots fuzzy de entrada — tornando seus antecedentes
literalmente afirmações sobre tags.
*Estado:* aberta.

**QP-8 — Qual codificador para os documentos?**
Hoje, hashing determinístico (placeholder). Opções: codificador treinado junto
com o modelo, codificador de linguagem congelado, ou modelo de linguagem externo
gerando descrições estruturadas que são então tokenizadas.
*Critério:* o ganho em F1 das orientações precisa compensar o custo e a perda de
auditabilidade. *Estado:* aberta.

**QP-9 — Qual o horizonte Δ da supervisão distante?**
Ordens de trabalho são rótulos tardios. Δ curto perde o período detectável; Δ
longo rotula operação normal como falha.
*Já sabemos:* no U-200, Δ=48 h satura os rótulos (dois deles ficam positivos em
100% das amostras); Δ=8 h dá um balanço utilizável (12–72%). Δ é hiperparâmetro
medido, nunca suposto.
*Estado:* parcialmente respondida.

**QP-10 — Como pesar as tarefas do treino multitarefa?**
As auxiliares são entropias cruzadas sobre ~100 slots (perda 3–7); a ação é um
Huber sobre um escalar (perda ~0,07). Sem repesar, o gradiente da ação
desaparece — foi exatamente o que aconteceu na primeira execução do Crawl.
Pesos fixos resolvem, mas de forma frágil; alternativas: normalização por
incerteza aprendida, ou por norma de gradiente.
*Estado:* contornada com pesos fixos; a solução principiada continua aberta.

**QP-11 — Como entra a condição do equipamento?**
O U-200 mostra que o dado de processo **não contém** a informação decisiva: a
perda de capacidade da FV-201 não é medida por nenhum instrumento, e as ordens
de trabalho que a explicam são anteriores à janela do historian. É preciso um
canal de *condição de equipamento* — tempo desde a última intervenção, tipo de
intervenção, achados — derivado do acervo de manutenção.
`u200.equipment_context` é um primeiro esboço.
*Estado:* aberta, e provavelmente a mais importante da lista.

**QP-12 — Como validar a ação de controle sem planta?**
Malha fechada em simulador responde por dinâmica, não por realismo. O caso U-200
oferece uma resposta melhor: um harness com contrato explícito
(`BaseController`) e métricas que penalizam o que o IAE esconde. Ainda assim,
resta a lacuna entre benchmark e planta — que só o *shadow mode* (R3) fecha.
*Estado:* encaminhada.

---

## Achados registrados

Coisas que já sabemos, e que custaram experimento:

1. **Tokens especiais precisam ser semeados com o estado atual.** Uma consulta
   `[ACT]` puramente aprendida (constante entre amostras) não recupera, só pela
   atenção, a diferença entre `k` e `k−1` de que a ação depende: a média sobre a
   janela a dilui. Semeando o token com o estado de `k` da própria variável
   manipulada, o modelo passa de desempenho nulo a `skill ≈ 0,46` em 300 passos.
   Um readout lido dos tokens de processo funciona; o `[ACT]` vazio não.
2. **A partição fuzzy precisa ser ancorada na faixa de operação.** Uniforme na
   faixa do instrumento, quase toda a massa cai em um único termo.
3. **O span de `trend` precisa vir dos dados.** Com span grande demais, tudo é
   "estável" e a dimensão não informa nada.
4. **O set-point precisa ser observável.** Sem ele, o erro de controle é
   inobservável e o alvo de imitação vira ruído.
5. **No U-200, a informação causal está fora da janela de dados.** As ordens de
   trabalho relevantes precedem o historian em 10 dias. Supervisão distante
   dentro da janela não as alcança — daí QP-11.
