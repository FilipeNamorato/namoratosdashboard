# Regras do Cartola FC — referência

Referência consolidada das regras do jogo, para consulta ao mexer no
`cartola_extractor.py` e nas métricas do dashboard.

> **Origem dos dados.** O `cartola.globo.com/#!/entenda-mais` é uma SPA com rota
> em hash — o conteúdo não vem no HTML e o domínio bloqueia fetch automatizado.
> Então: o que é **verificável pela API** (esquemas, posições, status) foi puxado
> direto dela; o resto vem de fontes secundárias e está marcado com o nível de
> confiança. Coletado em 22/08/2026 (temporada 2026, rodada 24).

---

## 1. Posições e status

Direto de `https://api.cartola.globo.com/atletas/mercado` — **fonte primária**.

| ID | Posição | Abrev. |
|---|---|---|
| 1 | Goleiro | gol |
| 2 | Lateral | lat |
| 3 | Zagueiro | zag |
| 4 | Meia | mei |
| 5 | Atacante | ata |
| 6 | Técnico | tec |

| ID | Status | Observação |
|---|---|---|
| 2 | Dúvida | pode ou não jogar |
| 3 | Suspenso | não joga |
| 5 | Contundido | não joga |
| 6 | Nulo | fora da rodada / sem jogo |
| 7 | Provável | expectativa de jogar |

O extractor mantém só `status_id ∈ {7, 2}` no `docs/data/current/atletas.csv`
(exclui o 6 — Nulo).

---

## 2. Esquemas táticos

Direto de `https://api.cartola.globo.com/esquemas` — **fonte primária**.

| Esquema | GOL | LAT | ZAG | MEI | ATA | TEC |
|---|---|---|---|---|---|---|
| 3-4-3 | 1 | 0 | 3 | 4 | 3 | 1 |
| 3-5-2 | 1 | 0 | 3 | 5 | 2 | 1 |
| 4-3-3 | 1 | 2 | 2 | 3 | 3 | 1 |
| 4-4-2 | 1 | 2 | 2 | 4 | 2 | 1 |
| 4-5-1 | 1 | 2 | 2 | 5 | 1 | 1 |
| 5-3-2 | 1 | 2 | 3 | 3 | 2 | 1 |
| 5-4-1 | 1 | 2 | 3 | 4 | 1 | 1 |

Sempre 11 jogadores de linha + 1 técnico. O `esquema_default_id` vem no
`/mercado/status` (hoje `4` = 4-4-2).

---

## 3. Tabela de scouts

Fonte secundária (cartolafcbrasil.com.br) — **confiança média**. Os valores de
`DE` e `DS` mudaram entre temporadas; se algum cálculo depender deles, vale
reconferir antes.

### Positivos

| Sigla | Nome | Pontos |
|---|---|---|
| G | Gol | **+8,0** |
| A | Assistência | **+5,0** |
| SG | Saldo de gols (não sofreu gol) | **+5,0** |
| DP | Defesa de pênalti | **+7,0** |
| FT | Finalização na trave | +3,0 |
| DS | Desarme | +1,5 |
| DE | Defesa (goleiro) | +1,3 |
| FD | Finalização defendida | +1,2 |
| PS | Pênalti sofrido | +1,0 |
| FF | Finalização para fora | +0,8 |
| FS | Falta sofrida | +0,5 |

### Negativos

| Sigla | Nome | Pontos |
|---|---|---|
| PP | Pênalti perdido | **−4,0** |
| GC | Gol contra | −3,0 |
| CV | Cartão vermelho | −3,0 |
| CA | Cartão amarelo | −1,0 |
| GS | Gol sofrido | −1,0 |
| PC | Pênalti cometido | −1,0 |
| FC | Falta cometida | −0,3 |
| I | Impedimento | −0,1 |

**SG** só vale para goleiro, lateral e zagueiro. **DE**, **DP** e **GS** são
exclusivos de goleiro.

Todas essas siglas existem como colunas `scout_*` no `atletas.csv`.

---

## 4. Pontuação do técnico

Fonte secundária — **confiança baixa, confirmar**. A descrição encontrada é que o
técnico pontua pela **média dos jogadores do seu clube que entraram em campo**,
não por scouts próprios. Isso é relevante porque significa que o técnico é uma
aposta no **desempenho coletivo do time**, não em eventos individuais — o que
casa com o `score_confronto_100` ser bom preditor para a posição.

---

## 5. Capitão

- O capitão pontua **em dobro**.
- A dobra vale **inclusive para pontuação negativa** — capitão que leva vermelho
  custa o dobro.
- Se o capitão **não entrar em campo**, não há dobra; o reserva que o substitui
  entra com pontuação simples.

Implicação prática: o capitão deve ser o jogador com maior **teto**, mas também
com titularidade mais garantida — risco de não jogar é penalizado duas vezes
(perde a dobra e ainda desperdiça o slot).

---

## 6. Banco de reservas

- **Não consome cartoletas.** Escalar reserva não gasta patrimônio — a única
  restrição é o teto de preço. (Erro comum: tratar o banco como se disputasse
  orçamento com os titulares.)
- Um reserva só pode custar **menor ou igual ao titular mais barato da mesma
  posição**. Ex.: se o ZAG titular mais barato custa C$6,33, o ZAG do banco
  precisa custar ≤ C$6,33.
- São 5 vagas: GOL, LAT, ZAG, MEI, ATA. **Não há reserva de técnico.**
- O banco é **opcional**.
- A substituição é **automática** e só ocorre se o titular **não entrar em campo
  em nenhum momento**. Titular que joga 1 minuto não é substituído.
- Se o reserva substitui um titular mais caro, o Cartola **devolve a diferença**
  ao patrimônio.

Consequência para escalação: reserva só vale se for alguém que **vai jogar** —
um reserva barato que também fica no banco do time real rende zero.

---

## 7. Valorização e desvalorização

Fonte secundária — **confiança baixa nos números exatos**.

- O preço varia conforme a pontuação da rodada frente à expectativa embutida no
  preço.
- Regra citada para a 1ª rodada: valoriza quem pontua ≈ **46% do preço**
  (`preço × 0,46`).
- A partir da 4ª rodada o critério passa a ser comparativo com o desempenho
  recente do próprio jogador.

O `atletas.csv` já traz `min_valorizar` (pontuação mínima estimada para
valorizar) e `variacao`, então preferir essas colunas a recalcular pela fórmula.

---

## 8. Mercado

Do `/mercado/status` — **fonte primária**.

- `status_mercado`: `1` = aberto, `2` = fechado (em andamento).
- `rodada_atual`, `game_over`, `fechamento` (dict com dia/mês/ano/hora/minuto e
  `timestamp`).
- `cartoleta_inicial`: 100 (patrimônio no início da temporada).
- Escalação só é aceita com o mercado **aberto**; fecha antes do primeiro jogo
  da rodada.

---

## 9. Pegadinhas que já causaram bug ou decisão errada aqui

1. **Reserva não gasta cartoleta** — ver seção 6. Já levou a descartar opções de
   banco por "falta de saldo" que na verdade eram gratuitas.
2. **Teto do reserva é por posição e olha o titular mais barato** — trocar um
   titular por um mais barato **derruba o teto** do reserva daquela posição e
   pode invalidar quem já estava no banco.
3. **Odds do visitante** — o `odds.csv` contém jogos de rodadas futuras. Chavear
   o merge só pelo mandante faz o `odd_vis`/`prob_vis` virem do jogo errado
   (corrigido em `enriquecer_partidas`, que agora usa a chave `(casa, visitante)`).
4. **`score_confronto_100` e `pontos_esperados` podem discordar** — um jogador
   pode ter `pontos_esperados` alto com confronto péssimo (< 40), porque o peso
   de `prob_vitoria` no score é pequeno perto de `media_bayesiana` e forma.
   Olhar as duas colunas antes de recomendar.

---

## Fontes

- API Cartola: `api.cartola.globo.com` — `/esquemas`, `/mercado/status`, `/atletas/mercado`
- [Scouts — Cartola FC Brasil](https://www.cartolafcbrasil.com.br/scouts)
- [Sistema de pontuação — Cartola FC Brasil](https://www.cartolafcbrasil.com.br/tutoriais/7/como-funciona-o-sistema-de-pontuacao-do-cartola-fc)
- [Banco de reservas — Dicas Cartola](https://www.dicascartola.com.br/como-jogar-cartola-fc/cartola-fc-como-funciona-banco-de-reservas/)
- [Banco de reservas — Cartola FC Mix](https://cartolafcmix.com/arquivo-mix/banco-de-reservas-no-cartola-fc-como-funciona-a-principal-novidade-de-2021/)
