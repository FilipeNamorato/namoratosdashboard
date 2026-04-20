# cartola-data

Coleta automática de dados da API do Cartola FC via GitHub Actions, com dashboard visual hospedado no GitHub Pages. Inclui calibração dinâmica de modelo para previsão de pontos esperados baseado em Ridge regression.

## Dashboard

Acesse em: **https://filipenamorato.github.io/extracaoCartola/**

## Arquivos gerados em `docs/data/`

| Arquivo | Conteúdo |
| --- | --- |
| `atletas_mercado.csv` | Todos os atletas do mercado com preço, média, variação e scouts |
| `atletas_mercado.json` | JSON bruto do endpoint `/atletas/mercado` |
| `atletas_pontuados.csv` | Atletas que pontuaram na rodada atual |
| `atletas_pontuados.json` | JSON bruto do endpoint `/atletas/pontuados` |
| `atletas_enriquecido.csv` | Atletas com colunas extras: mandante, adversário, tendência, custo-benefício, rank, armadilha |
| `partidas.csv` | Partidas da rodada |
| `partidas.json` | JSON bruto do endpoint `/partidas` |
| `rodadas.csv` | Histórico de rodadas |
| `rodadas.json` | JSON bruto do endpoint `/rodadas` |
| `mercado_status.csv` | Status atual do mercado com rodada atual e horário de fechamento |
| `mercado_status.json` | JSON bruto do endpoint `/mercado/status` |
| `log.csv` | Histórico de execuções com status e timestamp |

## Colunas extras em `atletas_enriquecido.csv`

| Coluna | Descrição |
| --- | --- |
| `mandante` | True se o clube joga em casa nessa rodada |
| `adversario` | Abreviação do adversário na rodada |
| `tendencia` | alta / baixa / estavel com base na variação |
| `custo_beneficio` | média ÷ preço |
| `cb_rank` | Ranking de custo-benefício dentro da posição |
| `armadilha` | True se preço acima da mediana mas média abaixo |
| `status_label` | Texto legível do status (Provável, Dúvida, etc.) |
| `pontos_esperados` | Pontos esperados (heurística ou modelo calibrado) |
| `media_bayesiana` | Média com encolhimento (prior por posição + faixa de preço) |
| `confiabilidade` | Fator 0-1 baseado no número de jogos |
| `oportunidade_confronto` | Percentil 0-1 de oportunidade ofensiva/defensiva por posição |
| `score_confronto_100` | Score composto do confronto normalizado 0-100 |

## Calibração de `pontos_esperados`

O sistema executa `calibrar_pontos_esperados.py` continuamente para melhorar as previsões.

**Heurística original:** `pontos_esperados = media_bayesiana × (score_confronto_100/50) × confiabilidade`

**Modelo aprendido (quando há dados suficientes ≥3 rodadas, ≥200 registros):**
- Ridge regression com features: `media_bayesiana`, `score_ratio`, `confiabilidade`, interação 3-way
- Features de forma recente: `media_3r`, `jogou_3r`, `tendencia` (últimas 3 rodadas anteriores)
- Ponderação por recência: rodadas antigas pesam menos (decay = 0.9)
- Validação: leave-one-round-out cross-validation
- **Adoção**: modelo só substitui heurística se `MAE_OOS < MAE_heuristica`

Saída: `docs/data/current/calibracao_pontos.json`

## Agendamento

Coleta automática 9 vezes ao dia via GitHub Actions:

| Horário BRT | Horário UTC |
| --- | --- |
| 06h00 | 09h00 |
| 10h00 | 13h00 |
| 12h40 | 15h40 |
| 15h00 | 18h00 |
| 17h00 | 20h00 |
| 18h30 | 21h30 |
| 20h00 | 23h00 |
| 21h30 | 00h30 |
| 23h00 | 02h00 |

Para rodar manualmente: `Actions > Coleta Cartola FC > Run workflow`

## Uso local

```bash
pip install requests pandas numpy
python cartola_extractor.py      # coleta e processa dados
python calibrar_pontos_esperados.py  # calibra modelo de previsão (rodadas ≥3 com ≥200 registros)
```

Os arquivos serão gerados em `docs/data/`.

## Dashboard — Brasileirão

A aba **Brasileirão** exibe:
- **Tabela**: posição, pontos, aproveitamento, momentum (ofensivo/defensivo), forma recente, sequência
- **Cards**: visão consolidada por time com estatísticas casa/fora
- **Nomes padronizados**: "Palmeiras (PAL)", "Atlético-MG (CAM)" etc. (mapeamento limpeza de nomes da API)
