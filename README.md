# Cartola Data — Dashboard & Escalação Inteligente

Pipeline completo de dados para o Cartola FC: coleta automática das APIs, enriquecimento estatístico, dashboard interativo e geração de escalação via LLM (Claude).

**Dashboard ao vivo:** https://filipenamorato.github.io/extracaoCartola/

---

## O que o projeto faz

1. **Coleta dados** da API do Cartola FC, The Odds API e football-data.org automaticamente 14x por dia via GitHub Actions
2. **Enriquece** cada atleta com métricas calculadas: média bayesiana, score de confronto, armadilha de preço, pontos esperados e mais
3. **Calibra** um modelo Ridge Regression com os dados reais de cada rodada, substituindo a heurística inicial quando o modelo performa melhor
4. **Publica** um dashboard interativo no GitHub Pages com tabela do Brasileirão, rankings de custo-benefício e análise de confrontos
5. **Gera escalação** automaticamente: envia os dados para o Claude (Anthropic) que analisa e monta o time ideal respeitando orçamento e formação
6. **Agenda lembretes** no Google Calendar antes do fechamento do mercado de cada rodada

---

## Stack

- **Linguagem:** Python 3.11
- **Dados:** pandas, numpy, Ridge Regression (scikit-learn)
- **APIs:** Cartola FC, The Odds API, football-data.org, Google Calendar
- **LLM:** Claude (Anthropic) via SDK
- **Infra:** GitHub Actions (CI/CD + coleta), GitHub Pages (dashboard)

---

## Métricas calculadas por atleta

| Coluna | Descrição |
| --- | --- |
| `media_bayesiana` | Média com encolhimento — prior por posição + faixa de preço, reduz ruído em atletas com poucos jogos |
| `pontos_esperados` | Previsão de pontos: heurística ou modelo Ridge calibrado (substituído automaticamente quando MAE melhora) |
| `score_confronto_100` | Score 0–100 composto por posição do adversário, momentum ofensivo/defensivo, forma recente e vantagem de mando |
| `condicao_mando` | `favoravel` / `favoravel_visitante` / `neutro` / `desfavoravel` — sinal de oportunidade do confronto |
| `armadilha_label` | `armadilha_forte` / `armadilha_leve` / `neutro` / `valor_bom` / `valor_oculto` — detecta jogadores caros com entrega abaixo do esperado |
| `confiabilidade` | Fator 0–1 baseado no número de jogos — pondera previsões de atletas com histórico curto |
| `oportunidade_confronto` | Percentil de oportunidade ofensiva ou defensiva por posição dentro da rodada |
| `recomendacao` | Indicação consolidada: `recomendado` / `monitorar` / `evitar` |
| `caro_e_vale` | Flag para jogadores acima do limiar de preço que ainda justificam o investimento |

---

## Calibração de `pontos_esperados`

O sistema roda `calibrar_pontos_esperados.py` a cada coleta para atualizar o modelo com dados reais.

**Heurística base:**
```
pontos_esperados = media_bayesiana × (score_confronto_100 / 50) × confiabilidade
```

**Modelo Ridge (ativo quando há ≥ 3 rodadas e ≥ 200 registros):**
- Features: `media_bayesiana`, `score_ratio`, `confiabilidade`, interação 3-way
- Features de forma recente: `media_3r`, `jogou_3r`, `tendencia` (últimas 3 rodadas)
- Ponderação por recência: decay = 0.9 (rodadas antigas pesam menos)
- Validação: leave-one-round-out cross-validation
- Adoção automática: modelo só entra se `MAE_OOS < MAE_heuristica`

Resultado salvo em `docs/data/current/calibracao_pontos.json`.

---

## Estrutura de dados

```
docs/data/
├── current/          # CSVs mais recentes (dashboard + LLM consomem daqui)
│   ├── atletas.csv          # ~250 atletas com todas as métricas
│   ├── atletas_pontuados.csv
│   ├── partidas.csv
│   ├── odds.csv
│   ├── tabela.csv
│   ├── times_rodada.csv
│   ├── status.csv
│   ├── calibracao_pontos.json
│   └── calibracao_score.json
├── historico/        # Snapshots pré/pós por rodada (base histórica do modelo)
│   └── rN/
│       ├── atletas_pre.csv
│       ├── atletas_pos.csv
│       └── atletas_pontuados.csv
└── raw/              # JSONs brutos das APIs (nunca modificados)
```

---

## Coleta automática

14 execuções por dia via GitHub Actions, com cobertura reforçada nos fins de semana (maioria dos jogos do Brasileirão):

| Dias | Horários BRT |
| --- | --- |
| Segunda a Sexta | 04h, 06h, 10h, 12h40, 15h, 17h, 18h30, 20h, 21h30, 23h |
| Sábado e Domingo | + 12h, 14h, 16h, 22h |

Cada execução: coleta → enriquece → calibra → commita CSVs → publica dashboard → agenda Google Calendar.

Para rodar manualmente: `Actions > Coleta Cartola FC > Run workflow`

---

## Uso local

```bash
pip install requests pandas numpy scikit-learn anthropic google-auth google-auth-httplib2 google-api-python-client

# Coleta e processa dados
python cartola_extractor.py

# Calibra modelo de previsão
python calibrar_pontos_esperados.py

# Gera escalação via LLM (requer ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sua_chave python gerarEscalacao.py
```

Os arquivos serão gerados em `docs/data/`.
