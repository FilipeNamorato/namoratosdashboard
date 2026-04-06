"""
cartola_extractor.py
--------------------
Extrai dados da API do Cartola FC, enriquece com colunas calculadas
e salva na estrutura de pastas do projeto.

ESTRUTURA DE SAÍDA:
    docs/data/raw/          → JSONs brutos das APIs (nunca modificados)
    docs/data/current/      → CSVs processados (dashboard + LLM consomem daqui)
    docs/data/historico/rN/ → Snapshots por rodada (base do modelo ML futuro)
    llm/input/              → CSVs pré-filtrados prontos para o modelo de linguagem

COLUNAS GERADAS em atletas.csv (current):
    mandante            → True se o clube joga em casa nessa rodada
    adversario          → abreviação do adversário
    tendencia           → 'alta' | 'baixa' | 'estavel'
    confiabilidade      → fator 0-1 baseado no número de jogos
    media_bayesiana     → média com encolhimento para a média da posição
    residuo_z           → z-score do desvio entre média e o esperado pelo preço
    armadilha_label     → armadilha_forte | armadilha_leve | neutro | valor_bom | valor_oculto
    pb_media            → Pontos Base médios (excluindo Gols, Assistências e SG)
    resiliencia_pct     → proporção da média que vem de pontos base (0-1)
    min_valorizar       → estimativa de pontos mínimos para não desvalorizar
    custo_beneficio     → media_bayesiana / preco x confiabilidade
    cb_rank             → ranking dentro da posição por custo-benefício
    status_label        → texto legível do status
    time_pos            → posição do time na tabela
    adv_pos             → posição do adversário na tabela
    vantagem_mando      → delta de aproveitamento no mando específico (pp)
    oportunidade_confronto → percentil 0-1 de oportunidade por posição
    time_momentum_of    → ratio gols marcados recentes / média da temporada
    time_momentum_def   → ratio gols sofridos recentes / média
    adv_momentum_of     → momentum ofensivo do adversário
    adv_momentum_def    → momentum defensivo do adversário
    sequencia_time      → +N = N jogos invicto, -N = N jogos sem ganhar
    forma_score_time    → pontuação de forma ponderada 0-1
    score_confronto_z   → score composto do confronto, normalizado por posição
    score_confronto_100 → score_confronto_z convertido para escala 0-100
"""

import json
import os
import unicodedata
import numpy as np
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE PASTAS
# ─────────────────────────────────────────────────────────────

# Dados públicos (GitHub Pages)
RAW_DIR       = Path("docs/data/raw")       # JSONs brutos das APIs
CURRENT_DIR   = Path("docs/data/current")   # CSVs processados (dashboard)
HISTORICO_DIR = Path("docs/data/historico") # Snapshots por rodada

# Dados privados (uso interno / LLM)
LLM_INPUT_DIR = Path("llm/input")           # CSVs pré-filtrados pro modelo

# Criar todas as pastas se não existirem
for pasta in [RAW_DIR, CURRENT_DIR, HISTORICO_DIR, LLM_INPUT_DIR]:
    pasta.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://api.cartola.globo.com"
ENDPOINTS = {
    "mercado":   "/atletas/mercado",
    "pontuados": "/atletas/pontuados",
    "partidas":  "/partidas",
    "rodadas":   "/rodadas",
    "status":    "/mercado/status",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json",
}
STATUS_LABEL = {
    2: "Dúvida",
    3: "Suspenso",
    5: "Contundido",
    6: "Nulo",
    7: "Provável",
}

ODDS_API_KEY      = os.environ.get("ODDS_API_KEY", "")
ODDS_URL          = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
BRASILEIRAO_ID    = "BSA"

NOMES_PARA_ABR = {
    # Nomes do football-data.org (fonte primária)
    "CR Flamengo":            "FLA",
    "SE Palmeiras":           "PAL",
    "CA Mineiro":             "CAM",
    "Fluminense FC":          "FLU",
    "SC Corinthians Paulista": "COR",
    "São Paulo FC":           "SAO",
    "SC Internacional":       "INT",
    "Grêmio FBPA":            "GRE",
    "Botafogo FR":            "BOT",
    "CR Vasco da Gama":       "VAS",
    "EC Bahia":               "BAH",
    "Cruzeiro EC":            "CRU",
    "CA Paranaense":          "CAP",
    "Santos FC":              "SAN",
    "EC Vitória":             "VIT",
    "RB Bragantino":          "RBB",
    "Mirassol FC":            "MIR",
    "Chapecoense AF":         "CHA",
    "Coritiba FBC":           "CFC",
    "Clube do Remo":          "REM",
    "Sport Recife":           "SPT",
    "Sport Club do Recife":   "SPT",
    "Ceará SC":               "CEA",
    "Fortaleza EC":           "FOR",
    "Juventude":              "JUV",
    "América Mineiro":        "AME",
    "Goiás EC":               "GOI",
    "Cuiabá EC":              "CUI",
    # Nomes alternativos / variações
    "Flamengo":               "FLA",
    "Palmeiras":              "PAL",
    "Atletico Mineiro":       "CAM",
    "Atletico MG":            "CAM",
    "Atlético Mineiro":       "CAM",
    "Fluminense":             "FLU",
    "Corinthians":            "COR",
    "Sao Paulo":              "SAO",
    "São Paulo":              "SAO",
    "Internacional":          "INT",
    "Gremio":                 "GRE",
    "Grêmio":                 "GRE",
    "Botafogo":               "BOT",
    "Vasco da Gama":          "VAS",
    "Bahia":                  "BAH",
    "Cruzeiro":               "CRU",
    "Atletico Paranaense":    "CAP",
    "Athletico Paranaense":   "CAP",
    "Atlético Paranaense":    "CAP",
    "Santos":                 "SAN",
    "Vitoria":                "VIT",
    "Vitória":                "VIT",
    "Bragantino-SP":          "RBB",
    "Red Bull Bragantino":    "RBB",
    "Mirassol":               "MIR",
    "Chapecoense":            "CHA",
    "Coritiba":               "CFC",
    "Remo":                   "REM",
    "Ceara":                  "CEA",
    "Ceará":                  "CEA",
    "Fortaleza":              "FOR",
    "America Mineiro":        "AME",
    "Goias":                  "GOI",
    "Goiás":                  "GOI",
    "Cuiaba":                 "CUI",
    "Cuiabá":                 "CUI",
}

BRT = timezone(timedelta(hours=-3))

# ── Parâmetros estatísticos ───────────────────────────────────
JOGOS_CONFIANCA_PLENA = 8
JANELA_MOMENTUM       = 5
PESOS_FORMA           = [0.10, 0.15, 0.20, 0.25, 0.30]
POSICOES_ATAQUE       = {"Atacante", "Meia", "Técnico"}
POSICOES_DEFESA       = {"Zagueiro", "Lateral", "Goleiro"}

# Pesos de pontuação por evento volátil (para cálculo do PB)
PESO_GOL_POS = {
    "Goleiro":  6.0,
    "Zagueiro": 6.0,
    "Lateral":  6.0,
    "Meia":     5.0,
    "Atacante": 8.0,
    "Técnico":  3.0,
}
PESO_SG_POS = {
    "Goleiro":  5.0,
    "Zagueiro": 3.0,
    "Lateral":  3.0,
    "Meia":     1.0,
    "Atacante": 1.0,
    "Técnico":  0.0,
}
PESO_ASSISTENCIA = 5.0

# Colunas que vão para o llm/input/ (apenas o que o modelo precisa)
COLUNAS_LLM = [
    "atleta_id", "nome", "clube", "posicao", "status_id", "status_label",
    "preco", "variacao", "media", "jogos",
    "mandante", "adversario", "tendencia",
    "min_valorizar", "pb_media", "resiliencia_pct", "confiabilidade",
    "media_bayesiana", "residuo_z", "armadilha_label",
    "custo_beneficio", "cb_rank",
    "oportunidade_confronto", "vantagem_mando",
    "score_confronto_z", "score_confronto_100",
    "time_momentum_of", "time_momentum_def",
    "adv_momentum_of", "adv_momentum_def",
    "sequencia_time", "forma_score_time",
]

# ─────────────────────────────────────────────────────────────
# HELPERS DE MAPEAMENTO
# ─────────────────────────────────────────────────────────────

def normalizar_nome(nome: str) -> str:
    return unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii").lower().strip()

def get_nomes_por_abr(abr: str) -> list:
    return [nome for nome, a in NOMES_PARA_ABR.items() if a == abr]

def get_tabela_row(abr: str, tabela_idx: pd.DataFrame):
    for nome in get_nomes_por_abr(abr):
        if nome in tabela_idx.index:
            return tabela_idx.loc[nome]
    abr_norm_nomes = [normalizar_nome(n) for n in get_nomes_por_abr(abr)]
    for idx_nome in tabela_idx.index:
        if normalizar_nome(idx_nome) in abr_norm_nomes:
            return tabela_idx.loc[idx_nome]
    if abr and abr not in ("—", ""):
        print(f"  [WARN] sem match para abr='{abr}'")
    return None

def get_momentum_time(abr: str, momentum: dict) -> dict:
    for nome in get_nomes_por_abr(abr):
        if nome in momentum:
            return momentum[nome]
    abr_norm_nomes = [normalizar_nome(n) for n in get_nomes_por_abr(abr)]
    for k in momentum:
        if normalizar_nome(k) in abr_norm_nomes:
            return momentum[k]
    return {}

# ─────────────────────────────────────────────────────────────
# REQUISIÇÃO CARTOLA
# ─────────────────────────────────────────────────────────────

def get_json(endpoint_key: str) -> dict:
    url = BASE_URL + ENDPOINTS[endpoint_key]
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()

def salvar_raw_json(nome: str, dados: dict):
    """Salva JSON bruto em docs/data/raw/ sem modificação."""
    path = RAW_DIR / f"{nome}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
# ODDS API
# ─────────────────────────────────────────────────────────────

def get_odds() -> pd.DataFrame:
    if not ODDS_API_KEY:
        print("  SKIP — ODDS_API_KEY não definida")
        return pd.DataFrame()

    resp = requests.get(ODDS_URL, params={
        "apiKey":      ODDS_API_KEY,
        "regions":     "eu",
        "markets":     "h2h",
        "oddsFormat":  "decimal",
    }, timeout=15)
    resp.raise_for_status()
    jogos = resp.json()

    # Salva JSON bruto antes de processar
    salvar_raw_json("odds", jogos)

    rows = []
    for jogo in jogos:
        abr_casa = NOMES_PARA_ABR.get(jogo["home_team"])
        abr_vis  = NOMES_PARA_ABR.get(jogo["away_team"])
        if not abr_casa or not abr_vis:
            continue

        odd_casa = odd_vis = odd_empate = None
        for bm in jogo.get("bookmakers", [])[:1]:
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        abr = NOMES_PARA_ABR.get(o["name"])
                        if abr == abr_casa:   odd_casa   = o["price"]
                        elif abr == abr_vis:  odd_vis    = o["price"]
                        else:                 odd_empate = o["price"]

        if not odd_casa or not odd_vis:
            continue

        soma      = (1/odd_casa) + (1/odd_vis) + (1/odd_empate if odd_empate else 0)
        prob_casa = round((1/odd_casa) / soma, 3) if soma else None
        prob_vis  = round((1/odd_vis)  / soma, 3) if soma else None

        def classificar(odd):
            if odd < 1.5: return "favorito_forte"
            if odd < 2.0: return "favorito"
            if odd < 2.5: return "equilibrado"
            return "zebra"

        rows.append({
            "abr_casa":      abr_casa,
            "abr_vis":       abr_vis,
            "odd_casa":      odd_casa,
            "odd_vis":       odd_vis,
            "odd_empate":    odd_empate,
            "prob_casa":     prob_casa,
            "prob_vis":      prob_vis,
            "forca_casa":    classificar(odd_casa),
            "forca_vis":     classificar(odd_vis),
            "commence_time": jogo.get("commence_time"),
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# FOOTBALL-DATA.ORG — BRASILEIRÃO
# ─────────────────────────────────────────────────────────────

def _fd_get(path: str, params: dict = None) -> dict:
    url = f"{FOOTBALL_DATA_URL}{path}"
    resp = requests.get(
        url,
        headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def _aprov(w: int, d: int, l: int) -> float:
    total = w + d + l
    return round((w * 3 + d) / (total * 3) * 100, 1) if total else 0.0

def build_team_history(matches: list) -> dict:
    history  = defaultdict(list)
    finished = sorted(
        [m for m in matches if m.get("status") == "FINISHED"],
        key=lambda m: m.get("utcDate", ""),
    )
    for match in finished:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue
        if gh > ga:
            history[home].append("W"); history[away].append("L")
        elif gh < ga:
            history[home].append("L"); history[away].append("W")
        else:
            history[home].append("D"); history[away].append("D")
    return dict(history)

def build_team_stats(matches: list) -> dict:
    stats = defaultdict(lambda: {
        "home": {"gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0},
        "away": {"gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0},
    })
    for match in matches:
        if match.get("status") != "FINISHED":
            continue
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue
        stats[home]["home"]["gf"] += gh; stats[home]["home"]["ga"] += ga
        stats[away]["away"]["gf"] += ga; stats[away]["away"]["ga"] += gh
        if gh > ga:
            stats[home]["home"]["w"] += 1; stats[away]["away"]["l"] += 1
        elif gh < ga:
            stats[home]["home"]["l"] += 1; stats[away]["away"]["w"] += 1
        else:
            stats[home]["home"]["d"] += 1; stats[away]["away"]["d"] += 1
    return {k: dict(v) for k, v in stats.items()}

def build_team_momentum(matches: list, n: int = JANELA_MOMENTUM) -> dict:
    finished = sorted(
        [m for m in matches if m.get("status") == "FINISHED"],
        key=lambda m: m.get("utcDate", ""),
    )
    series = defaultdict(list)
    for match in finished:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue
        series[home].append({"gf": gh, "ga": ga})
        series[away].append({"gf": ga, "ga": gh})

    momentum = {}
    for team, jogos in series.items():
        if not jogos:
            continue
        all_gf = [j["gf"] for j in jogos]
        all_ga = [j["ga"] for j in jogos]
        media_gf_temp = sum(all_gf) / len(all_gf)
        media_ga_temp = sum(all_ga) / len(all_ga)

        ultimos_n    = jogos[-n:]
        media_gf_rec = sum(j["gf"] for j in ultimos_n) / len(ultimos_n)
        media_ga_rec = sum(j["ga"] for j in ultimos_n) / len(ultimos_n)

        momentum_of  = round(media_gf_rec / media_gf_temp, 3) if media_gf_temp > 0 else 1.0
        momentum_def = round(media_ga_rec / media_ga_temp, 3) if media_ga_temp > 0 else 1.0

        ultimos5   = jogos[-5:]
        offset     = 5 - len(ultimos5)
        forma_score = 0.0
        for i, j in enumerate(ultimos5):
            peso = PESOS_FORMA[i + offset]
            if j["gf"] > j["ga"]:   forma_score += peso * 1.0
            elif j["gf"] == j["ga"]: forma_score += peso * 0.5

        seq_invicto = 0
        for j in reversed(jogos):
            if j["gf"] >= j["ga"]: seq_invicto += 1
            else: break

        if seq_invicto > 0:
            sequencia = seq_invicto
        else:
            seq_sem_ganhar = 0
            for j in reversed(jogos):
                if j["gf"] < j["ga"]: seq_sem_ganhar -= 1
                else: break
            sequencia = seq_sem_ganhar

        momentum[team] = {
            "momentum_of":        momentum_of,
            "momentum_def":       momentum_def,
            "forma_score":        round(forma_score, 3),
            "sequencia":          sequencia,
            "media_gf_temporada": round(media_gf_temp, 2),
            "media_ga_temporada": round(media_ga_temp, 2),
            "media_gf_recente":   round(media_gf_rec, 2),
            "media_ga_recente":   round(media_ga_rec, 2),
        }
    return momentum

def build_tabela_csv(tabela, history, team_stats, momentum=None) -> pd.DataFrame:
    rows = []
    for entry in tabela:
        nome = entry.get("team", {}).get("name", "")
        w    = entry.get("won", 0)
        d    = entry.get("draw", 0)
        l    = entry.get("lost", 0)
        ts   = team_stats.get(nome, {})
        h    = ts.get("home", {}); a = ts.get("away", {})
        hist = history.get(nome, [])
        forma = ",".join(hist[-5:]) if hist else ""
        mom  = (momentum or {}).get(nome, {})
        rows.append({
            "posicao":            entry.get("position"),
            "time":               nome,
            "pts":                entry.get("points"),
            "j":                  entry.get("playedGames"),
            "v": w, "e": d, "d": l,
            "gp":                 entry.get("goalsFor"),
            "gc":                 entry.get("goalsAgainst"),
            "sg":                 entry.get("goalDifference"),
            "aprov_pct":          _aprov(w, d, l),
            "casa_v":  h.get("w", 0), "casa_e":  h.get("d", 0), "casa_d":  h.get("l", 0),
            "casa_gp": h.get("gf", 0), "casa_gc": h.get("ga", 0),
            "casa_aprov_pct":     _aprov(h.get("w", 0), h.get("d", 0), h.get("l", 0)),
            "fora_v":  a.get("w", 0), "fora_e":  a.get("d", 0), "fora_d":  a.get("l", 0),
            "fora_gp": a.get("gf", 0), "fora_gc": a.get("ga", 0),
            "fora_aprov_pct":     _aprov(a.get("w", 0), a.get("d", 0), a.get("l", 0)),
            "forma":              forma,
            "momentum_of":        mom.get("momentum_of"),
            "momentum_def":       mom.get("momentum_def"),
            "forma_score":        mom.get("forma_score"),
            "sequencia":          mom.get("sequencia"),
            "media_gf_temporada": mom.get("media_gf_temporada"),
            "media_ga_temporada": mom.get("media_ga_temporada"),
            "media_gf_recente":   mom.get("media_gf_recente"),
            "media_ga_recente":   mom.get("media_ga_recente"),
        })
    return pd.DataFrame(rows)

def get_brasileirao_data() -> tuple:
    if not FOOTBALL_DATA_KEY:
        print("  SKIP — FOOTBALL_DATA_KEY não definida")
        return {}, pd.DataFrame(), {}

    standings_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/standings")
    tables        = {t["type"]: t["table"] for t in standings_raw.get("standings", [])}
    tabela_total  = tables.get("TOTAL", [])

    matches_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/matches")
    matches     = matches_raw.get("matches", [])

    scorers_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/scorers", {"limit": 15})
    scorers     = scorers_raw.get("scorers", [])

    history    = build_team_history(matches)
    team_stats = build_team_stats(matches)
    momentum   = build_team_momentum(matches)
    df_tabela  = build_tabela_csv(tabela_total, history, team_stats, momentum)

    # Salva JSON bruto consolidado
    dados_brutos = {
        "tabela":          tabela_total,
        "artilheiros":     scorers,
        "historico_times": history,
        "team_stats":      team_stats,
    }
    salvar_raw_json("brasileirao", dados_brutos)

    # Salva CSV processado
    df_tabela.to_csv(CURRENT_DIR / "tabela.csv", index=False, encoding="utf-8-sig")
    print(f"  tabela.csv salvo — {len(df_tabela)} times")

    return dados_brutos, df_tabela, momentum

# ─────────────────────────────────────────────────────────────
# NORMALIZADORES
# ─────────────────────────────────────────────────────────────

def normalizar_mercado(raw: dict) -> pd.DataFrame:
    atletas  = raw.get("atletas", [])
    clubes   = {int(k): v.get("abreviacao", k) for k, v in raw.get("clubes",   {}).items()}
    posicoes = {int(k): v.get("nome",       k) for k, v in raw.get("posicoes", {}).items()}
    rows = []
    for a in atletas:
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id":     a.get("atleta_id"),
            "nome":          a.get("apelido", a.get("nome")),
            "clube_id":      a.get("clube_id"),
            "clube":         clubes.get(int(a.get("clube_id", 0)), a.get("clube_id")),
            "posicao_id":    a.get("posicao_id"),
            "posicao":       posicoes.get(a.get("posicao_id"), a.get("posicao_id")),
            "status_id":     a.get("status_id"),
            "preco":         a.get("preco_num"),
            "variacao":      a.get("variacao_num"),
            "media":         a.get("media_num"),
            "jogos":         a.get("jogos_num"),
            "pontos_rodada":   a.get("pontos_num"),
            "entrou_em_campo": a.get("entrou_em_campo", False),
            **{f"scout_{k}": v for k, v in scouts.items()},
        })
    return pd.DataFrame(rows)

def normalizar_pontuados(raw: dict) -> pd.DataFrame:
    atletas = raw.get("atletas", {})
    clubes  = {int(k): v.get("abreviacao", k) for k, v in raw.get("clubes", {}).items()}
    rows = []
    for atleta_id, a in atletas.items():
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id": int(atleta_id),
            "nome":      a.get("apelido", a.get("nome")),
            "clube_id":  a.get("clube_id"),
            "clube":     clubes.get(a.get("clube_id"), a.get("clube_id")),
            "posicao_id": a.get("posicao_id"),
            "pontos":    a.get("pontos_num"),
            "preco":     a.get("preco_num"),
            "variacao":  a.get("variacao_num"),
            **{f"scout_{k}": v for k, v in scouts.items()},
        })
    return pd.DataFrame(rows)

def normalizar_partidas(raw: dict) -> pd.DataFrame:
    lista = raw.get("partidas", raw) if isinstance(raw, dict) else raw
    if isinstance(lista, dict):
        lista = list(lista.values())
    return pd.json_normalize(lista) if lista else pd.DataFrame()

def normalizar_rodadas(raw: list) -> pd.DataFrame:
    return pd.json_normalize(raw) if raw else pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# ENRIQUECIMENTO CARTOLA
# ─────────────────────────────────────────────────────────────

def calcular_min_valorizar(preco: float) -> float:
    if preco <= 0:  return 0.0
    if preco <= 4:  return round(preco * 2.8, 1)
    if preco <= 8:  return round(preco * 2.3, 1)
    if preco <= 15: return round(preco * 2.0, 1)
    if preco <= 25: return round(preco * 1.8, 1)
    return round(preco * 1.6, 1)

def calcular_pb_media(row: pd.Series) -> float:
    j   = row["jogos"]
    med = row["media"]
    if j < 1 or med <= 0:
        return 0.0
    pos = row.get("posicao", "")
    g   = float(row.get("scout_G",  0) or 0)
    a   = float(row.get("scout_A",  0) or 0)
    sg  = float(row.get("scout_SG", 0) or 0)
    pts_evento_por_jogo = (
        (g  / j) * PESO_GOL_POS.get(pos, 5.0) +
        (a  / j) * PESO_ASSISTENCIA +
        (sg / j) * PESO_SG_POS.get(pos, 1.0)
    )
    return round(max(med - pts_evento_por_jogo, 0.0), 2)

def enriquecer(df_mercado: pd.DataFrame, df_partidas: pd.DataFrame) -> pd.DataFrame:
    df = df_mercado.copy()

    # ── Mandante / adversário ────────────────────────────────
    mapa_confronto = {}
    if not df_partidas.empty:
        col_casa = next((c for c in df_partidas.columns if "casa_id"      in c), None)
        col_vis  = next((c for c in df_partidas.columns if "visitante_id" in c), None)
        mapa_abr = {}
        for _, row in df_mercado.iterrows():
            if pd.notna(row.get("clube_id")) and pd.notna(row.get("clube")):
                mapa_abr[int(row["clube_id"])] = row["clube"]
        for _, p in df_partidas.iterrows():
            try:
                id_casa  = int(p[col_casa]) if col_casa else None
                id_vis   = int(p[col_vis])  if col_vis  else None
                abr_casa = mapa_abr.get(id_casa, str(id_casa))
                abr_vis  = mapa_abr.get(id_vis,  str(id_vis))
                if id_casa: mapa_confronto[id_casa] = {"mandante": True,  "adversario": abr_vis}
                if id_vis:  mapa_confronto[id_vis]  = {"mandante": False, "adversario": abr_casa}
            except Exception:
                continue

    def get_confronto(clube_id, campo, default):
        try:    return mapa_confronto.get(int(clube_id), {}).get(campo, default)
        except: return default

    df["mandante"]  = df["clube_id"].apply(lambda x: get_confronto(x, "mandante",  None))
    df["adversario"] = df["clube_id"].apply(lambda x: get_confronto(x, "adversario", "—"))

    # ── Tendência ────────────────────────────────────────────
    def calcular_tendencia(v):
        try:
            v = float(v)
            if v > 0.5:  return "alta"
            if v < -0.5: return "baixa"
            return "estavel"
        except: return "estavel"

    df["tendencia"] = df["variacao"].apply(calcular_tendencia)

    # ── Tipos numéricos ──────────────────────────────────────
    df["preco"]    = pd.to_numeric(df["preco"],    errors="coerce").fillna(0)
    df["media"]    = pd.to_numeric(df["media"],    errors="coerce").fillna(0)
    df["jogos"]    = pd.to_numeric(df["jogos"],    errors="coerce").fillna(0)
    df["variacao"] = pd.to_numeric(df["variacao"], errors="coerce").fillna(0)

    # ── Mínimo para valorizar ────────────────────────────────
    df["min_valorizar"] = df["preco"].apply(calcular_min_valorizar)

    # ── Pontos Base e Resiliência ────────────────────────────
    for scout_col in ["scout_G", "scout_A", "scout_SG"]:
        if scout_col in df.columns:
            df[scout_col] = pd.to_numeric(df[scout_col], errors="coerce").fillna(0)
        else:
            df[scout_col] = 0.0

    df["pb_media"]       = df.apply(calcular_pb_media, axis=1)
    df["resiliencia_pct"] = df.apply(
        lambda r: round(r["pb_media"] / r["media"], 3) if r["media"] > 0 else 0.0, axis=1
    )

    # ── Confiabilidade ───────────────────────────────────────
    df["confiabilidade"] = (df["jogos"] / JOGOS_CONFIANCA_PLENA).clip(upper=1.0).round(3)

    # ── Média bayesiana ──────────────────────────────────────
    media_prior = df[df["jogos"] >= 3].groupby("posicao")["media"].mean()

    def calcular_media_bayesiana(row):
        j = row["jogos"]
        if j < 1: return 0.0
        prior = media_prior.get(row["posicao"], row["media"])
        return round(
            (j * row["media"] + JOGOS_CONFIANCA_PLENA * prior) / (j + JOGOS_CONFIANCA_PLENA), 3
        )

    df["media_bayesiana"] = df.apply(calcular_media_bayesiana, axis=1)

    # ── Resíduo z-score por regressão linear ─────────────────
    residuos = []
    for pos, grp in df.groupby("posicao"):
        x = grp["preco"].values
        y = grp["media_bayesiana"].values
        if len(grp) < 3 or x.std() == 0:
            residuos.append(pd.Series(0.0, index=grp.index))
            continue
        coeffs = np.polyfit(x, y, 1)
        y_hat  = np.polyval(coeffs, x)
        resid  = y - y_hat
        std    = resid.std()
        z      = resid / std if std > 0 else np.zeros_like(resid)
        residuos.append(pd.Series(z.round(3), index=grp.index))

    df["residuo_z"] = pd.concat(residuos).reindex(df.index).fillna(0)

    def armadilha_label(z):
        if z < -1.5: return "armadilha_forte"
        if z < -0.5: return "armadilha_leve"
        if z > 1.5:  return "valor_oculto"
        if z > 0.5:  return "valor_bom"
        return "neutro"

    df["armadilha_label"] = df["residuo_z"].apply(armadilha_label)

    # ── Custo-benefício ──────────────────────────────────────
    df["custo_beneficio"] = df.apply(
        lambda r: round(r["media_bayesiana"] / r["preco"] * r["confiabilidade"], 3)
        if r["preco"] > 0 else 0, axis=1
    )
    df["cb_rank"] = (
        df[df["preco"] > 0]
        .groupby("posicao")["custo_beneficio"]
        .rank(ascending=False, method="min")
        .reindex(df.index).fillna(0).astype(int)
    )

    # ── Status legível ───────────────────────────────────────
    df["status_label"] = df["status_id"].apply(
        lambda x: STATUS_LABEL.get(int(x), "Desconhecido") if pd.notna(x) else "Desconhecido"
    )

    return df

def enriquecer_partidas(df_partidas, df_odds, mapa_clubes) -> pd.DataFrame:
    df = df_partidas.copy()
    if df_odds.empty:
        return df

    col_casa = next((c for c in df.columns if "casa_id"      in c), None)
    col_vis  = next((c for c in df.columns if "visitante_id" in c), None)
    if not col_casa or not col_vis:
        return df

    mapa_odds = {}
    for _, o in df_odds.iterrows():
        mapa_odds[o["abr_casa"]] = {
            "odd_casa":   o["odd_casa"],   "odd_vis":   o["odd_vis"],
            "odd_empate": o["odd_empate"], "prob_casa": o["prob_casa"],
            "prob_vis":   o["prob_vis"],   "forca_casa": o["forca_casa"],
            "forca_vis":  o["forca_vis"],
        }

    def get_odd(clube_id, campo):
        try:
            abr = mapa_clubes.get(int(clube_id))
            return mapa_odds.get(abr, {}).get(campo)
        except: return None

    df["odd_casa"]   = df[col_casa].apply(lambda x: get_odd(x, "odd_casa"))
    df["odd_vis"]    = df[col_vis].apply( lambda x: get_odd(x, "odd_vis"))
    df["odd_empate"] = df[col_casa].apply(lambda x: get_odd(x, "odd_empate"))
    df["prob_casa"]  = df[col_casa].apply(lambda x: get_odd(x, "prob_casa"))
    df["prob_vis"]   = df[col_vis].apply( lambda x: get_odd(x, "prob_vis"))
    df["forca_casa"] = df[col_casa].apply(lambda x: get_odd(x, "forca_casa"))
    df["forca_vis"]  = df[col_vis].apply( lambda x: get_odd(x, "forca_vis"))

    return df

# ─────────────────────────────────────────────────────────────
# ENRIQUECIMENTO COM DADOS DO BRASILEIRÃO
# ─────────────────────────────────────────────────────────────

def enriquecer_com_confronto(df, df_tabela, momentum) -> pd.DataFrame:
    if df_tabela.empty:
        return df

    df = df.copy()
    t  = df_tabela.copy()

    t["j_casa"] = t["casa_v"] + t["casa_e"] + t["casa_d"]
    t["j_fora"] = t["fora_v"] + t["fora_e"] + t["fora_d"]
    t["gc_pg_casa"]      = t.apply(lambda r: round(r["casa_gc"] / r["j_casa"], 3) if r["j_casa"] > 0 else 0, axis=1)
    t["gc_pg_fora"]      = t.apply(lambda r: round(r["fora_gc"] / r["j_fora"], 3) if r["j_fora"] > 0 else 0, axis=1)
    t["gp_pg_casa"]      = t.apply(lambda r: round(r["casa_gp"] / r["j_casa"], 3) if r["j_casa"] > 0 else 0, axis=1)
    t["gp_pg_fora"]      = t.apply(lambda r: round(r["fora_gp"] / r["j_fora"], 3) if r["j_fora"] > 0 else 0, axis=1)
    t["percentil_def_casa"] = t["gc_pg_casa"].rank(pct=True).round(3)
    t["percentil_def_fora"] = t["gc_pg_fora"].rank(pct=True).round(3)
    t["percentil_of_casa"]  = t["gp_pg_casa"].rank(pct=True).round(3)
    t["percentil_of_fora"]  = t["gp_pg_fora"].rank(pct=True).round(3)

    tabela_idx = t.set_index("time")
    results = []

    for _, atleta in df.iterrows():
        clube_abr = str(atleta.get("clube", ""))
        adv_abr   = str(atleta.get("adversario", ""))
        mandante  = atleta.get("mandante")
        posicao   = str(atleta.get("posicao", ""))

        row_time = get_tabela_row(clube_abr, tabela_idx)
        row_adv  = get_tabela_row(adv_abr,   tabela_idx)
        mom_time = get_momentum_time(clube_abr, momentum)
        mom_adv  = get_momentum_time(adv_abr,   momentum)

        rec = {
            "time_pos":              int(row_time["posicao"]) if row_time is not None else None,
            "adv_pos":               int(row_adv["posicao"])  if row_adv  is not None else None,
            "time_momentum_of":      mom_time.get("momentum_of"),
            "time_momentum_def":     mom_time.get("momentum_def"),
            "adv_momentum_of":       mom_adv.get("momentum_of"),
            "adv_momentum_def":      mom_adv.get("momentum_def"),
            "sequencia_time":        mom_time.get("sequencia"),
            "forma_score_time":      mom_time.get("forma_score"),
            "vantagem_mando":        None,
            "oportunidade_confronto": None,
        }

        if row_time is not None and row_adv is not None and mandante is not None:
            if mandante:
                rec["vantagem_mando"] = round(float(row_time["casa_aprov_pct"]) - float(row_adv["fora_aprov_pct"]), 1)
            else:
                rec["vantagem_mando"] = round(float(row_time["fora_aprov_pct"]) - float(row_adv["casa_aprov_pct"]), 1)

        if row_adv is not None and mandante is not None:
            if posicao in POSICOES_ATAQUE:
                rec["oportunidade_confronto"] = round(float(
                    row_adv["percentil_def_fora"] if mandante else row_adv["percentil_def_casa"]
                ), 3)
            elif posicao in POSICOES_DEFESA:
                rec["oportunidade_confronto"] = round(1.0 - float(
                    row_adv["percentil_of_fora"] if mandante else row_adv["percentil_of_casa"]
                ), 3)

        results.append(rec)

    for col in results[0].keys():
        df[col] = [r[col] for r in results]

    # ── Score composto dinâmico por posição ──────────────────
    oc       = df["oportunidade_confronto"].fillna(0.5)
    vm       = ((df["vantagem_mando"].fillna(0).clip(-50, 50) + 50) / 100)
    tof_norm = ((df["time_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)
    fs       = df["forma_score_time"].fillna(0.5)
    adv_norm = ((df["adv_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)

    def calcular_score(row):
        if row["posicao"] in POSICOES_DEFESA:
            return (0.40 * row["oc"] + 0.30 * row["vm"] + 0.10 * row["tof_norm"]
                    + 0.10 * row["fs"] + 0.10 * (1 - row["adv_norm"]))
        else:
            return (0.35 * row["oc"] + 0.15 * row["vm"] + 0.30 * row["tof_norm"]
                    + 0.15 * row["fs"] + 0.05 * (1 - row["adv_norm"]))

    df_temp = pd.DataFrame({
        "posicao": df["posicao"],
        "oc": oc, "vm": vm, "tof_norm": tof_norm, "fs": fs, "adv_norm": adv_norm,
    })
    df["score_confronto"] = df_temp.apply(calcular_score, axis=1).round(4)

    mask_validos = (
        df["adversario"].notna() &
        (df["adversario"] != "—") &
        (df["adversario"] != "")
    )

    def z_score_seguro(x):
        std = x.std()
        if pd.isna(std) or std < 1e-6:
            return pd.Series(0.0, index=x.index)
        return ((x - x.mean()) / std).clip(-3.0, 3.0)

    df["score_confronto_z"] = np.nan
    df.loc[mask_validos, "score_confronto_z"] = (
        df[mask_validos]
        .groupby("posicao")["score_confronto"]
        .transform(z_score_seguro)
        .round(3)
        .clip(lower=-3.0, upper=3.0)
    )

    df["score_confronto_100"] = (50 + (df["score_confronto_z"] * 15)).round(1).clip(0, 100)

    return df

# ─────────────────────────────────────────────────────────────
# SNAPSHOT HISTÓRICO
# ─────────────────────────────────────────────────────────────

# Colunas salvas nos snapshots históricos
# Inclui entrou_em_campo para calcular taxa de participação futuramente
COLUNAS_SNAPSHOT = [
    "atleta_id", "nome", "clube", "posicao", "status_id",
    "preco", "variacao", "media", "jogos", "pontos_rodada",
    "entrou_em_campo",                          # ← base para taxa de participação
    "mandante", "adversario",
    "residuo_z", "armadilha_label",
    "media_bayesiana", "pb_media", "confiabilidade",
    "score_confronto_100",
]

def _pasta_rodada(rodada: int) -> Path:
    pasta = HISTORICO_DIR / f"r{rodada:02d}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta

def _colunas_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas as colunas relevantes para o snapshot."""
    cols = [c for c in COLUNAS_SNAPSHOT if c in df.columns]
    return df[cols].copy()

def salvar_snapshot_pre(df_atletas: pd.DataFrame, df_partidas: pd.DataFrame,
                        rodada: int):
    """
    Snapshot PRÉ-rodada — salvo uma única vez, dentro de 2h do fechamento.
    Captura o preço inicial antes de qualquer valorização.
    """
    pasta = _pasta_rodada(rodada)
    path  = pasta / "atletas_pre.csv"

    if path.exists():
        print(f"  [PRÉ] já existe — rodada {rodada}, pulando")
        return

    _colunas_snapshot(df_atletas).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [PRÉ] salvo — rodada {rodada} ({len(df_atletas)} atletas)")

    # Partidas da rodada — salva junto com o PRÉ, só uma vez
    path_partidas = pasta / "partidas.csv"
    if not path_partidas.exists() and not df_partidas.empty:
        df_partidas.to_csv(path_partidas, index=False, encoding="utf-8-sig")
        print(f"  [PRÉ] partidas salvas — rodada {rodada}")

def salvar_snapshot_pos(df_atletas: pd.DataFrame, rodada: int):
    """
    Snapshot PÓS-rodada — salvo uma única vez quando o mercado reabre.
    Usa os dados atuais — preços já estabilizados após a rodada.
    """
    pasta = _pasta_rodada(rodada)
    path  = pasta / "atletas_pos.csv"

    if path.exists():
        print(f"  [PÓS] já existe — rodada {rodada}, pulando")
        return

    _colunas_snapshot(df_atletas).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [PÓS] salvo — rodada {rodada} ({len(df_atletas)} atletas)")

def gerenciar_snapshots(df_atletas: pd.DataFrame, df_partidas: pd.DataFrame,
                        raw_status: dict):
    """
    Orquestra os snapshots baseado no status do mercado.

    Fluxo:
    ┌─────────────────────────────────────────────────────────┐
    │  mercado aberto + 2h antes fechamento                   │
    │  → salva PRÉ da rodada atual (uma vez)                  │
    │                                                         │
    │  mercado aberto + rodada incrementou                    │
    │  → salva PÓS da rodada anterior com dados atuais        │
    │    (preços já estabilizados após os jogos)              │
    └─────────────────────────────────────────────────────────┘
    """
    rodada_atual   = int(raw_status.get("rodada_atual", 0))
    status_mercado = int(raw_status.get("status_mercado", 0))
    fechamento_ts  = raw_status.get("fechamento", {}).get("timestamp", 0)
    agora_ts       = datetime.now(timezone.utc).timestamp()

    if rodada_atual == 0:
        print("  [SNAPSHOT] rodada_atual = 0, pulando")
        return

    # Só age quando mercado está aberto
    if status_mercado != 1:
        print("  [SNAPSHOT] mercado fechado — aguardando reabertura")
        return

    # Janela de 2h antes do fechamento → PRÉ da rodada atual
    dentro_janela_pre = agora_ts > (fechamento_ts - 7200)
    if dentro_janela_pre:
        salvar_snapshot_pre(df_atletas, df_partidas, rodada_atual)

    # Rodada anterior existe e não tem PÓS → salva com dados atuais
    rodada_anterior = rodada_atual - 1
    if rodada_anterior > 0:
        pasta_anterior = _pasta_rodada(rodada_anterior)
        pre_anterior   = pasta_anterior / "atletas_pre.csv"
        pos_anterior   = pasta_anterior / "atletas_pos.csv"

        if pre_anterior.exists() and not pos_anterior.exists():
            salvar_snapshot_pos(df_atletas, rodada_anterior)

# ─────────────────────────────────────────────────────────────
# GERAÇÃO DO llm/input/
# ─────────────────────────────────────────────────────────────

def gerar_llm_input(df_atletas: pd.DataFrame, df_partidas: pd.DataFrame,
                    df_tabela: pd.DataFrame, raw_status: dict,
                    df_rodadas: pd.DataFrame, df_odds: pd.DataFrame):
    """
    Gera os CSVs pré-filtrados para o modelo de linguagem em llm/input/.

    Filtros aplicados nos atletas:
    - status_id = 7 (Provável)
    - preco > 0
    - adversario definido (não '—')
    - armadilha_label != 'armadilha_forte'
    - apenas as colunas que o prompt usa
    """

    # Atletas — filtrado e com colunas reduzidas
    colunas_disponiveis = [c for c in COLUNAS_LLM if c in df_atletas.columns]
    df_llm = df_atletas[
        (df_atletas["status_id"] == 7) &
        (df_atletas["preco"] > 0) &
        (df_atletas["adversario"] != "—") &
        (df_atletas["armadilha_label"] != "armadilha_forte")
    ][colunas_disponiveis].copy()

    # Ordena por posição e residuo_z desc — facilita leitura do modelo
    df_llm = df_llm.sort_values(["posicao", "residuo_z"], ascending=[True, False])
    df_llm.to_csv(LLM_INPUT_DIR / "atletas.csv", index=False, encoding="utf-8-sig")
    print(f"  llm/atletas.csv — {len(df_llm)} atletas elegíveis")

    # Partidas
    if not df_partidas.empty:
        df_partidas.to_csv(LLM_INPUT_DIR / "partidas.csv", index=False, encoding="utf-8-sig")

    # Tabela
    if not df_tabela.empty:
        df_tabela.to_csv(LLM_INPUT_DIR / "tabela.csv", index=False, encoding="utf-8-sig")

    # Status
    pd.DataFrame([raw_status]).to_csv(LLM_INPUT_DIR / "status.csv", index=False, encoding="utf-8-sig")

    # Rodadas
    if not df_rodadas.empty:
        df_rodadas.to_csv(LLM_INPUT_DIR / "rodadas.csv", index=False, encoding="utf-8-sig")

    # Odds
    if not df_odds.empty:
        df_odds.to_csv(LLM_INPUT_DIR / "odds.csv", index=False, encoding="utf-8-sig")

    print(f"  llm/input/ atualizado — {len(df_llm)} atletas, rodada {raw_status.get('rodada_atual')}")

# ─────────────────────────────────────────────────────────────
# SCORE POR TIME DA RODADA
# ─────────────────────────────────────────────────────────────

def normalizar_serie(s: pd.Series) -> pd.Series:
    """Normaliza uma série para 0-1. Retorna 0.5 se todos os valores forem iguais."""
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-6:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def gerar_times_rodada(df_atletas: pd.DataFrame, df_partidas: pd.DataFrame,
                       df_tabela: pd.DataFrame, df_odds: pd.DataFrame,
                       momentum: dict = None) -> pd.DataFrame:
    """
    Gera times_rodada.csv — visão consolidada por time para a rodada atual.

    Cruza três fontes:
        odds            → prob_vitoria, forca
        brasileirao     → momentum, sequencia, fragilidade defensiva do adversário
        atletas (st=7)  → qualidade e disponibilidade do plantel

    Score composto (0-1):
        0.25 × prob_vitoria          (favoritismo)
        0.20 × momentum_of           (momento ofensivo)
        0.20 × adv_fragilidade       (quão vulnerável é o adversário)
        0.15 × media_residuo_z       (qualidade do plantel disponível)
        0.10 × concentracao_valor    (% de valor_oculto + valor_bom no plantel)
        0.10 × (1 - preco_medio)     (acessibilidade — plantel barato = mais CPP)

    Tier:
        >= 0.70 → A  (minerar sem hesitar)
        >= 0.55 → B  (boas opções, avaliar individualmente)
        >= 0.40 → C  (neutro)
        <  0.40 → D  (evitar como estratégia de time)
    """
    if df_partidas.empty or df_atletas.empty:
        print("  [TIMES] partidas ou atletas vazios — pulando")
        return pd.DataFrame()

    # ── Identificar times e confrontos da rodada ──────────────
    col_casa = next((c for c in df_partidas.columns if "casa_id"      in c), None)
    col_vis  = next((c for c in df_partidas.columns if "visitante_id" in c), None)
    if not col_casa or not col_vis:
        print("  [TIMES] colunas de partida não encontradas — pulando")
        return pd.DataFrame()

    # Mapa clube_id → abreviacao
    mapa_abr = {}
    for _, row in df_atletas.iterrows():
        if pd.notna(row.get("clube_id")) and pd.notna(row.get("clube")):
            mapa_abr[int(row["clube_id"])] = row["clube"]

    # Monta lista de confrontos
    confrontos = []
    for _, p in df_partidas.iterrows():
        try:
            id_casa = int(p[col_casa])
            id_vis  = int(p[col_vis])
            abr_casa = mapa_abr.get(id_casa)
            abr_vis  = mapa_abr.get(id_vis)
            if not abr_casa or not abr_vis:
                continue
            confrontos.append({
                "time":      abr_casa,
                "adversario": abr_vis,
                "mandante":  True,
                "prob_vitoria": float(p.get("prob_casa") or 0),
                "forca":     p.get("forca_casa", ""),
            })
            confrontos.append({
                "time":      abr_vis,
                "adversario": abr_casa,
                "mandante":  False,
                "prob_vitoria": float(p.get("prob_vis") or 0),
                "forca":     p.get("forca_vis", ""),
            })
        except Exception:
            continue

    if not confrontos:
        print("  [TIMES] nenhum confronto mapeado — pulando")
        return pd.DataFrame()

    df_conf = pd.DataFrame(confrontos)

    # ── Fragilidade defensiva do adversário ───────────────────
    adv_fragilidade = {}
    if not df_tabela.empty:
        t = df_tabela.copy()
        t["j_casa"] = t["casa_v"] + t["casa_e"] + t["casa_d"]
        t["j_fora"] = t["fora_v"] + t["fora_e"] + t["fora_d"]
        t["gc_pg_casa"] = t.apply(
            lambda r: r["casa_gc"] / r["j_casa"] if r["j_casa"] > 0 else 0, axis=1)
        t["gc_pg_fora"] = t.apply(
            lambda r: r["fora_gc"] / r["j_fora"] if r["j_fora"] > 0 else 0, axis=1)
        tabela_frag = t.set_index("time")

        for _, conf in df_conf.iterrows():
            adv_abr  = conf["adversario"]
            mandante = conf["mandante"]
            # Usa get_tabela_row que faz mapeamento abr → nome completo
            row_adv  = get_tabela_row(adv_abr, tabela_frag)
            if row_adv is not None:
                gc_pg = float(row_adv["gc_pg_fora"] if mandante else row_adv["gc_pg_casa"])
                adv_fragilidade[conf["time"]] = gc_pg
            else:
                # Fallback: usa momentum do adversário se tabela falhar
                m_adv = get_momentum_time(adv_abr, momentum or {})
                adv_fragilidade[conf["time"]] = m_adv.get("media_ga_recente", 1.0) if m_adv else 1.0

    df_conf["adv_gc_pg"] = df_conf["time"].map(adv_fragilidade).fillna(0)

    # ── Métricas do plantel (atletas status_id=7) ─────────────
    provaveis = df_atletas[
        (df_atletas["status_id"] == 7) &
        (df_atletas["preco"] > 0)
    ].copy()

    plantel_stats = []
    for time in df_conf["time"].unique():
        pl = provaveis[provaveis["clube"] == time]
        if pl.empty:
            plantel_stats.append({
                "time": time, "n_provaveis": 0,
                "n_valor_oculto": 0, "n_valor_bom": 0,
                "media_residuo_z": 0, "media_pb": 0, "preco_medio": 0,
                "concentracao_valor": 0,
            })
            continue

        n_prov   = len(pl)
        n_oculto = (pl["armadilha_label"] == "valor_oculto").sum()
        n_bom    = (pl["armadilha_label"] == "valor_bom").sum()
        conc     = (n_oculto + n_bom) / n_prov if n_prov > 0 else 0

        plantel_stats.append({
            "time":               time,
            "n_provaveis":        n_prov,
            "n_valor_oculto":     int(n_oculto),
            "n_valor_bom":        int(n_bom),
            "media_residuo_z":    round(pl["residuo_z"].mean(), 3),
            "media_pb":           round(pl["pb_media"].mean(), 3)
                                  if "pb_media" in pl.columns else 0,
            "preco_medio":        round(pl["preco"].mean(), 2),
            "concentracao_valor": round(conc, 3),
        })

    df_plantel = pd.DataFrame(plantel_stats)
    df_times   = df_conf.merge(df_plantel, on="time", how="left")

    # ── Momentum do time ──────────────────────────────────────
    mom_of_list  = []
    mom_def_list = []
    seq_list     = []
    for abr in df_times["time"]:
        m = get_momentum_time(abr, momentum or {})
        mom_of_list.append(float(m.get("momentum_of",  1.0)) if m else 1.0)
        mom_def_list.append(float(m.get("momentum_def", 1.0)) if m else 1.0)
        seq_list.append(int(m.get("sequencia", 0)) if m else 0)

    df_times["momentum_of"]  = pd.Series(mom_of_list,  index=df_times.index, dtype=float).fillna(1.0)
    df_times["momentum_def"] = pd.Series(mom_def_list, index=df_times.index, dtype=float).fillna(1.0)
    df_times["sequencia"]    = pd.Series(seq_list,     index=df_times.index, dtype=float).fillna(0)

    # ── Score composto normalizado ────────────────────────────
    df_times["prob_vitoria"]       = pd.to_numeric(df_times["prob_vitoria"],       errors="coerce").fillna(0)
    df_times["adv_gc_pg"]          = pd.to_numeric(df_times["adv_gc_pg"],          errors="coerce").fillna(0)
    df_times["media_residuo_z"]    = pd.to_numeric(df_times["media_residuo_z"],    errors="coerce").fillna(0)
    df_times["concentracao_valor"] = pd.to_numeric(df_times["concentracao_valor"], errors="coerce").fillna(0)
    df_times["preco_medio"]        = pd.to_numeric(df_times["preco_medio"],        errors="coerce").fillna(0)
    df_times["momentum_of"]        = pd.to_numeric(df_times["momentum_of"],        errors="coerce").fillna(1.0)

    n_prob   = normalizar_serie(df_times["prob_vitoria"])
    n_mom    = normalizar_serie(df_times["momentum_of"])
    n_adv    = normalizar_serie(df_times["adv_gc_pg"])
    n_resid  = normalizar_serie(df_times["media_residuo_z"])
    n_conc   = normalizar_serie(df_times["concentracao_valor"])
    n_preco  = normalizar_serie(df_times["preco_medio"])

    df_times["score_time"] = (
        0.25 * n_prob  +
        0.20 * n_mom   +
        0.20 * n_adv   +
        0.15 * n_resid +
        0.10 * n_conc  +
        0.10 * (1 - n_preco)
    ).round(3)

    def tier(s):
        if s >= 0.70: return "A"
        if s >= 0.55: return "B"
        if s >= 0.40: return "C"
        return "D"

    df_times["tier_time"] = df_times["score_time"].apply(tier)

    # Ordena por score desc
    df_times = df_times.sort_values("score_time", ascending=False).reset_index(drop=True)

    # ── Salva em current/ e llm/input/ ───────────────────────
    df_times.to_csv(CURRENT_DIR   / "times_rodada.csv", index=False, encoding="utf-8-sig")
    df_times.to_csv(LLM_INPUT_DIR / "times_rodada.csv", index=False, encoding="utf-8-sig")
    print(f"  times_rodada.csv — {len(df_times)} times | tiers: "
          f"A={( df_times['tier_time']=='A').sum()} "
          f"B={(df_times['tier_time']=='B').sum()} "
          f"C={(df_times['tier_time']=='C').sum()} "
          f"D={(df_times['tier_time']=='D').sum()}")

    return df_times

# ─────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────

log = []

# ── 1. Coleta e normalização dos endpoints Cartola ───────────
EXTRATORES = {
    "mercado":   (normalizar_mercado,   "mercado"),
    "pontuados": (normalizar_pontuados, "atletas_pontuados"),
    "partidas":  (normalizar_partidas,  "partidas"),
    "rodadas":   (normalizar_rodadas,   "rodadas"),
}

dados_brutos = {}
for key, (normalizador, nome_arquivo) in EXTRATORES.items():
    print(f"Extraindo {key}...")
    try:
        raw = get_json(key)
        salvar_raw_json(nome_arquivo, raw)                          # → docs/data/raw/
        df  = normalizador(raw)
        df.to_csv(CURRENT_DIR / f"{nome_arquivo}.csv",             # → docs/data/current/
                  index=False, encoding="utf-8-sig")
        dados_brutos[key] = df
        print(f"  OK — {len(df)} registros")
        log.append({"endpoint": key, "registros": len(df), "status": "OK", "erro": ""})
    except Exception as e:
        print(f"  ERRO: {e}")
        log.append({"endpoint": key, "registros": 0, "status": "ERRO", "erro": str(e)})
        dados_brutos[key] = pd.DataFrame()

# ── 2. Mercado status ────────────────────────────────────────
print("Extraindo status do mercado...")
raw_status = {}
try:
    raw_status = get_json("status")
    salvar_raw_json("mercado_status", raw_status)                   # → docs/data/raw/
    pd.DataFrame([raw_status]).to_csv(                             # → docs/data/current/
        CURRENT_DIR / "status.csv", index=False, encoding="utf-8-sig")
    print("  OK")
    log.append({"endpoint": "status", "registros": 1, "status": "OK", "erro": ""})
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "status", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── 3. Odds ──────────────────────────────────────────────────
print("Extraindo odds...")
df_odds = pd.DataFrame()
try:
    df_odds = get_odds()
    if not df_odds.empty:
        df_odds.to_csv(CURRENT_DIR / "odds.csv", index=False, encoding="utf-8-sig")
        print(f"  OK — {len(df_odds)} jogos")
        log.append({"endpoint": "odds", "registros": len(df_odds), "status": "OK", "erro": ""})
    else:
        print("  SKIP — sem dados de odds")
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "odds", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── 4. Partidas enriquecidas com odds ────────────────────────
print("Enriquecendo partidas com odds...")
df_mercado  = dados_brutos.get("mercado",  pd.DataFrame())
df_partidas = dados_brutos.get("partidas", pd.DataFrame())
mapa_clubes = {}
if not df_mercado.empty:
    for _, row in df_mercado.iterrows():
        if pd.notna(row.get("clube_id")) and pd.notna(row.get("clube")):
            mapa_clubes[int(row["clube_id"])] = row["clube"]

try:
    if not df_partidas.empty and not df_odds.empty:
        df_partidas = enriquecer_partidas(df_partidas, df_odds, mapa_clubes)
        df_partidas.to_csv(CURRENT_DIR / "partidas.csv", index=False, encoding="utf-8-sig")
        print(f"  OK — odds cruzadas em {len(df_partidas)} partidas")
    else:
        print("  SKIP — partidas ou odds vazias")
except Exception as e:
    print(f"  ERRO: {e}")

# ── 5. Brasileirão ───────────────────────────────────────────
print("Extraindo tabela do Brasileirão...")
df_tabela   = pd.DataFrame()
momentum    = {}
try:
    _, df_tabela, momentum = get_brasileirao_data()
    log.append({"endpoint": "brasileirao", "registros": len(df_tabela), "status": "OK", "erro": ""})
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "brasileirao", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── 6. Enriquecimento dos atletas ────────────────────────────
print("Gerando atletas enriquecidos...")
df_atletas_enriquecido = pd.DataFrame()
try:
    if not df_mercado.empty:
        df_enr = enriquecer(df_mercado, df_partidas)
        if not df_tabela.empty:
            df_enr = enriquecer_com_confronto(df_enr, df_tabela, momentum)
        df_atletas_enriquecido = df_enr

        # current/atletas.csv — só status relevantes (exclui status=6 Nulo)
        df_current = df_enr[df_enr["status_id"].isin([7, 2])].copy()
        df_current.to_csv(CURRENT_DIR / "atletas.csv", index=False, encoding="utf-8-sig")

        print(f"  OK — {len(df_enr)} total | {len(df_current)} no current/ (excl. status=6)")
        log.append({"endpoint": "atletas_enriquecido", "registros": len(df_current), "status": "OK", "erro": ""})
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "atletas_enriquecido", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── 7. Snapshots históricos ──────────────────────────────────
print("Gerenciando snapshots históricos...")
try:
    if not df_atletas_enriquecido.empty and raw_status:
        gerenciar_snapshots(df_atletas_enriquecido, df_partidas, raw_status)
except Exception as e:
    print(f"  ERRO nos snapshots: {e}")

# ── 8. Geração do llm/input/ ─────────────────────────────────
print("Gerando llm/input/...")
try:
    if not df_atletas_enriquecido.empty:
        df_rodadas = dados_brutos.get("rodadas", pd.DataFrame())
        gerar_llm_input(
            df_atletas_enriquecido, df_partidas, df_tabela,
            raw_status, df_rodadas, df_odds
        )
except Exception as e:
    print(f"  ERRO no llm/input/: {e}")

# ── 9. Score por time da rodada ──────────────────────────────
print("Gerando times_rodada.csv...")
try:
    if not df_atletas_enriquecido.empty and not df_partidas.empty:
        gerar_times_rodada(
            df_atletas_enriquecido, df_partidas, df_tabela, df_odds, momentum
        )
        log.append({"endpoint": "times_rodada", "registros": 1, "status": "OK", "erro": ""})
except Exception as e:
    print(f"  ERRO no times_rodada: {e}")
    log.append({"endpoint": "times_rodada", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── 10. Log de execução ──────────────────────────────────────
df_log = pd.DataFrame(log)
df_log["timestamp"] = datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT")
df_log.to_csv(CURRENT_DIR / "log.csv", index=False, encoding="utf-8-sig")
print(f"\nConcluído — {len(log)} endpoints processados")