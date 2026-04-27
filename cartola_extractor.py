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
                          Defesa: mistura oc, vm, tof, tdef (mom. defensivo próprio),
                                  fs e (1-adv_of) (adv fraco no ataque).
                          Ataque: mistura oc, vm, tof, fs e adv_def (defesa do adv
                                  sofrendo mais gols recentemente).
                          Pesos em PESOS_SCORE_DEFAULT; sobrescritos por
                          calibracao_score.json quando calibrado empiricamente.
    condicao_mando      → 'favoravel' | 'favoravel_visitante' | 'neutro' | 'desfavoravel'
                          favoravel: mandante + vantagem_mando>5 + sc>60 + rz>0
                          favoravel_visitante: visitante + sc>70 + rz>0 (ex: 1º vs último fora de casa)
                          (proxy enquanto histórico individual acumula; substituir por delta_mando_jogador quando disponível)
    pontos_esperados    → media_bayesiana × (score_confronto_100/50) × confiabilidade — retorno absoluto esperado na rodada
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

# Pesos default do score_confronto. Somam 1.0 por grupo.
# Sobrescritos por docs/data/current/calibracao_score.json quando disponível
# (ver calibrar_score.py).
#
# Grupo defesa: Goleiro/Zagueiro/Lateral. Inclui tdef (momentum defensivo
# próprio) e adv_of (adv ofensivo — usado invertido como 1-adv_of_norm).
# Grupo ataque: Atacante/Meia/Técnico. Usa adv_def (fragilidade defensiva
# recente do adv, não mais adv_of que não tem sentido lógico aqui).
PESOS_SCORE_DEFAULT = {
    "defesa": {
        "oc":       0.30,  # fragilidade ofensiva do adv (percentil posicional)
        "vm":       0.25,  # vantagem de aproveitamento no mando
        "tof":      0.10,  # momentum ofensivo próprio (time ganhando segura SG)
        "tdef":     0.15,  # momentum defensivo próprio (1 - ratio, invertido)
        "fs":       0.05,  # forma recente (V/E/D)
        "adv_of":   0.10,  # 1 - adv_momentum_of (adv fraco no ataque = bom)
        "prob_gols": 0.05, # prob over 2.5 invertida (jogo fechado = bom p/ defesa)
    },
    "ataque": {
        "oc":       0.25,  # fragilidade defensiva do adv (percentil posicional)
        "vm":       0.15,  # vantagem de aproveitamento no mando
        "tof":      0.25,  # momentum ofensivo próprio
        "fs":       0.10,  # forma recente (V/E/D)
        "adv_def":  0.15,  # adv_momentum_def (defesa adv sofrendo mais)
        "prob_gols": 0.10, # prob over 2.5 (jogo aberto = mais scouts p/ atacantes)
    },
}

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

# Colunas que vão para o llm/input/ (apenas o que o modelo precisa).
# Enxutas: removidos atleta_id (ruído pra LLM), score_confronto_z
# (redundante com _100), forma_score_time (já capturado em tier/score_time de
# times_rodada.csv). Adicionadas recomendacao + caro_e_vale para que a LLM não
# precise recalcular heurísticas.
COLUNAS_LLM = [
    "nome", "clube", "posicao", "status_label",
    "preco", "variacao", "media", "jogos",
    "mandante", "adversario", "tendencia", "prob_vitoria",
    "min_valorizar", "pb_media", "resiliencia_pct", "confiabilidade",
    "media_bayesiana", "residuo_z", "armadilha_label",
    "custo_beneficio", "cb_rank",
    "oportunidade_confronto", "vantagem_mando",
    "score_confronto_100",
    "condicao_mando", "pontos_esperados",
    "caro_e_vale", "recomendacao",
    "time_momentum_of", "time_momentum_def",
    "adv_momentum_of", "adv_momentum_def",
    "sequencia_time",
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
        "markets":     "h2h,totals",
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
        odd_over_25 = odd_under_25 = None
        for bm in jogo.get("bookmakers", [])[:1]:
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        abr = NOMES_PARA_ABR.get(o["name"])
                        if abr == abr_casa:   odd_casa   = o["price"]
                        elif abr == abr_vis:  odd_vis    = o["price"]
                        else:                 odd_empate = o["price"]
                elif market["key"] == "totals":
                    for o in market["outcomes"]:
                        if abs(float(o.get("point", 0)) - 2.5) < 0.01:
                            if o["name"] == "Over":  odd_over_25  = o["price"]
                            if o["name"] == "Under": odd_under_25 = o["price"]

        if not odd_casa or not odd_vis:
            continue

        soma      = (1/odd_casa) + (1/odd_vis) + (1/odd_empate if odd_empate else 0)
        prob_casa = round((1/odd_casa) / soma, 3) if soma else None
        prob_vis  = round((1/odd_vis)  / soma, 3) if soma else None

        # Probabilidade implícita de over 2.5 gols (remove vig)
        prob_over_25 = None
        if odd_over_25 and odd_under_25:
            soma_gols = (1/odd_over_25) + (1/odd_under_25)
            prob_over_25 = round((1/odd_over_25) / soma_gols, 3)

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
            "prob_over_25":  prob_over_25,
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
        print("  [WARN] normalizar_partidas: 'partidas' não encontrado no raw — usando dict completo")
        lista = list(lista.values())
    return pd.json_normalize(lista) if lista else pd.DataFrame()

def normalizar_rodadas(raw: list) -> pd.DataFrame:
    return pd.json_normalize(raw) if raw else pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# ENRIQUECIMENTO CARTOLA
# ─────────────────────────────────────────────────────────────

def calcular_min_valorizar(row) -> float:
    """
    Estimativa de pontos mínimos para valorizar.
    Usa pontos_rodada (última pontuação) como proxy — não é o MPV
    real calculado pelo Cartola (que depende da variação de preço
    esperada pela plataforma).
    Fallback: média histórica quando não há pontuação recente.
    """
    pontos = float(row.get("pontos_rodada") or 0)
    if pontos > 0:
        return round(pontos, 1)
    
    # Fallback quando não tem pontuação recente
    media = float(row.get("media") or 0)
    if media > 0:
        return round(media, 1)
    
    return 0.0

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
                id_casa   = int(p[col_casa]) if col_casa else None
                id_vis    = int(p[col_vis])  if col_vis  else None
                abr_casa  = mapa_abr.get(id_casa, str(id_casa))
                abr_vis   = mapa_abr.get(id_vis,  str(id_vis))
                prob_casa  = float(p.get("prob_casa")   or 0)
                prob_vis   = float(p.get("prob_vis")    or 0)
                prob_gols  = p.get("prob_over_25")
                prob_gols  = float(prob_gols) if prob_gols is not None else None
                if id_casa: mapa_confronto[id_casa] = {
                    "mandante": True,  "adversario": abr_vis,  "prob_vitoria": prob_casa,
                    "prob_gols": prob_gols}
                if id_vis:  mapa_confronto[id_vis]  = {
                    "mandante": False, "adversario": abr_casa, "prob_vitoria": prob_vis,
                    "prob_gols": prob_gols}
            except Exception:
                continue

    def get_confronto(clube_id, campo, default):
        try:    return mapa_confronto.get(int(clube_id), {}).get(campo, default)
        except: return default

    df["mandante"]     = df["clube_id"].apply(lambda x: get_confronto(x, "mandante",     None))
    df["adversario"]   = df["clube_id"].apply(lambda x: get_confronto(x, "adversario",   "—"))
    df["prob_vitoria"] = df["clube_id"].apply(lambda x: get_confronto(x, "prob_vitoria", 0.0))
    df["prob_gols"]    = df["clube_id"].apply(lambda x: get_confronto(x, "prob_gols",    None))

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
    df["min_valorizar"] = df.apply(calcular_min_valorizar, axis=1)


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

    # ── Média bayesiana com prior estratificado por faixa de preço ──
    # Prior estratificado: jogadores do mesmo nível de preço na posição
    # são um peer group mais justo do que toda a posição misturada.
    def _calcular_faixa(series: pd.Series) -> pd.Series:
        try:
            return pd.qcut(
                series, q=3, labels=["baixo", "medio", "alto"], duplicates="drop"
            ).astype(str)
        except Exception:
            return pd.Series("medio", index=series.index)

    df["_faixa_preco"] = df.groupby("posicao")["preco"].transform(_calcular_faixa)

    elegiveis = df[df["jogos"] >= 3]
    prior_estratificado = (
        elegiveis.groupby(["posicao", "_faixa_preco"])["media"].mean().to_dict()
    )
    prior_posicao = elegiveis.groupby("posicao")["media"].mean().to_dict()

    def calcular_media_bayesiana(row):
        j = row["jogos"]
        if j < 1: return 0.0
        prior = (
            prior_estratificado.get((row["posicao"], row["_faixa_preco"]))
            or prior_posicao.get(row["posicao"])
            or row["media"]
        )
        return round(
            (j * row["media"] + JOGOS_CONFIANCA_PLENA * prior) / (j + JOGOS_CONFIANCA_PLENA), 3
        )

    df["media_bayesiana"] = df.apply(calcular_media_bayesiana, axis=1)
    df.drop(columns=["_faixa_preco"], inplace=True)

    # ── Resíduo z-score por regressão log-linear ──────────────
    # log(preco) lineariza melhor a relação preço→média porque a
    # distribuição de preços é assimétrica (poucos jogadores muito caros).
    residuos = []
    for _, grp in df.groupby("posicao"):
        x = np.log1p(grp["preco"].values)   # log1p evita log(0)
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
            "odd_casa":    o["odd_casa"],    "odd_vis":    o["odd_vis"],
            "odd_empate":  o["odd_empate"],  "prob_casa":  o["prob_casa"],
            "prob_vis":    o["prob_vis"],    "forca_casa": o["forca_casa"],
            "forca_vis":   o["forca_vis"],   "prob_over_25": o.get("prob_over_25"),
        }

    def get_odd(clube_id, campo):
        try:
            abr = mapa_clubes.get(int(clube_id))
            return mapa_odds.get(abr, {}).get(campo)
        except: return None

    df["odd_casa"]    = df[col_casa].apply(lambda x: get_odd(x, "odd_casa"))
    df["odd_vis"]     = df[col_vis].apply( lambda x: get_odd(x, "odd_vis"))
    df["odd_empate"]  = df[col_casa].apply(lambda x: get_odd(x, "odd_empate"))
    df["prob_casa"]   = df[col_casa].apply(lambda x: get_odd(x, "prob_casa"))
    df["prob_vis"]    = df[col_vis].apply( lambda x: get_odd(x, "prob_vis"))
    df["prob_over_25"]= df[col_casa].apply(lambda x: get_odd(x, "prob_over_25"))
    df["forca_casa"]  = df[col_casa].apply(lambda x: get_odd(x, "forca_casa"))
    df["forca_vis"]   = df[col_vis].apply( lambda x: get_odd(x, "forca_vis"))

    return df

def enriquecer_partidas_btts(df_partidas: pd.DataFrame, df_tabela: pd.DataFrame,
                              mapa_clubes: dict) -> pd.DataFrame:
    """
    Estima a probabilidade de 'ambos marcam' via Poisson:
        λ_casa = 0.5 × (gf_recente_casa + ga_recente_vis)
        λ_vis  = 0.5 × (gf_recente_vis + ga_recente_casa)
        P(BTTS) = (1 - e^-λ_casa) × (1 - e^-λ_vis)

    Substitui a fórmula errada que estava no prompt (multiplicar probabilidades
    de vitória — eventos mutuamente excludentes, não BTTS).
    """
    if df_partidas.empty or df_tabela.empty:
        return df_partidas

    df = df_partidas.copy()
    col_casa = next((c for c in df.columns if "casa_id"      in c), None)
    col_vis  = next((c for c in df.columns if "visitante_id" in c), None)
    if not col_casa or not col_vis:
        return df

    tabela_idx = df_tabela.set_index("time")

    def _lambda_btts(row) -> tuple:
        try:
            abr_casa = mapa_clubes.get(int(row[col_casa]))
            abr_vis  = mapa_clubes.get(int(row[col_vis]))
        except Exception:
            return (None, None, None)
        r_casa = get_tabela_row(abr_casa, tabela_idx) if abr_casa else None
        r_vis  = get_tabela_row(abr_vis,  tabela_idx) if abr_vis  else None
        if r_casa is None or r_vis is None:
            return (None, None, None)

        gf_c = float(r_casa.get("media_gf_recente") or r_casa.get("media_gf_temporada") or 0)
        ga_c = float(r_casa.get("media_ga_recente") or r_casa.get("media_ga_temporada") or 0)
        gf_v = float(r_vis .get("media_gf_recente") or r_vis .get("media_gf_temporada") or 0)
        ga_v = float(r_vis .get("media_ga_recente") or r_vis .get("media_ga_temporada") or 0)

        lam_casa = 0.5 * (gf_c + ga_v)
        lam_vis  = 0.5 * (gf_v + ga_c)
        p_btts = (1 - np.exp(-lam_casa)) * (1 - np.exp(-lam_vis))
        return (round(lam_casa, 2), round(lam_vis, 2), round(float(p_btts), 3))

    triples = df.apply(_lambda_btts, axis=1)
    df["lambda_gols_casa"] = [t[0] for t in triples]
    df["lambda_gols_vis"]  = [t[1] for t in triples]
    df["prob_btts_aprox"]  = [t[2] for t in triples]
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

    # Pré-computa lookups por abreviação única (~20 times) em vez de por atleta (~500x)
    abrevs_unicas = set(df["clube"].astype(str).unique()) | set(df["adversario"].astype(str).unique())
    cache_row = {abr: get_tabela_row(abr, tabela_idx) for abr in abrevs_unicas}
    cache_mom = {abr: get_momentum_time(abr, momentum) for abr in abrevs_unicas}

    def _rec_confronto(atleta):
        clube_abr = str(atleta["clube"])
        adv_abr   = str(atleta["adversario"])
        mandante  = atleta["mandante"]
        posicao   = str(atleta["posicao"])

        row_time = cache_row.get(clube_abr)
        row_adv  = cache_row.get(adv_abr)
        mom_time = cache_mom.get(clube_abr, {})
        mom_adv  = cache_mom.get(adv_abr,  {})

        rec = {
            "time_pos":               int(row_time["posicao"]) if row_time is not None else None,
            "adv_pos":                int(row_adv["posicao"])  if row_adv  is not None else None,
            "time_momentum_of":       mom_time.get("momentum_of"),
            "time_momentum_def":      mom_time.get("momentum_def"),
            "adv_momentum_of":        mom_adv.get("momentum_of"),
            "adv_momentum_def":       mom_adv.get("momentum_def"),
            "sequencia_time":         mom_time.get("sequencia"),
            "forma_score_time":       mom_time.get("forma_score"),
            "vantagem_mando":         None,
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
        return rec

    results_df = df.apply(_rec_confronto, axis=1, result_type="expand")
    for col in results_df.columns:
        df[col] = results_df[col].values

    # ── Score composto dinâmico por posição ──────────────────
    oc           = df["oportunidade_confronto"].fillna(0.5)
    vm           = ((df["vantagem_mando"].fillna(0).clip(-50, 50) + 50) / 100)
    tof_norm     = ((df["time_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)
    # tdef: momentum defensivo próprio. Ratio < 1 = defesa em boa fase (sofre
    # menos gols recentemente). Invertido para que "defesa sólida" = valor alto.
    tdef_norm    = 1 - ((df["time_momentum_def"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)
    fs           = df["forma_score_time"].fillna(0.5)
    adv_of_norm  = ((df["adv_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)
    # adv_def: momentum defensivo do adv. Ratio > 1 = defesa sofrendo mais gols
    # recentemente. Direto (não invertido): valor alto = defesa adv fraca =
    # oportunidade para o ataque do time_alvo. Substitui o antigo adv_of_norm
    # usado erroneamente para ataque.
    adv_def_norm = ((df["adv_momentum_def"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)

    # ── Bônus contínuo por probabilidade de vitória ───────────
    SENSIBILIDADE_PROB = {
        "Atacante": 0.40,
        "Meia":     0.30,
        "Técnico":  0.30,
        "Goleiro":  0.15,
        "Lateral":  0.15,
        "Zagueiro": 0.15,
    }

    def bonus_prob(prob: float, posicao: str) -> float:
        """
        Bônus multiplicativo baseado na prob_vitoria contínua.
        Ponto neutro = 0.333 (empate técnico entre 3 resultados).
        Abaixo do neutro = sem penalidade, bônus = 1.0.
        Acima do neutro = bônus proporcional à sensibilidade da posição.
        """
        excesso = max(0.0, float(prob or 0) - 0.333)
        sens    = SENSIBILIDADE_PROB.get(posicao, 0.20)
        return round(1.0 + excesso * sens, 4)

    prob_series = pd.to_numeric(
        df["prob_vitoria"] if "prob_vitoria" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0)

    # prob_over_25: jogo esperado aberto (>2.5 gols). Neutro = 0.5.
    # Para defesa usamos o complemento (1 - prob_gols): jogo fechado = melhor SG.
    # Para ataque usamos direto: mais gols esperados = mais scouts.
    prob_gols_series = pd.to_numeric(
        df["prob_gols"] if "prob_gols" in df.columns else pd.Series(None, index=df.index),
        errors="coerce"
    ).fillna(0.5)

    pesos_cfg = _carregar_pesos_score()

    def calcular_score(row):
        if row["posicao"] in POSICOES_DEFESA:
            p = pesos_cfg["defesa"]
            base = (p.get("oc",        0) * row["oc"]
                  + p.get("vm",        0) * row["vm"]
                  + p.get("tof",       0) * row["tof_norm"]
                  + p.get("tdef",      0) * row["tdef_norm"]
                  + p.get("fs",        0) * row["fs"]
                  + p.get("adv_of",    0) * (1 - row["adv_of_norm"])
                  + p.get("prob_gols", 0) * (1 - row["prob_gols"]))
        else:
            p = pesos_cfg["ataque"]
            base = (p.get("oc",        0) * row["oc"]
                  + p.get("vm",        0) * row["vm"]
                  + p.get("tof",       0) * row["tof_norm"]
                  + p.get("fs",        0) * row["fs"]
                  + p.get("adv_def",   0) * row["adv_def_norm"]
                  + p.get("prob_gols", 0) * row["prob_gols"])
        return base * bonus_prob(row["prob"], row["posicao"])

    df_temp = pd.DataFrame({
        "posicao":      df["posicao"],
        "oc":           oc,
        "vm":           vm,
        "tof_norm":     tof_norm,
        "tdef_norm":    tdef_norm,
        "fs":           fs,
        "adv_of_norm":  adv_of_norm,
        "adv_def_norm": adv_def_norm,
        "prob":         prob_series,
        "prob_gols":    prob_gols_series,
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

    # ── Condição de Mando ─────────────────────────────────────
    # Proxy enquanto o histórico individual (media_casa / media_fora por jogador)
    # acumula rodadas suficientes. Quando disponível, substituir vantagem_mando > 5
    # pelo delta_mando_jogador > 1.5 calculado do histórico.
    #
    # Critérios para 'favoravel' (mandante):
    #   - mandante == True
    #   - vantagem_mando > 5 (time performa bem em casa vs. adversário fora)
    #   - score_confronto_100 > 60 (confronto favorável)
    #   - residuo_z > 0 (jogador entrega acima do esperado pelo preço)
    #
    # Critérios para 'favoravel_visitante' (visitante privilegiado):
    #   - mandante == False, mas confronto muito favorável (sc > 70)
    #   - residuo_z > 0 — ex: 1º vs. último, time forte fora contra fraco em casa
    def _condicao_mando(row):
        vm  = row["vantagem_mando"]  if pd.notna(row["vantagem_mando"])  else 0.0
        sc  = row["score_confronto_100"] if pd.notna(row["score_confronto_100"]) else 50.0
        rz  = row["residuo_z"]       if pd.notna(row["residuo_z"])       else 0.0
        m   = row["mandante"]        if pd.notna(row["mandante"])        else False
        if m and vm > 5 and sc > 60 and rz > 0:
            return "favoravel"
        if (not m) and sc > 70 and rz > 0:
            return "favoravel_visitante"
        if (not m) and vm < -5 and sc < 40:
            return "desfavoravel"
        return "neutro"

    df["condicao_mando"] = df.apply(_condicao_mando, axis=1)

    # ── Pontos Esperados ─────────────────────────────────────
    # Retorno absoluto esperado na rodada. Usa modelo calibrado a partir do histórico
    # (calibracao_pontos.json) quando há dados suficientes; caso contrário, aplica a
    # heurística multiplicativa original.
    df["pontos_esperados"] = calcular_pontos_esperados(df).round(2)

    # ── Caro e vale + Recomendação ───────────────────────────
    df = _classificar_atletas(df)

    return df


# Limiares de "caro" por posição — mesmos do prompt. Jogador acima do limiar
# exige justificativa forte (condição de mando favorável + pontos esperados
# acima da mediana da posição).
LIMIAR_CARO = {
    "Goleiro":  8.0,
    "Lateral":  9.0,
    "Zagueiro": 9.0,
    "Meia":    12.0,
    "Atacante": 14.0,
    "Técnico":  6.0,
}

def _classificar_atletas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona as colunas derivadas que resumem a decisão:

    caro_e_vale (bool): preço acima do limiar + condição de mando favorável +
        pontos esperados acima da mediana da posição. Responde "esse jogador
        caro vale o custo nessa rodada?" sem a LLM ter de recalcular.

    recomendacao (categórica):
        EVITAR       — armadilha_forte ou status != Provável
        RESERVA_LUXO — valor_oculto com alto teto (pontos_esperados no top 20% da posição)
        TITULAR      — pontos_esperados >= mediana, confiável, resiliente
        BANCO        — barato (preço ≤ p25), confiável e com residuo >= 0
        WATCH        — demais casos
    """
    df = df.copy()

    mediana_pe = df.groupby("posicao")["pontos_esperados"].transform("median")
    p20_pe     = df.groupby("posicao")["pontos_esperados"].transform(lambda s: s.quantile(0.80))
    p25_preco  = df.groupby("posicao")["preco"].transform(lambda s: s.quantile(0.25))

    limiar_preco = df["posicao"].map(LIMIAR_CARO).fillna(10.0)

    df["caro_e_vale"] = (
        (df["preco"] > limiar_preco) &
        (df["condicao_mando"].isin(["favoravel", "favoravel_visitante"])) &
        (df["pontos_esperados"] > mediana_pe)
    )

    is_armadilha_f   = df["armadilha_label"] == "armadilha_forte"
    is_armadilha_l   = df["armadilha_label"] == "armadilha_leve"
    is_valor_oculto  = df["armadilha_label"] == "valor_oculto"
    conf             = df["confiabilidade"].astype(float)
    resil            = df.get("resiliencia_pct", pd.Series(0.0, index=df.index)).astype(float)
    resid            = df["residuo_z"].astype(float)

    # Status bloqueante: Suspenso(3), Contundido(5), Nulo(6). Dúvida(2) NÃO bloqueia.
    is_status_bloqueante = df["status_id"].astype(float).isin([3, 5, 6])

    titular_cand = (
        (df["pontos_esperados"] >= mediana_pe) &
        (resil >= 0.5) & (conf >= 0.6) &
        (~is_armadilha_l) & (~is_armadilha_f) & (~is_status_bloqueante)
    )
    alto_teto = is_valor_oculto & (df["pontos_esperados"] >= p20_pe)
    banco_cand = (
        (df["preco"] <= p25_preco) & (conf >= 0.4) & (resid >= 0) & (~is_armadilha_f) & (~is_status_bloqueante)
    )

    # Ordem de prioridade (primeira regra que bate vence):
    # 1) EVITAR: armadilha_forte OU status bloqueante (Suspenso/Contundido/Nulo)
    # 2) alto teto com inconsistência → RESERVA_LUXO
    # 3) TITULAR: consistente, confiável, resiliente, sem armadilhas
    # 4) BANCO: barato, estável
    # 5) WATCH: demais casos (inclui Dúvida com valor)
    recomendacao = pd.Series("WATCH", index=df.index)
    recomendacao[banco_cand]                       = "BANCO"
    recomendacao[titular_cand]                     = "TITULAR"
    recomendacao[alto_teto & ~titular_cand]      = "RESERVA_LUXO"
    recomendacao[is_armadilha_f | is_status_bloqueante] = "EVITAR"

    df["recomendacao"] = recomendacao
    return df


def _carregar_calibracao() -> dict:
    path = CURRENT_DIR / "calibracao_pontos.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _carregar_pesos_score() -> dict:
    """
    Lê os pesos calibrados do score_confronto de calibracao_score.json.
    Retorna PESOS_SCORE_DEFAULT quando o arquivo não existe, está inválido
    ou quando o modelo ainda não superou a heurística (status != 'ok').
    """
    path = CURRENT_DIR / "calibracao_score.json"
    if not path.exists():
        return PESOS_SCORE_DEFAULT
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return PESOS_SCORE_DEFAULT
    if payload.get("status") != "ok":
        return PESOS_SCORE_DEFAULT
    pesos = payload.get("pesos", {})
    if not (isinstance(pesos, dict) and "defesa" in pesos and "ataque" in pesos):
        return PESOS_SCORE_DEFAULT
    # merge preservando defaults para chaves ausentes
    merged = {
        "defesa": {**PESOS_SCORE_DEFAULT["defesa"], **pesos.get("defesa", {})},
        "ataque": {**PESOS_SCORE_DEFAULT["ataque"], **pesos.get("ataque", {})},
    }
    return merged


def _calcular_forma_historica(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Carrega histórico de pontuações reais de docs/data/historico/ e calcula
    features de forma recente para cada atleta em df (sem vazamento — todas
    as rodadas no histórico são anteriores à rodada sendo prevista).

    Colunas adicionadas:
        forma_media_3r   — média das últimas N pontuações (fallback: media_bayesiana)
        forma_jogou_3r   — quantas das últimas N rodadas o atleta entrou em campo
        forma_tendencia  — última pontuação menos média das anteriores
    """
    df = df.copy()
    historico: dict = {}

    if HISTORICO_DIR.exists():
        for pasta in sorted(HISTORICO_DIR.glob("r*"), key=lambda p: int(p.name.lstrip("r"))):
            pts_path = pasta / "atletas_pontuados.csv"
            if not pts_path.exists():
                continue
            try:
                pts = pd.read_csv(pts_path, encoding="utf-8-sig")
                if "atleta_id" not in pts.columns or "pontuacao" not in pts.columns:
                    continue
                entrou_col = "entrou_em_campo" if "entrou_em_campo" in pts.columns else None
                for row in pts.itertuples(index=False):
                    if entrou_col:
                        entrou = str(getattr(row, entrou_col)).lower() in ("true", "1", "1.0")
                        if not entrou:
                            continue
                    aid = int(row.atleta_id)
                    historico.setdefault(aid, []).append(float(row.pontuacao))
            except Exception:
                continue

    bayes = df["media_bayesiana"].astype(float).values
    medias, jogou, tendencias = [], [], []

    for i, aid in enumerate(df["atleta_id"]):
        try:
            hist = historico.get(int(aid), [])
        except Exception:
            hist = []
        ultimas = hist[-n:]

        if not ultimas:
            medias.append(bayes[i])  # sem histórico: usa media_bayesiana como proxy
            jogou.append(0)
            tendencias.append(0.0)
        else:
            media = float(np.mean(ultimas))
            medias.append(media)
            jogou.append(len(ultimas))
            tend = float(ultimas[-1] - np.mean(ultimas[:-1])) if len(ultimas) >= 2 else 0.0
            tendencias.append(tend)

    df["forma_media_3r"]  = medias
    df["forma_jogou_3r"]  = jogou
    df["forma_tendencia"] = tendencias
    return df


def calcular_pontos_esperados(df: pd.DataFrame) -> pd.Series:
    bayes       = df["media_bayesiana"].astype(float)
    score_ratio = (df["score_confronto_100"].fillna(50).astype(float) / 50)
    conf        = df["confiabilidade"].astype(float)

    calib = _carregar_calibracao()
    if calib.get("status") == "ok" and "coefs" in calib:
        c = calib["coefs"]

        # Calcula forma recente se o modelo foi treinado com essas features
        if "forma_media_3r" in c:
            df_forma    = _calcular_forma_historica(df)
            forma_media = df_forma["forma_media_3r"].astype(float)
            forma_jogou = df_forma["forma_jogou_3r"].astype(float)
            forma_tend  = df_forma["forma_tendencia"].astype(float)
        else:
            forma_media = bayes
            forma_jogou = pd.Series(0.0, index=df.index)
            forma_tend  = pd.Series(0.0, index=df.index)

        return (
            c.get("intercept",        0.0)
            + c.get("bayes",          0.0) * bayes
            + c.get("score_ratio",    0.0) * score_ratio
            + c.get("conf",           0.0) * conf
            + c.get("interacao",      0.0) * bayes * score_ratio * conf
            + c.get("forma_media_3r", 0.0) * forma_media
            + c.get("forma_jogou_3r", 0.0) * forma_jogou
            + c.get("forma_tendencia",0.0) * forma_tend
        )

    # Fallback: heurística multiplicativa original
    return bayes * score_ratio * conf

# ─────────────────────────────────────────────────────────────
# SNAPSHOT HISTÓRICO
# ─────────────────────────────────────────────────────────────

# Colunas salvas nos snapshots históricos
# Inclui entrou_em_campo para calcular taxa de participação futuramente
# Features brutas de confronto (oc, vm, momentums, forma, prob) são necessárias
# para calibrar os pesos do score_confronto via calibrar_score.py.
COLUNAS_SNAPSHOT = [
    "atleta_id", "nome", "clube", "posicao", "status_id",
    "preco", "variacao", "media", "jogos", "pontos_rodada",
    "entrou_em_campo",                          # ← base para taxa de participação
    "mandante", "adversario",
    "residuo_z", "armadilha_label",
    "media_bayesiana", "pb_media", "confiabilidade",
    "score_confronto_100",
    "condicao_mando", "pontos_esperados",  # ← base para split real de mando no futuro
    # Features brutas para calibração do score_confronto
    "oportunidade_confronto", "vantagem_mando",
    "time_momentum_of", "time_momentum_def",
    "adv_momentum_of",  "adv_momentum_def",
    "forma_score_time", "prob_vitoria", "prob_gols",
]

def _pasta_rodada(rodada: int) -> Path:
    pasta = HISTORICO_DIR / f"r{rodada}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta

def _colunas_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas as colunas relevantes para o snapshot."""
    cols = [c for c in COLUNAS_SNAPSHOT if c in df.columns]
    return df[cols].copy()

def salvar_snapshot_pontuados(raw_pontuados: dict):
    """
    Snapshot de pontuação — salvo uma única vez por rodada, logo após os jogos.
    Usa o raw do endpoint /atletas/pontuados, que inclui scouts individuais.

    Colunas salvas:
        atleta_id, nome, clube, posicao, pontuacao, entrou_em_campo, scout_*

    A rodada é extraída do próprio JSON (campo 'rodada').
    O endpoint só retorna dados durante/logo após a rodada — depois fica vazio.
    """
    rodada = raw_pontuados.get("rodada")
    atletas = raw_pontuados.get("atletas", {})
    if not rodada or not atletas:
        print("  [PONTUADOS] sem rodada ou atletas no raw, pulando")
        return

    pasta = _pasta_rodada(int(rodada))
    path  = pasta / "atletas_pontuados.csv"
    if path.exists():
        print(f"  [PONTUADOS] já existe — rodada {rodada}, pulando")
        return

    clubes   = {int(k): v.get("abreviacao", k) for k, v in raw_pontuados.get("clubes", {}).items()}
    posicoes = {int(k): v if isinstance(v, str) else v.get("nome", k)
                for k, v in raw_pontuados.get("posicoes", {}).items()}

    rows = []
    for atleta_id, a in atletas.items():
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id":       int(atleta_id),
            "nome":            a.get("apelido") or a.get("nome"),
            "clube":           clubes.get(int(a.get("clube_id", 0)), a.get("clube_id")),
            "posicao":         posicoes.get(int(a.get("posicao_id", 0)), a.get("posicao_id")),
            "pontuacao":       a.get("pontuacao"),
            "entrou_em_campo": a.get("entrou_em_campo"),
            **{f"scout_{k}": v for k, v in scouts.items()},
        })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [PONTUADOS] salvo — rodada {rodada} ({len(rows)} atletas) → {path}")


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

    # Janela de 4h antes do fechamento → PRÉ da rodada atual
    # (configurável via env var JANELA_PRE_HORAS, padrão 4h)
    janela_pre_seg = int(os.environ.get("JANELA_PRE_HORAS", "8")) * 3600
    dentro_janela_pre = agora_ts > (fechamento_ts - janela_pre_seg)
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
                       df_tabela: pd.DataFrame,
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

if __name__ == "__main__":
    log = []

    # ── 1. Coleta e normalização dos endpoints Cartola ───────────
    EXTRATORES = {
        "mercado":   (normalizar_mercado,   "mercado"),
        "pontuados": (normalizar_pontuados, "atletas_pontuados"),
        "partidas":  (normalizar_partidas,  "partidas"),
        "rodadas":   (normalizar_rodadas,   "rodadas"),
    }

    dados_brutos = {}
    raw_pontuados = {}  # preserva o raw para salvar_snapshot_pontuados
    for key, (normalizador, nome_arquivo) in EXTRATORES.items():
        print(f"Extraindo {key}...")
        try:
            raw = get_json(key)
            salvar_raw_json(nome_arquivo, raw)                          # → docs/data/raw/
            df  = normalizador(raw)
            df.to_csv(CURRENT_DIR / f"{nome_arquivo}.csv",             # → docs/data/current/
                      index=False, encoding="utf-8-sig")
            dados_brutos[key] = df
            if key == "pontuados":
                raw_pontuados = raw                                     # ← preserva para historico
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

    # ── 5b. BTTS aproximado via Poisson ──────────────────────────
    try:
        if not df_partidas.empty and not df_tabela.empty:
            df_partidas = enriquecer_partidas_btts(df_partidas, df_tabela, mapa_clubes)
            df_partidas.to_csv(CURRENT_DIR / "partidas.csv", index=False, encoding="utf-8-sig")
            print("  OK — prob_btts_aprox calculada via Poisson")
    except Exception as e:
        print(f"  ERRO no BTTS: {e}")

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

    print("Salvando pontuações da rodada...")
    try:
        if raw_pontuados:
            salvar_snapshot_pontuados(raw_pontuados)
    except Exception as e:
        print(f"  ERRO no snapshot pontuados: {e}")

    # ── 7b. Calibração do pontos_esperados ───────────────────────
    # Ajusta os coeficientes sempre que há novos snapshots. O run SEGUINTE do
    # extractor consome calibracao_pontos.json via calcular_pontos_esperados().
    print("Calibrando pontos_esperados...")
    try:
        import calibrar_pontos_esperados as calibrador
        calibrador.main()
    except Exception as e:
        print(f"  ERRO na calibração: {e}")

    # ── 7c. Calibração dos pesos do score_confronto ──────────────
    # Só produz modelo quando snapshots rN tiverem as features brutas
    # (oportunidade_confronto, vantagem_mando, *_momentum_*, forma_score_time).
    # Snapshots pré-migração ficam ignorados sem quebrar.
    print("Calibrando score_confronto...")
    try:
        import calibrar_score as cal_score
        cal_score.main()
    except Exception as e:
        print(f"  ERRO na calibração do score: {e}")

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
                df_atletas_enriquecido, df_partidas, df_tabela, momentum
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
