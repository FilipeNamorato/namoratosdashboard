"""
calibrar_pontos_esperados.py
----------------------------
Varre os snapshots históricos (docs/data/historico/rN/), cruza o snapshot PRÉ
com as pontuações reais, e calibra os coeficientes de pontos_esperados.

Heurística original do extractor:
    pontos_esperados = media_bayesiana × (score_confronto_100 / 50) × confiabilidade

Modelo aprendido (Ridge com interação + forma recente, ponderado por recência):
    y = a0 + a1·bayes + a2·score_ratio + a3·conf + a4·(bayes×score_ratio×conf)
      + a5·forma_media_3r + a6·forma_jogou_3r + a7·forma_tendencia

Features de forma recente calculadas sem vazamento (só rodadas anteriores à prevista).
MAE calculado via leave-one-round-out CV. Modelo só adotado se bater a heurística OOS.

Saída: docs/data/current/calibracao_pontos.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

BRT           = timezone(timedelta(hours=-3))
HISTORICO_DIR = Path("docs/data/historico")
CURRENT_DIR   = Path("docs/data/current")
OUT_PATH      = CURRENT_DIR / "calibracao_pontos.json"

MIN_RODADAS   = 3
MIN_REGISTROS = 200
RIDGE_LAMBDA  = 1.0     # regularização L2
RECENCY_DECAY = 0.9     # peso = RECENCY_DECAY ^ (idade_em_rodadas)


def coletar_dataset() -> pd.DataFrame:
    """Cruza PRE com pontuados em todas rodadas disponíveis e adiciona forma recente."""
    if not HISTORICO_DIR.exists():
        return pd.DataFrame()

    frames = []
    for pasta in sorted(HISTORICO_DIR.glob("r*"), key=lambda p: int(p.name.lstrip("r"))):
        pre_path = pasta / "atletas_pre.csv"
        pts_path = pasta / "atletas_pontuados.csv"
        if not (pre_path.exists() and pts_path.exists()):
            continue

        pre = pd.read_csv(pre_path, encoding="utf-8-sig")
        pts = pd.read_csv(pts_path, encoding="utf-8-sig")

        cols_pre = ["atleta_id", "media_bayesiana", "score_confronto_100", "confiabilidade", "posicao"]
        cols_pre = [c for c in cols_pre if c in pre.columns]
        if "atleta_id" not in cols_pre:
            continue

        cols_pts = ["atleta_id", "pontuacao", "entrou_em_campo"]
        cols_pts = [c for c in cols_pts if c in pts.columns]
        if "pontuacao" not in cols_pts or "entrou_em_campo" not in cols_pts:
            continue

        merged = pre[cols_pre].merge(pts[cols_pts], on="atleta_id", how="inner")
        merged["rodada"] = pasta.name
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return _adicionar_forma_recente(df)


def _adicionar_forma_recente(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Adiciona features de forma recente sem vazamento de dados:
    - forma_media_Nr: média de pontuação das últimas N rodadas anteriores
    - forma_jogou_Nr: quantas das últimas N rodadas o atleta entrou em campo
    - forma_tendencia: última pontuação menos média das anteriores (captura alta/queda)

    Atletas sem histórico recebem NaN (tratado em montar_features).
    """
    rodadas_ord = sorted(df["rodada"].unique(), key=lambda r: int(r.lstrip("r")))
    historico: dict[int, list[float]] = {}  # atleta_id -> lista de pontuações (ordem crescente)

    result_frames = []
    for rodada in rodadas_ord:
        subset = df[df["rodada"] == rodada].copy()

        medias, jogou, tendencias = [], [], []
        for row in subset.itertuples(index=False):
            hist = historico.get(row.atleta_id, [])
            ultimas = hist[-n:]

            if not ultimas:
                medias.append(np.nan)
                jogou.append(0)
                tendencias.append(np.nan)
            else:
                media = float(np.mean(ultimas))
                medias.append(media)
                jogou.append(len(ultimas))
                tend = float(ultimas[-1] - np.mean(ultimas[:-1])) if len(ultimas) >= 2 else 0.0
                tendencias.append(tend)

        subset[f"forma_media_{n}r"]    = medias
        subset[f"forma_jogou_{n}r"]    = jogou
        subset["forma_tendencia"]       = tendencias
        result_frames.append(subset)

        # atualiza histórico com pontuações desta rodada (sem vazar para a próxima)
        for row in subset.itertuples(index=False):
            if row.atleta_id not in historico:
                historico[row.atleta_id] = []
            historico[row.atleta_id].append(float(row.pontuacao))

    return pd.concat(result_frames, ignore_index=True)


def montar_features(df: pd.DataFrame) -> tuple:
    bayes       = df["media_bayesiana"].astype(float).values
    score_ratio = (df["score_confronto_100"].astype(float).fillna(50) / 50).values
    conf        = df["confiabilidade"].astype(float).values
    interacao   = bayes * score_ratio * conf

    # forma recente: NaN (sem histórico) → usa media_bayesiana como fallback
    forma_media = pd.to_numeric(df.get("forma_media_3r", pd.Series(dtype=float)), errors="coerce").values
    forma_media = np.where(np.isnan(forma_media), bayes, forma_media)

    forma_jogou = pd.to_numeric(df.get("forma_jogou_3r", pd.Series(dtype=float)), errors="coerce").fillna(0).values

    forma_tend  = pd.to_numeric(df.get("forma_tendencia", pd.Series(dtype=float)), errors="coerce").values
    forma_tend  = np.where(np.isnan(forma_tend), 0.0, forma_tend)

    X = np.column_stack([
        np.ones_like(bayes),  # intercept
        bayes,
        score_ratio,
        conf,
        interacao,
        forma_media,
        forma_jogou,
        forma_tend,
    ])
    y = df["pontuacao"].astype(float).values
    return X, y


def pesos_recencia(rodadas: np.ndarray) -> np.ndarray:
    """Pesos exponenciais: rodada mais nova = 1.0, mais antiga decai."""
    idx = np.array([int(r.lstrip("r")) for r in rodadas])
    idade = idx.max() - idx
    return RECENCY_DECAY ** idade


def fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Ridge ponderado: β = (XᵀWX + λI)⁻¹ XᵀWy. Intercept não é regularizado."""
    W = np.diag(w)
    n_feat = X.shape[1]
    I = np.eye(n_feat)
    I[0, 0] = 0.0  # não regulariza intercept
    A = X.T @ W @ X + lam * I
    b = X.T @ W @ y
    return np.linalg.solve(A, b)


def avaliar_in_sample(X: np.ndarray, y: np.ndarray, coefs: np.ndarray) -> tuple:
    y_hat = X @ coefs
    resid = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(resid)))
    return r2, mae


def cv_leave_one_round_out(df: pd.DataFrame) -> float:
    """MAE out-of-sample: para cada rodada, treina nas demais e prevê nela."""
    rodadas = df["rodada"].unique()
    erros = []
    for r in rodadas:
        treino = df[df["rodada"] != r]
        teste  = df[df["rodada"] == r]
        if len(treino) < 50 or len(teste) == 0:
            continue
        X_tr, y_tr = montar_features(treino)
        X_te, y_te = montar_features(teste)
        w_tr = pesos_recencia(treino["rodada"].values)
        coefs = fit_ridge(X_tr, y_tr, w_tr)
        y_pred = X_te @ coefs
        erros.extend(np.abs(y_te - y_pred).tolist())
    return float(np.mean(erros)) if erros else float("inf")


def mae_heuristica(df: pd.DataFrame) -> float:
    bayes = df["media_bayesiana"].astype(float).values
    score_ratio = (df["score_confronto_100"].astype(float).fillna(50) / 50).values
    conf = df["confiabilidade"].astype(float).values
    pred = bayes * score_ratio * conf
    return float(np.mean(np.abs(df["pontuacao"].astype(float).values - pred)))


def main() -> None:
    print("Calibrando pontos_esperados...")
    df = coletar_dataset()
    if df.empty:
        _salvar_status_insuficiente(n_rodadas=0, n_registros=0, motivo="sem snapshots PRE+pontuados")
        return

    df = df[df["entrou_em_campo"].astype(str).str.lower().isin(["true", "1", "1.0"])].copy()
    df = df[df["media_bayesiana"].astype(float) > 0].copy()

    n_rodadas = df["rodada"].nunique()
    n_registros = len(df)
    print(f"  {n_rodadas} rodada(s), {n_registros} registros com entrou_em_campo=True")

    if n_rodadas < MIN_RODADAS or n_registros < MIN_REGISTROS:
        mae_h = mae_heuristica(df) if n_registros > 0 else None
        _salvar_status_insuficiente(
            n_rodadas=n_rodadas,
            n_registros=n_registros,
            mae_heuristica=mae_h,
            motivo=f"precisa ≥{MIN_RODADAS} rodadas e ≥{MIN_REGISTROS} registros",
        )
        return

    X, y = montar_features(df)
    w = pesos_recencia(df["rodada"].values)
    coefs = fit_ridge(X, y, w)
    r2, mae_in = avaliar_in_sample(X, y, coefs)
    mae_cv = cv_leave_one_round_out(df)
    mae_h = mae_heuristica(df)

    # só adota o modelo se bater a heurística out-of-sample
    if mae_cv >= mae_h:
        print(f"  modelo NÃO superou heurística OOS (MAE {mae_cv:.3f} vs {mae_h:.3f}) — mantendo fórmula")
        _salvar_status_insuficiente(
            n_rodadas=n_rodadas,
            n_registros=n_registros,
            mae_heuristica=mae_h,
            motivo=f"modelo OOS ({mae_cv:.3f}) não bate heurística ({mae_h:.3f})",
        )
        return

    payload = {
        "status": "ok",
        "n_rodadas": int(n_rodadas),
        "n_registros": int(n_registros),
        "coefs": {
            "intercept":       float(coefs[0]),
            "bayes":           float(coefs[1]),
            "score_ratio":     float(coefs[2]),
            "conf":            float(coefs[3]),
            "interacao":       float(coefs[4]),
            "forma_media_3r":  float(coefs[5]),
            "forma_jogou_3r":  float(coefs[6]),
            "forma_tendencia": float(coefs[7]),
        },
        "metricas": {
            "mae_heuristica":  round(float(mae_h),  3),
            "mae_modelo_cv":   round(float(mae_cv), 3),
            "mae_modelo_in":   round(float(mae_in), 3),
            "r2_modelo_in":    round(float(r2),     3),
            "ganho_pct_cv":    round(float((mae_h - mae_cv) / mae_h * 100), 1) if mae_h > 0 else 0.0,
        },
        "hiperparams": {
            "ridge_lambda":  RIDGE_LAMBDA,
            "recency_decay": RECENCY_DECAY,
        },
        "atualizado_em": datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT"),
    }

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  modelo calibrado — MAE_cv {mae_cv:.3f} vs heurística {mae_h:.3f} "
          f"(ganho {payload['metricas']['ganho_pct_cv']}%) | R²_in {r2:.3f}")
    print(f"  salvo em {OUT_PATH}")


def _salvar_status_insuficiente(n_rodadas: int, n_registros: int,
                                 mae_heuristica: float = None,
                                 motivo: str = "") -> None:
    payload = {
        "status":        "insuficiente",
        "n_rodadas":     int(n_rodadas),
        "n_registros":   int(n_registros),
        "motivo":        motivo,
        "atualizado_em": datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT"),
    }
    if mae_heuristica is not None:
        payload["metricas"] = {"mae_heuristica": round(float(mae_heuristica), 3)}
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [INSUFICIENTE] {motivo} — extractor usará fórmula heurística")


if __name__ == "__main__":
    main()
