"""
calibrar_pontos_esperados.py
----------------------------
Varre os snapshots históricos (docs/data/historico/rN/), cruza o snapshot PRÉ
com as pontuações reais, e calibra os coeficientes de pontos_esperados.

Heurística original do extractor:
    pontos_esperados = media_bayesiana × (score_confronto_100 / 50) × confiabilidade

Modelo aprendido (OLS com interação):
    y = a0 + a1·bayes + a2·score_ratio + a3·conf + a4·(bayes × score_ratio × conf)

Saída: docs/data/current/calibracao_pontos.json
    {
      "status": "ok" | "insuficiente",
      "n_rodadas": int,
      "n_registros": int,
      "coefs": {...},
      "metricas": {"mae_heuristica": float, "mae_modelo": float, "r2_modelo": float},
      "atualizado_em": "YYYY-MM-DD HH:MM"
    }

Se status == "insuficiente", o extractor mantém a fórmula heurística.
Critério atual: >= 3 rodadas com PRÉ+pontuados e >= 200 registros com entrou_em_campo=True.
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


def coletar_dataset() -> pd.DataFrame:
    """Cruza PRE com pontuados em todas rodadas disponíveis."""
    if not HISTORICO_DIR.exists():
        return pd.DataFrame()

    frames = []
    for pasta in sorted(HISTORICO_DIR.glob("r*")):
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

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def montar_features(df: pd.DataFrame) -> tuple:
    bayes = df["media_bayesiana"].astype(float).values
    score_ratio = (df["score_confronto_100"].astype(float).fillna(50) / 50).values
    conf = df["confiabilidade"].astype(float).values
    interacao = bayes * score_ratio * conf

    X = np.column_stack([
        np.ones_like(bayes),     # intercept
        bayes,
        score_ratio,
        conf,
        interacao,
    ])
    y = df["pontuacao"].astype(float).values
    return X, y


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    resid = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(resid))
    return coefs, r2, mae


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
    coefs, r2, mae_m = fit_ols(X, y)
    mae_h = mae_heuristica(df)

    payload = {
        "status": "ok",
        "n_rodadas": int(n_rodadas),
        "n_registros": int(n_registros),
        "coefs": {
            "intercept":  float(coefs[0]),
            "bayes":      float(coefs[1]),
            "score_ratio": float(coefs[2]),
            "conf":       float(coefs[3]),
            "interacao":  float(coefs[4]),
        },
        "metricas": {
            "mae_heuristica": round(float(mae_h), 3),
            "mae_modelo":     round(float(mae_m), 3),
            "r2_modelo":      round(float(r2),    3),
            "ganho_pct":      round(float((mae_h - mae_m) / mae_h * 100), 1) if mae_h > 0 else 0.0,
        },
        "atualizado_em": datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT"),
    }

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  modelo calibrado — MAE {mae_m:.3f} vs heurística {mae_h:.3f} "
          f"(ganho {payload['metricas']['ganho_pct']}%) | R² {r2:.3f}")
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
