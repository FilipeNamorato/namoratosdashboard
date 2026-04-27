"""
calibrar_score.py
-----------------
Calibra os pesos do score_confronto contra a pontuação real observada nos
snapshots históricos (docs/data/historico/rN/atletas_pre.csv cruzado com
atletas_pontuados.csv).

Treina um modelo Ridge ponderado por recência, separado por grupo:

    defesa  (Goleiro/Zagueiro/Lateral):
        pontuacao ~ oc + vm + tof_norm + tdef_norm + fs + (1 - adv_of_norm)

    ataque  (Atacante/Meia/Técnico):
        pontuacao ~ oc + vm + tof_norm + fs + adv_def_norm

Só filtra atletas com entrou_em_campo=True (evita enviesar com reservas que
não jogaram). Valida out-of-sample via leave-one-round-out: se o MAE do modelo
NÃO superar a heurística (soma ponderada com PESOS_SCORE_DEFAULT), mantém
default.

Saída:
    docs/data/current/calibracao_score.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

BRT           = timezone(timedelta(hours=-3))
HISTORICO_DIR = Path("docs/data/historico")
CURRENT_DIR   = Path("docs/data/current")
OUT_PATH      = CURRENT_DIR / "calibracao_score.json"

MIN_RODADAS   = 3
MIN_REGISTROS = 150    # por grupo (defesa/ataque)
RIDGE_LAMBDA  = 1.0
RECENCY_DECAY = 0.9

POSICOES_ATAQUE = {"Atacante", "Meia", "Técnico"}
POSICOES_DEFESA = {"Zagueiro", "Lateral", "Goleiro"}

# Mesmos defaults do extractor — replicados aqui para evitar import circular
# quando o extractor chamar este módulo.
PESOS_SCORE_DEFAULT = {
    "defesa": {
        "oc": 0.30, "vm": 0.25, "tof": 0.10,
        "tdef": 0.15, "fs": 0.05, "adv_of": 0.15,
    },
    "ataque": {
        "oc": 0.30, "vm": 0.15, "tof": 0.30,
        "fs": 0.10, "adv_def": 0.15,
    },
}

FEATURES_DEF = ["oc", "vm", "tof", "tdef", "fs", "adv_of"]
FEATURES_ATK = ["oc", "vm", "tof", "fs", "adv_def"]


# ─────────────────────────────────────────────────────────────
# Coleta e normalização
# ─────────────────────────────────────────────────────────────

def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduz a normalização do extractor para as features brutas."""
    out = pd.DataFrame(index=df.index)
    out["oc"]      = pd.to_numeric(df.get("oportunidade_confronto"), errors="coerce").fillna(0.5)
    vm_raw         = pd.to_numeric(df.get("vantagem_mando"), errors="coerce").fillna(0)
    out["vm"]      = (vm_raw.clip(-50, 50) + 50) / 100
    tof_raw        = pd.to_numeric(df.get("time_momentum_of"), errors="coerce").fillna(1.0)
    out["tof"]     = (tof_raw.clip(0.3, 2.0) - 0.3) / 1.7
    tdef_raw       = pd.to_numeric(df.get("time_momentum_def"), errors="coerce").fillna(1.0)
    out["tdef"]    = 1 - ((tdef_raw.clip(0.3, 2.0) - 0.3) / 1.7)
    out["fs"]      = pd.to_numeric(df.get("forma_score_time"), errors="coerce").fillna(0.5)
    adv_of_raw     = pd.to_numeric(df.get("adv_momentum_of"), errors="coerce").fillna(1.0)
    # Na fórmula do extractor, defesa usa (1 - adv_of_norm). Usamos a mesma
    # transformação aqui para que o coeficiente tenha interpretação direta.
    out["adv_of"]  = 1 - ((adv_of_raw.clip(0.3, 2.0) - 0.3) / 1.7)
    adv_def_raw    = pd.to_numeric(df.get("adv_momentum_def"), errors="coerce").fillna(1.0)
    out["adv_def"] = (adv_def_raw.clip(0.3, 2.0) - 0.3) / 1.7
    return out


def coletar_dataset() -> pd.DataFrame:
    """Cruza atletas_pre com atletas_pontuados por rodada e normaliza features."""
    if not HISTORICO_DIR.exists():
        return pd.DataFrame()

    cols_necessarias = {
        "atleta_id", "posicao", "oportunidade_confronto", "vantagem_mando",
        "time_momentum_of", "time_momentum_def",
        "adv_momentum_of", "adv_momentum_def",
        "forma_score_time",
    }

    frames = []
    for pasta in sorted(HISTORICO_DIR.glob("r*"), key=lambda p: int(p.name.lstrip("r"))):
        pre_path = pasta / "atletas_pre.csv"
        pts_path = pasta / "atletas_pontuados.csv"
        if not (pre_path.exists() and pts_path.exists()):
            continue

        pre = pd.read_csv(pre_path, encoding="utf-8-sig")
        pts = pd.read_csv(pts_path, encoding="utf-8-sig")

        # Se o snapshot PRE não tem as features brutas necessárias, pula
        # (acontece em rodadas salvas antes da migração do COLUNAS_SNAPSHOT).
        if not cols_necessarias.issubset(set(pre.columns)):
            continue

        if "pontuacao" not in pts.columns or "entrou_em_campo" not in pts.columns:
            continue

        merged = pre.merge(
            pts[["atleta_id", "pontuacao", "entrou_em_campo"]],
            on="atleta_id", how="inner",
            suffixes=("_pre", "_pts"),
        )
        # só mantém quem efetivamente jogou (usa entrou_em_campo do pontuados, não do pré)
        ec_col = "entrou_em_campo_pts" if "entrou_em_campo_pts" in merged.columns else "entrou_em_campo"
        merged = merged[
            merged[ec_col].astype(str).str.lower().isin(["true", "1", "1.0"])
        ].copy()
        if merged.empty:
            continue
        merged["rodada"] = pasta.name
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# Ridge ponderado por recência
# ─────────────────────────────────────────────────────────────

def _pesos_recencia(rodadas: np.ndarray) -> np.ndarray:
    idx = np.array([int(r.lstrip("r")) for r in rodadas])
    idade = idx.max() - idx
    return RECENCY_DECAY ** idade


def _fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """β = (Xᵀ W X + λ I)⁻¹ Xᵀ W y. Intercept (coluna 0) não é regularizado."""
    W = np.diag(w)
    n_feat = X.shape[1]
    I = np.eye(n_feat)
    I[0, 0] = 0.0
    A = X.T @ W @ X + lam * I
    b = X.T @ W @ y
    return np.linalg.solve(A, b)


def _monta_X(df_norm: pd.DataFrame, features: list) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df_norm)),  # intercept
        *[df_norm[f].astype(float).values for f in features],
    ])


# ─────────────────────────────────────────────────────────────
# Avaliação
# ─────────────────────────────────────────────────────────────

def _mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def _mae_heuristica(df_norm: pd.DataFrame, y: np.ndarray, pesos: dict, features: list) -> float:
    """MAE da predição que usa os pesos default (sem intercept aprendido)."""
    y_hat = np.zeros(len(df_norm))
    for f in features:
        y_hat += pesos.get(f, 0.0) * df_norm[f].astype(float).values
    return _mae(y, y_hat)


def _cv_leave_one_round_out(df_full: pd.DataFrame, df_norm: pd.DataFrame,
                             features: list) -> float:
    rodadas = df_full["rodada"].unique()
    erros = []
    for r in rodadas:
        mask_tr = df_full["rodada"] != r
        mask_te = df_full["rodada"] == r
        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            continue
        X_tr = _monta_X(df_norm[mask_tr], features)
        X_te = _monta_X(df_norm[mask_te], features)
        y_tr = df_full.loc[mask_tr, "pontuacao"].astype(float).values
        y_te = df_full.loc[mask_te, "pontuacao"].astype(float).values
        w_tr = _pesos_recencia(df_full.loc[mask_tr, "rodada"].values)
        coefs = _fit_ridge(X_tr, y_tr, w_tr)
        y_pred = X_te @ coefs
        erros.extend(np.abs(y_te - y_pred).tolist())
    return float(np.mean(erros)) if erros else float("inf")


# ─────────────────────────────────────────────────────────────
# Pipeline por grupo
# ─────────────────────────────────────────────────────────────

def _calibrar_grupo(df_grupo: pd.DataFrame, features: list, pesos_default: dict) -> dict:
    """Retorna dict com coefs, pesos normalizados e métricas, ou status insuficiente."""
    if df_grupo.empty or df_grupo["rodada"].nunique() < MIN_RODADAS or len(df_grupo) < MIN_REGISTROS:
        return {
            "status": "insuficiente",
            "n_rodadas": int(df_grupo["rodada"].nunique()) if not df_grupo.empty else 0,
            "n_registros": int(len(df_grupo)),
            "motivo": f"precisa ≥{MIN_RODADAS} rodadas e ≥{MIN_REGISTROS} registros",
        }

    df_norm = _normalizar(df_grupo)
    y       = df_grupo["pontuacao"].astype(float).values
    w       = _pesos_recencia(df_grupo["rodada"].values)
    X       = _monta_X(df_norm, features)
    coefs   = _fit_ridge(X, y, w)

    mae_in  = _mae(y, X @ coefs)
    mae_cv  = _cv_leave_one_round_out(df_grupo, df_norm, features)
    mae_h   = _mae_heuristica(df_norm, y, pesos_default, features)

    # Só adota se superar a heurística out-of-sample
    if mae_cv >= mae_h:
        return {
            "status": "insuficiente",
            "n_rodadas": int(df_grupo["rodada"].nunique()),
            "n_registros": int(len(df_grupo)),
            "motivo": f"modelo OOS MAE={mae_cv:.3f} não supera heurística={mae_h:.3f}",
            "metricas": {"mae_heuristica": round(mae_h, 3), "mae_modelo_cv": round(mae_cv, 3)},
        }

    # Normaliza os coeficientes das features (sem intercept) para soma=1,
    # preservando a magnitude relativa. Coef negativos viram 0 (feature
    # descartada porque não apóia a hipótese).
    feat_coefs = coefs[1:].copy()
    feat_coefs = np.clip(feat_coefs, 0, None)
    soma = feat_coefs.sum()
    pesos_norm = {
        f: round(float(feat_coefs[i] / soma), 4) if soma > 0 else pesos_default.get(f, 0.0)
        for i, f in enumerate(features)
    }

    return {
        "status": "ok",
        "n_rodadas": int(df_grupo["rodada"].nunique()),
        "n_registros": int(len(df_grupo)),
        "pesos": pesos_norm,
        "coefs_brutos": {f: round(float(coefs[i + 1]), 4) for i, f in enumerate(features)},
        "intercept": round(float(coefs[0]), 4),
        "metricas": {
            "mae_heuristica": round(mae_h, 3),
            "mae_modelo_cv":  round(mae_cv, 3),
            "mae_modelo_in":  round(mae_in, 3),
            "ganho_pct_cv":   round((mae_h - mae_cv) / mae_h * 100, 1) if mae_h > 0 else 0.0,
        },
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibrando score_confronto...")
    df = coletar_dataset()
    if df.empty:
        _salvar_insuficiente("sem snapshots com features brutas de confronto", n_rodadas=0, n_registros=0)
        return

    n_rod_total = df["rodada"].nunique()
    n_reg_total = len(df)
    print(f"  {n_rod_total} rodada(s) com features brutas, {n_reg_total} registros (jogou=True)")

    df_def = df[df["posicao"].isin(POSICOES_DEFESA)].copy()
    df_atk = df[df["posicao"].isin(POSICOES_ATAQUE)].copy()

    res_def = _calibrar_grupo(df_def, FEATURES_DEF, PESOS_SCORE_DEFAULT["defesa"])
    res_atk = _calibrar_grupo(df_atk, FEATURES_ATK, PESOS_SCORE_DEFAULT["ataque"])

    # Se pelo menos um grupo foi calibrado, salva status ok (consumidor faz
    # merge com defaults).
    status_final = "ok" if (res_def.get("status") == "ok" or res_atk.get("status") == "ok") else "insuficiente"

    pesos = {
        "defesa": res_def.get("pesos", PESOS_SCORE_DEFAULT["defesa"]),
        "ataque": res_atk.get("pesos", PESOS_SCORE_DEFAULT["ataque"]),
    }

    payload = {
        "status":        status_final,
        "n_rodadas":     int(n_rod_total),
        "n_registros":   int(n_reg_total),
        "pesos":         pesos,
        "grupos": {
            "defesa": res_def,
            "ataque": res_atk,
        },
        "hiperparams": {
            "ridge_lambda":  RIDGE_LAMBDA,
            "recency_decay": RECENCY_DECAY,
            "min_rodadas":   MIN_RODADAS,
            "min_registros": MIN_REGISTROS,
        },
        "atualizado_em": datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT"),
    }

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for grupo, res in (("defesa", res_def), ("ataque", res_atk)):
        if res.get("status") == "ok":
            m = res["metricas"]
            print(f"  [{grupo}] calibrado — MAE_cv {m['mae_modelo_cv']} vs heur {m['mae_heuristica']} "
                  f"(ganho {m['ganho_pct_cv']}%)")
        else:
            print(f"  [{grupo}] {res.get('motivo', 'insuficiente')}")
    print(f"  salvo em {OUT_PATH} (status={status_final})")


def _salvar_insuficiente(motivo: str, n_rodadas: int, n_registros: int) -> None:
    payload = {
        "status":        "insuficiente",
        "n_rodadas":     int(n_rodadas),
        "n_registros":   int(n_registros),
        "motivo":        motivo,
        "atualizado_em": datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT"),
    }
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [INSUFICIENTE] {motivo} — extractor usará PESOS_SCORE_DEFAULT")


if __name__ == "__main__":
    main()
