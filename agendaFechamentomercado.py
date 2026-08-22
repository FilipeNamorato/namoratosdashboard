import csv
import ast
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
HORAS_ANTES_LEMBRETE = int(os.environ.get("HORAS_ANTES_LEMBRETE", "3"))
TIMEZONE = "America/Sao_Paulo"

STATUS_MERCADO_ABERTO = "1"


def carregar_dados_mercado(caminho_csv: str) -> dict:
    with open(caminho_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return row


def extrair_fechamento(row: dict) -> datetime.datetime:
    fechamento_raw = row["fechamento"]
    fechamento_dict = ast.literal_eval(fechamento_raw)
    timestamp = int(fechamento_dict["timestamp"])
    return datetime.datetime.fromtimestamp(timestamp, tz=ZoneInfo(TIMEZONE))


def construir_servico():
    credentials_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not credentials_json:
        raise EnvironmentError("Variavel GOOGLE_SERVICE_ACCOUNT_JSON nao definida.")
    
    credentials_json = credentials_json.strip()  # remove espaços e quebras nas bordas


    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES
    )
    return build("calendar", "v3", credentials=credentials)


def buscar_eventos_existentes(service, rodada: str) -> list:
    summary_alvo = f"Cartola FC - Fechamento Rodada {rodada}"
    agora = datetime.datetime.now(tz=ZoneInfo(TIMEZONE))
    time_min = (agora - datetime.timedelta(days=1)).isoformat()
    time_max = (agora + datetime.timedelta(days=14)).isoformat()

    resultado = (
        service.events()
        .list(
            calendarId=CALENDAR_ID,
            timeMin=time_min,
            timeMax=time_max,
            q=summary_alvo,
            singleEvents=True,
        )
        .execute()
    )
    return resultado.get("items", [])


def apagar_eventos(service, eventos: list):
    for evento in eventos:
        service.events().delete(calendarId=CALENDAR_ID, eventId=evento["id"]).execute()
        print(f"Evento removido: {evento.get('summary')} ({evento.get('id')})")


def criar_evento_fechamento(service, dt_fechamento: datetime.datetime, rodada: str):
    summary = f"Cartola FC - Fechamento Rodada {rodada}"
    dt_fim = dt_fechamento + datetime.timedelta(minutes=30)

    evento = {
        "summary": summary,
        "description": (
            f"Mercado do Cartola FC fecha {dt_fechamento.strftime('%H:%M do dia %d/%m/%Y')}.\n"
            f"Rodada {rodada} - Faça sua escalacao antes do fechamento.\n\n"
            f"Lembrete automatico enviado pelo Google Calendar {HORAS_ANTES_LEMBRETE}h antes."
        ),
        "start": {
            "dateTime": dt_fechamento.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": dt_fim.isoformat(),
            "timeZone": TIMEZONE,
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},   # 1 dia antes
                {"method": "popup", "minutes": 4 * 60},    # 5 horas antes
                {"method": "popup", "minutes": 60},         # 1 hora antes

            ],
        },
        "colorId": "11",
    }

    resultado = (
        service.events().insert(calendarId=CALENDAR_ID, body=evento).execute()
    )
    print(f"Evento criado: {resultado.get('htmlLink')}")
    return resultado


def timestamp_mudou(eventos_existentes: list, dt_fechamento: datetime.datetime) -> bool:
    if not eventos_existentes:
        return True
    evento = eventos_existentes[0]
    start_existente = evento.get("start", {}).get("dateTime", "")
    if not start_existente:
        return True
    dt_existente = datetime.datetime.fromisoformat(start_existente)
    return dt_existente != dt_fechamento


def main():
    caminho_csv = os.environ.get("MERCADO_STATUS_CSV", "docs/data/current/status.csv")

    print(f"Lendo mercado status de: {caminho_csv}")
    row = carregar_dados_mercado(caminho_csv)
    
    
    status = str(row.get("status_mercado", "")).strip()
    rodada = str(row.get("rodada_atual", "?")).strip()
    game_over = str(row.get("game_over", "False")).strip()

    print(f"Rodada: {rodada} | Status mercado: {status} | Game over: {game_over}")

    if game_over.lower() == "true":
        print("Temporada encerrada (game_over=True). Nada a fazer.")
        sys.exit(0)

    if status != STATUS_MERCADO_ABERTO:
        print(f"Mercado nao esta aberto (status={status}). Nada a fazer.")
        sys.exit(0)

    dt_fechamento = extrair_fechamento(row)
    print(f"Fechamento previsto: {dt_fechamento.strftime('%d/%m/%Y %H:%M')}")

    agora = datetime.datetime.now(tz=ZoneInfo(TIMEZONE))
    if dt_fechamento < agora:
        print("Data de fechamento ja passou. Nada a fazer.")
        sys.exit(0)

    service = construir_servico()

    print(f"[DIAG] CALENDAR_ID usado: {CALENDAR_ID}")

    eventos_existentes = buscar_eventos_existentes(service, rodada)
    for ev in eventos_existentes:
        print(f"[DIAG] Evento encontrado: {ev.get('htmlLink')} (organizer={ev.get('organizer')})")

    if eventos_existentes and not timestamp_mudou(eventos_existentes, dt_fechamento):
        print("Evento ja existe com o mesmo horario. Nada a fazer.")
        sys.exit(0)

    if eventos_existentes:
        print(f"Timestamp mudou. Removendo {len(eventos_existentes)} evento(s) antigo(s).")
        apagar_eventos(service, eventos_existentes)

    criar_evento_fechamento(service, dt_fechamento, rodada)
    print("Concluido.")


if __name__ == "__main__":
    main()
