# streamlit_app.py
import os
from datetime import datetime, timedelta, date, time

import pandas as pd
import streamlit as st
from dateutil import tz

# === Google Sheets ===
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# (tenuto per futuri grafici)
import plotly.express as px  # noqa: F401

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Gestionale Ordini", layout="wide")

TZ = tz.gettz("Europe/Rome")

DATA_DIR = os.getenv("DATA_DIR", "data")        # solo per allegati locali
FILES_DIR = os.path.join(DATA_DIR, "files")

SHEET_NAME = "streamlit-gestionale"             # nome del file su Google Drive
WORKSHEET_TITLE = None                          # None = primo foglio

# 👇 aggiunta colonna 'completato'
COLUMNS = [
    "id", "titolo", "cliente", "materiale", "metri_lineari",
    "coeff", "ore_stimate", "inizio_iso", "fine_iso",
    "scadenza_iso", "note", "completato"
]

# Giorni/ore lavorative
WORK_START_HOUR = 8
WORK_END_HOUR = 16
WORK_DAYS = {0, 1, 2, 3, 4}  # lun=0 ... dom=6

# === Config dei coefficienti su Google Sheets ===
CONFIG_WS_TITLE = "config"  # worksheet con i coefficienti
MATERIAL_COEFF_DEFAULT = {
    "marmo": 0.20,
    "granito": 0.25,
    "ceramica": 0.12,
}

# Formato italiano per salvataggio su Sheets (senza secondi)
DATE_FMT_SHEET = "%d/%m/%Y %H:%M"


# ==============================
# STORAGE (allegati)
# ==============================
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)


def order_dir(order_id: int) -> str:
    path = os.path.join(FILES_DIR, str(order_id))
    os.makedirs(path, exist_ok=True)
    return path


def list_files(order_id: int):
    d = order_dir(order_id)
    files = []
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if os.path.isfile(full):
            files.append(full)
    return files


def save_uploaded_files(order_id: int, uploaded_files):
    if not uploaded_files:
        return
    d = order_dir(order_id)
    for f in uploaded_files:
        target = os.path.join(d, f.name)
        with open(target, "wb") as out:
            out.write(f.getbuffer())


# ==============================
# GOOGLE SHEETS HELPERS
# ==============================
def _load_service_info():
    """Credenziali da st.secrets o variabile env JSON."""
    if "gcp_service_account" in st.secrets:
        return dict(st.secrets["gcp_service_account"])
    else:
        import json
        return json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])


def _open_spreadsheet():
    info = _load_service_info()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open(SHEET_NAME)


def _open_ws():
    sh = _open_spreadsheet()
    return sh.get_worksheet(0) if WORKSHEET_TITLE is None else sh.worksheet(WORKSHEET_TITLE)


def _get_or_create_config_ws():
    sh = _open_spreadsheet()
    try:
        return sh.worksheet(CONFIG_WS_TITLE)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=CONFIG_WS_TITLE, rows=50, cols=3)
        ws.update("A1:B1", [["materiale", "coeff"]])
        rows = [[m, MATERIAL_COEFF_DEFAULT[m]] for m in MATERIAL_COEFF_DEFAULT]
        if rows:
            ws.update(f"A2:B{len(rows)+1}", rows)
        return ws


@st.cache_data(ttl=60, show_spinner=False)
def load_coeffs_from_sheet() -> dict:
    """Legge i coefficienti dal worksheet 'config' (o crea con default)."""
    ws = _get_or_create_config_ws()
    data = ws.get_all_records()
    coeffs = {}
    for row in data:
        m = str(row.get("materiale", "")).strip().lower()
        c = row.get("coeff", None)
        if m and c is not None:
            try:
                coeffs[m] = float(c)
            except Exception:
                pass
    if not coeffs:
        coeffs = MATERIAL_COEFF_DEFAULT.copy()
    return coeffs


def save_coeffs_to_sheet(coeffs: dict):
    """Salva i coefficienti nel worksheet 'config' e invalida la cache."""
    ws = _get_or_create_config_ws()
    ws.clear()
    ws.update("A1:B1", [["materiale", "coeff"]])
    rows = [[k, v] for k, v in coeffs.items()]
    if rows:
        ws.update(f"A2:B{len(rows)+1}", rows)
    load_coeffs_from_sheet.clear()  # invalida cache


def load_orders() -> pd.DataFrame:
    ws = _open_ws()
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
    if df.empty:
        # crea df vuoto con tutte le colonne, inclusa 'completato'
        empty = pd.DataFrame(columns=COLUMNS)
        empty["completato"] = empty.get("completato", pd.Series(dtype="boolean"))
        return empty

    # Tipi numerici
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    for c in ["metri_lineari", "coeff", "ore_stimate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Date: dayfirst=True per formati italiani (es. 31/01/2026 08:00)
    for col in ["inizio_iso", "fine_iso", "scadenza_iso"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Colonna 'completato'
    if "completato" not in df.columns:
        df["completato"] = False
    else:
        df["completato"] = (
            df["completato"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "si", "sì", "y", "yes"])
        )

    # Completa colonne mancanti e ordine
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    df["completato"] = df["completato"].astype("boolean")
    return df[COLUMNS]


def save_orders(df: pd.DataFrame):
    df2 = df.copy()

    # Date -> stringhe in formato italiano senza secondi
    for col in ["inizio_iso", "fine_iso", "scadenza_iso"]:
        if col in df2.columns:
            s = pd.to_datetime(df2[col], errors="coerce", dayfirst=True)
            df2[col] = s.dt.strftime(DATE_FMT_SHEET)

    # Booleani come TRUE/FALSE
    if "completato" in df2.columns:
        df2["completato"] = df2["completato"].map(lambda v: bool(v) if pd.notna(v) else False)

    # Colonne/ordine
    for c in COLUMNS:
        if c not in df2.columns:
            df2[c] = pd.NA
    df2 = df2[COLUMNS]

    ws = _open_ws()
    ws.clear()
    set_with_dataframe(ws, df2, include_index=False, include_column_header=True, resize=True)


# ==============================
# BUSINESS UTILS
# ==============================
def next_work_start(dt: datetime) -> datetime:
    dt_local = dt
    while dt_local.weekday() not in WORK_DAYS:
        dt_local = (dt_local + timedelta(days=1)).replace(
            hour=WORK_START_HOUR, minute=0, second=0, microsecond=0
        )

    start_today = dt_local.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    end_today = dt_local.replace(hour=WORK_END_HOUR, minute=0, second=0, microsecond=0)

    if dt_local < start_today:
        return start_today
    if dt_local >= end_today:
        nd = dt_local + timedelta(days=1)
        while nd.weekday() not in WORK_DAYS:
            nd += timedelta(days=1)
        return nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    return dt_local


def add_work_hours(start: datetime, hours: float) -> datetime:
    remaining = float(hours or 0)
    current = next_work_start(start)

    while remaining > 1e-9:
        work_end = current.replace(hour=WORK_END_HOUR, minute=0, second=0, microsecond=0)
        avail = (work_end - current).total_seconds() / 3600.0
        if avail <= 0:
            nd = current + timedelta(days=1)
            while nd.weekday() not in WORK_DAYS:
                nd += timedelta(days=1)
            current = nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
            continue

        if remaining <= avail + 1e-9:
            return current + timedelta(hours=remaining)

        remaining -= avail
        nd = current + timedelta(days=1)
        while nd.weekday() not in WORK_DAYS:
            nd += timedelta(days=1)
        current = nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)

    return current


def compute_estimate(materiale: str, ml: float) -> tuple[float, float]:
    """Usa sempre i coefficienti correnti dal worksheet 'config'."""
    coeffs = load_coeffs_from_sheet()
    if coeffs:
        coeff = coeffs.get(str(materiale).lower(), list(coeffs.values())[0])
    else:
        coeff = MATERIAL_COEFF_DEFAULT.get(str(materiale).lower(), 0.20)
    ore = round(coeff * float(ml or 0), 2)
    return ore, coeff


def combine_date_time(d: date, t: time, tzinfo) -> datetime:
    # ritorna naive; bastano giorno/ora (timezone non usata nei calcoli)
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def human(dtobj) -> str:
    try:
        if pd.isna(pd.Timestamp(dtobj)):
            return "-"
    except Exception:
        return "-"
    return pd.to_datetime(dtobj, dayfirst=True).strftime("%d/%m/%Y %H:%M")


def due_status_color(scadenza: pd.Timestamp, today: datetime) -> str:
    """
    Verde: >14 giorni
    Giallo: >4 giorni
    Rosso: <=4 giorni o scaduto
    """
    if pd.isna(scadenza):
        return "giallo"
    days = (pd.to_datetime(scadenza).normalize() - pd.Timestamp(today.date())).days
    if days > 14:
        return "verde"
    if days > 4:
        return "giallo"
    return "rosso"


# ==============================
# STATE INIT
# ==============================
ensure_storage()
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None


# ==============================
# SIDEBAR (menu pagine)
# ==============================
st.sidebar.title("📁 Navigazione")
page = st.sidebar.selectbox("Pagina", ["Dashboard", "Aggiungi ordine", "Impostazioni"])


# ==============================
# PAGINA: IMPOSTAZIONI
# ==============================
if page == "Impostazioni":
    st.title("Impostazioni")
    st.subheader("⏱️ Coefficienti orari per materiale (h per metro lineare)")

    coeffs = load_coeffs_from_sheet().copy()

    st.write("Modifica i valori e premi **Salva**. Puoi anche aggiungere o togliere materiali.")
    new_coeffs = {}

    # Editor materiali esistenti
    for m, c in coeffs.items():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.text_input("Materiale", value=m, key=f"m_{m}", disabled=True)
        with col2:
            new_c = st.number_input("Coeff (h/ml)", min_value=0.0, step=0.01, value=float(c), key=f"c_{m}")
        with col3:
            keep = st.checkbox("Mantieni", value=True, key=f"keep_{m}")
        if keep:
            new_coeffs[m] = float(new_c)

    st.markdown("---")
    st.write("**Aggiungi nuovo materiale**")
    colA, colB = st.columns([2, 2])
    with colA:
        add_m = st.text_input("Nome materiale (minuscolo consigliato)", key="add_m").strip().lower()
    with colB:
        add_c = st.number_input("Coeff (h/ml) nuovo", min_value=0.0, step=0.01, value=0.20, key="add_c")

    colS1, colS2 = st.columns([1, 1])
    with colS1:
        if st.button("➕ Aggiungi materiale"):
            if add_m:
                new_coeffs[add_m] = float(add_c)
                save_coeffs_to_sheet(new_coeffs)
                st.success(f"Materiale '{add_m}' aggiunto/aggiornato.")
                st.rerun()

    with colS2:
        if st.button("💾 Salva modifiche"):
            save_coeffs_to_sheet(new_coeffs)
            st.success("Coefficienti salvati.")
            st.rerun()

    st.markdown("---")
    st.subheader("🔁 (Opzionale) Ricalcola le *fine previste*")
    st.caption("Aggiorna **solo** le fine previste (e facoltativamente le ore stimate); non tocca le scadenze.")
    recalc_ore = st.checkbox("Aggiorna anche 'ore_stimate' = coeff(materiale) × metri_lineari", value=False)

    if st.button("Ricalcola per tutti gli ordini"):
        df = load_orders()

        def recalc_row(row):
            try:
                start_dt = pd.to_datetime(row["inizio_iso"], errors="coerce", dayfirst=True)
                materiale = str(row["materiale"]).lower()
                coeff_map = load_coeffs_from_sheet()
                coeff = coeff_map.get(materiale, MATERIAL_COEFF_DEFAULT.get(materiale, 0.20))
                ml = float(row.get("metri_lineari", 0) or 0)
                ore = float(row.get("ore_stimate", 0) or 0)
                if recalc_ore:
                    ore = round(coeff * ml, 2)
                    row["ore_stimate"] = ore
                    row["coeff"] = coeff
                new_end = add_work_hours(start_dt, ore)
                row["fine_iso"] = new_end
            except Exception:
                pass
            return row

        df = df.apply(recalc_row, axis=1)
        save_orders(df)
        st.success("Ricalcolo completato.")
        st.rerun()

    st.stop()  # evita che si disegni altro


# ==============================
# PAGINE: DASHBOARD / AGGIUNGI
# ==============================
df_all = load_orders().sort_values("inizio_iso", ascending=False)

if page == "Aggiungi ordine":
    st.title("➕ Nuovo ordine")
    with st.form("add_order", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            titolo = st.text_input("Titolo ordine*", placeholder="Davanzali lotto 12")
            cliente = st.text_input("Cliente*", placeholder="Impresa Rossi srl")
            # Materiali = chiavi coeff correnti
            material_options = list(load_coeffs_from_sheet().keys()) or list(MATERIAL_COEFF_DEFAULT.keys())
            materiale = st.selectbox("Materiale*", options=material_options)
        with col2:
            ml = st.number_input("Metri lineari*", min_value=0.0, step=0.5)
            d_inizio = st.date_input("Data inizio lavori*", value=date.today())
            t_inizio = st.time_input("Ora inizio lavori*", value=time(8, 0))

        col_scad1, col_scad2 = st.columns(2)
        with col_scad1:
            d_scad = st.date_input("Data scadenza*", value=date.today())
        with col_scad2:
            t_scad = st.time_input("Ora scadenza*", value=time(18, 0))

        note = st.text_area("Note", placeholder="Annotazioni interne...")
        files = st.file_uploader("Allega file (opzionale)", accept_multiple_files=True)

        submitted = st.form_submit_button("Crea ordine", use_container_width=True)

        if submitted:
            if not titolo or not cliente or ml <= 0:
                st.error("Compila i campi obbligatori e inserisci metri lineari > 0.")
            else:
                start_dt = combine_date_time(d_inizio, t_inizio, TZ)
                ore, coeff = compute_estimate(materiale, ml)
                end_dt = add_work_hours(start_dt, ore)
                scadenza_dt = combine_date_time(d_scad, t_scad, TZ)

                df_now = load_orders()
                new_id = 1 if df_now.empty else int(pd.to_numeric(df_now["id"], errors="coerce").max() or 0) + 1

                new_row = {
                    "id": new_id,
                    "titolo": titolo,
                    "cliente": cliente,
                    "materiale": materiale,
                    "metri_lineari": ml,
                    "coeff": coeff,
                    "ore_stimate": ore,
                    "inizio_iso": start_dt,
                    "fine_iso": end_dt,
                    "scadenza_iso": scadenza_dt,
                    "note": note,
                    "completato": False,  # parte aperto
                }

                df_now = pd.concat([df_now, pd.DataFrame([new_row])], ignore_index=True)
                save_orders(df_now)
                save_uploaded_files(new_id, files)

                st.success(f"Ordine #{new_id} creato. Fine prevista: {human(end_dt)} • Scadenza: {human(scadenza_dt)}")
                st.session_state.selected_id = new_id
                st.rerun()

else:
    # ======== DASHBOARD ========
    st.title("Gestionale Ordini")
    st.subheader("📅 Scadenze (solo giorni con ordini)")

    if df_all.empty:
        st.info("Nessun ordine presente. Vai su 'Aggiungi ordine'.")
    else:
        today = datetime.now(TZ)

        # Filtri globali dashboard
        colflt1, colflt2 = st.columns(2)
        with colflt1:
            show_past = st.toggle("Mostra anche scadenze passate", value=False)
        with colflt2:
            show_done = st.toggle("Mostra anche ordini eseguiti", value=False)

        # Prepara scadenze (fallback alla fine prevista), evitando ri-parse
        cal = df_all.copy()
        if not show_done:
            cal = cal[~cal["completato"].fillna(False)]

        cal["scadenza_data"] = cal["scadenza_iso"].copy()
        fallback = cal["fine_iso"].copy()
        cal.loc[cal["scadenza_data"].isna(), "scadenza_data"] = fallback
        cal = cal.dropna(subset=["scadenza_data"]).copy()

        cal["scadenza_day"] = pd.to_datetime(cal["scadenza_data"], errors="coerce", dayfirst=True).dt.date
        cal["status"] = cal["scadenza_data"].apply(lambda d: due_status_color(pd.to_datetime(d, dayfirst=True), today))

        if not show_past:
            cal = cal[cal["scadenza_day"] >= today.date()]

        if cal.empty:
            st.info("Non ci sono scadenze nel periodo/selezione.")
        else:
            # Aggrega per giorno: status più severo + conteggio + df ordini
            rank = {"rosso": 3, "giallo": 2, "verde": 1}
            day_info = {}
            for d, g in cal.groupby("scadenza_day"):
                stt = max(g["status"], key=lambda s: rank.get(s, 0))
                day_info[d] = {"status": stt, "count": len(g), "df": g.sort_values("scadenza_data")}

            days_list = sorted(day_info.keys())

            # Paginazione
            PAGE_SIZE = 12
            if "day_page" not in st.session_state:
                st.session_state.day_page = 0
            max_page = max((len(days_list) - 1) // PAGE_SIZE, 0)

            cprev, cinfo, cnext = st.columns([1, 3, 1])
            with cprev:
                if st.button("◀︎", use_container_width=True, disabled=(st.session_state.day_page == 0)):
                    st.session_state.day_page -= 1
                    st.rerun()
            with cinfo:
                st.markdown(
                    f"<div style='text-align:center;opacity:.8'>Pagina {st.session_state.day_page+1} / {max_page+1}</div>",
                    unsafe_allow_html=True
                )
            with cnext:
                if st.button("▶︎", use_container_width=True, disabled=(st.session_state.day_page == max_page)):
                    st.session_state.day_page += 1
                    st.rerun()

            start = st.session_state.day_page * PAGE_SIZE
            end = start + PAGE_SIZE
            page_days = days_list[start:end]

            # Stili “chip”
            st.markdown("""
            <style>
            .chip{border:1px solid rgba(255,255,255,.12); border-radius:999px; padding:8px 12px; margin:6px 6px 0 0;
                    display:inline-flex; align-items:center; gap:8px;}
            .dot{width:10px;height:10px;border-radius:50%;}
            .green{background:#2ca02c;} .yellow{background:#ffbf00;} .red{background:#d62728;}
            .badge{font-size:.8rem; opacity:.85;}
            </style>
            """, unsafe_allow_html=True)

            # RIGA DI CHIP (solo bottoni)
            chip_area = st.container()
            with chip_area:
                for d in page_days:
                    info = day_info[d]
                    stt = info["status"]
                    icon = "🟢" if stt == "verde" else "🟡" if stt == "giallo" else "🔴"

                    giorni = {
                        "Mon": "Lunedì", "Tue": "Martedì", "Wed": "Mercoledì",
                        "Thu": "Giovedì", "Fri": "Venerdì", "Sat": "Sabato", "Sun": "Domenica"
                    }
                    giorno = giorni.get(pd.Timestamp(d).strftime("%a"), pd.Timestamp(d).strftime("%a"))

                    n = info["count"]
                    ordini_txt = "ordine" if n == 1 else "ordini"
                    label = f"{icon} {giorno} {pd.Timestamp(d).strftime('%d/%m')} — {n} {ordini_txt}"

                    if st.button(label, key=f"chip_{pd.Timestamp(d).date().isoformat()}", use_container_width=True):
                        st.session_state.selected_day = d
                        st.rerun()

            # Lista ordini del giorno selezionato
            if "selected_day" not in st.session_state and page_days:
                st.session_state.selected_day = page_days[0]

            st.markdown(f"### 📌 Ordini del {pd.Timestamp(st.session_state.selected_day).strftime('%d/%m/%Y')}")
            todays = day_info.get(st.session_state.selected_day, {}).get("df", pd.DataFrame())

            if todays is None or todays.empty:
                st.info("Nessun ordine in questa data.")
            else:
                # Vista tabellare con colonne formattate in italiano
                show = todays[["id", "titolo", "cliente", "scadenza_iso", "fine_iso", "completato"]].copy()
                show.rename(columns={"scadenza_iso": "Scadenza", "fine_iso": "Fine prevista", "titolo": "Titolo",
                                     "cliente": "Cliente", "id": "ID", "completato": "Completato"}, inplace=True)
                show["Titolo"] = show.apply(lambda r: ("✅ " if bool(r["Completato"]) else "") + str(r["Titolo"]), axis=1)
                show["Scadenza"] = show["Scadenza"].apply(human)
                show["Fine prevista"] = show["Fine prevista"].apply(human)

                st.dataframe(show[["ID", "Titolo", "Cliente", "Scadenza", "Fine prevista", "Completato"]],
                             use_container_width=True, hide_index=True)

                # Azioni rapide per ogni ordine del giorno
                cols_open = st.columns(min(4, len(todays)))
                for idx, (_, rowx) in enumerate(todays.iterrows()):
                    c = cols_open[idx % len(cols_open)]
                    with c:
                        stt = due_status_color(rowx["scadenza_data"], today)
                        icon = "🟢" if stt == "verde" else "🟡" if stt == "giallo" else "🔴"
                        oid = int(rowx["id"])
                        b1 = st.button(f"{icon} Apri #{oid}", key=f"open_{oid}", use_container_width=True)
                        done = bool(rowx.get("completato", False))
                        action_label = ("↩︎ Riattiva" if done else "✅ Completa") + f" #{oid}"
                        b2 = st.button(action_label, key=f"toggle_done_{oid}", use_container_width=True)
                        if b1:
                            st.session_state.selected_id = oid
                            st.session_state["order_selector"] = str(oid)  # sync radio
                            st.query_params = {**st.query_params, "selected_id": str(oid)}
                            st.rerun()
                        if b2:
                            odf2 = load_orders()
                            odf2.loc[odf2["id"] == oid, "completato"] = (not done)
                            save_orders(odf2)
                            st.success(f"Ordine #{oid} {'riattivato' if done else 'completato'}.")
                            st.rerun()

    st.markdown("---")

    # ===== Sidebar: selettore ordine =====
    st.sidebar.markdown("### Seleziona ordine")

    if df_all.empty:
        st.sidebar.info("Nessun ordine presente.")
        st.session_state.selected_id = None
    else:
        options = [str(int(x)) for x in df_all["id"].dropna().astype(int)]

        def label_for_row(row):
            base = f"#{int(row.id)} — {row.titolo} | {row.ore_stimate}h • fine: {human(row.fine_iso)}"
            return ("✅ " + base) if bool(row.get("completato", False)) else base

        labels = [label_for_row(row) for _, row in df_all.iterrows()]

        if st.session_state.get("selected_id") is None and options:
            st.session_state.selected_id = int(options[0])

        if str(st.session_state.selected_id) not in options and options:
            st.session_state.selected_id = int(options[0])

        if "order_selector" not in st.session_state:
            st.session_state.order_selector = str(st.session_state.selected_id)

        selected_str = st.sidebar.radio(
            "Ordini",
            options=options,
            format_func=lambda v: labels[options.index(v)],
            key="order_selector"
        )
        st.session_state.selected_id = int(selected_str)

    # ===== Dettaglio =====
    st.subheader("🔎 Dettaglio ordine")
    if st.session_state.selected_id is None:
        st.info("Seleziona un ordine dalla sidebar per visualizzarlo qui.")
    else:
        odf = load_orders()
        row = odf[odf["id"] == st.session_state.selected_id]
        if row.empty:
            st.warning("Ordine non trovato (forse è stato rimosso).")
        else:
            r = row.iloc[0]
            c1, c2 = st.columns([2, 1])
            with c1:
                badge = " ✅" if bool(r.get("completato", False)) else ""
                st.markdown(f"### #{int(r['id'])} — {r['titolo']}{badge}")
                st.write(f"**Cliente:** {r['cliente']}")
                st.write(f"**Materiale:** {r['materiale']}  |  **ML:** {r['metri_lineari']}")
                st.write(f"**Ore stimate:** {r['ore_stimate']} (coeff: {r['coeff']} h/ml)")
                st.write(f"**Inizio lavori:** {human(r['inizio_iso'])}")
                st.write(f"**Fine prevista:** {human(r['fine_iso'])}")
                st.write(f"**Scadenza:** {human(r['scadenza_iso'])}")
                if isinstance(r.get("note", ""), str) and str(r["note"]).strip():
                    st.write("**Note:**")
                    st.write(r["note"])

            with c2:
                st.write("**Azioni**")
                # Toggle completato
                done_now = st.toggle("Segna come completato", value=bool(r.get("completato", False)))
                if done_now != bool(r.get("completato", False)):
                    odf.loc[odf["id"] == r["id"], "completato"] = done_now
                    save_orders(odf)
                    st.success(f"Ordine #{int(r['id'])} {'completato' if done_now else 'riattivato'}.")
                    st.rerun()

                if st.button("Ricalcola fine (regole attuali)", use_container_width=True):
                    start_dt = pd.to_datetime(r["inizio_iso"], errors="coerce", dayfirst=True)
                    ore = float(r["ore_stimate"] or 0)
                    new_end = add_work_hours(start_dt, ore)
                    odf.loc[odf["id"] == r["id"], "fine_iso"] = new_end
                    save_orders(odf)
                    st.success(f"Fine aggiornata: {human(new_end)}")
                    st.rerun()

                with st.form(f"edit_deadline_{int(r['id'])}"):
                    st.write("Aggiorna scadenza")
                    current_scad = pd.to_datetime(r["scadenza_iso"], errors="coerce", dayfirst=True)
                    d_scad = st.date_input(
                        "Data scadenza",
                        value=(current_scad.date() if pd.notna(current_scad) else date.today())
                    )
                    t_scad = st.time_input(
                        "Ora scadenza",
                        value=(current_scad.time() if pd.notna(current_scad) else time(18, 0))
                    )
                    if st.form_submit_button("Salva scadenza", use_container_width=True):
                        scad_dt = combine_date_time(d_scad, t_scad, TZ)
                        odf.loc[odf["id"] == r["id"], "scadenza_iso"] = scad_dt
                        save_orders(odf)
                        st.success(f"Scadenza aggiornata: {human(scad_dt)}")
                        st.rerun()

            st.markdown("#### 📎 Allegati")
            files = list_files(int(r["id"]))
            if not files:
                st.info("Nessun file allegato.")
            else:
                for fp in files:
                    name = os.path.basename(fp)
                    with open(fp, "rb") as f:
                        data = f.read()
                    st.download_button(
                        label=f"Scarica {name}",
                        data=data,
                        file_name=name,
                        mime=None,
                        use_container_width=True
                    )

            st.markdown("#### ➕ Aggiungi altri allegati")
            up = st.file_uploader("Carica file aggiuntivi", accept_multiple_files=True, key=f"up_{r['id']}")
            if up:
                save_uploaded_files(int(r["id"]), up)
                st.success("File caricati.")
                st.rerun()
