import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date, time
from dateutil import tz
import os
import io

# =============== CONFIG ===============
st.set_page_config(page_title="Gestionale Ordini", layout="wide")

# in cima, vicino a DATA_DIR
DATA_DIR = os.getenv("DATA_DIR", "data")   # <— invece del valore fisso
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")
FILES_DIR = os.path.join(DATA_DIR, "files")


TZ = tz.gettz("Europe/Rome")

# Coefficienti ore per materiale (modifica liberamente)
MATERIAL_COEFF = {
    "marmo": 0.20,     # ore per metro lineare
    "granito": 0.25,
    "ceramica": 0.12,
}

WORK_START_HOUR = 8
WORK_END_HOUR = 16
WORK_HOURS_PER_DAY = WORK_END_HOUR - WORK_START_HOUR
WORK_DAYS = {0, 1, 2, 3, 4}  # lun=0 ... dom=6

# =============== UTILS ===============
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)
    if not os.path.exists(ORDERS_CSV):
        df = pd.DataFrame(columns=[
            "id","titolo","cliente","materiale","metri_lineari",
            "coeff","ore_stimate","inizio_iso","fine_iso","note"
        ])
        df.to_csv(ORDERS_CSV, index=False)

def load_orders() -> pd.DataFrame:
    ensure_storage()
    df = pd.read_csv(ORDERS_CSV)
    # parsing comodo
    for col in ["inizio_iso","fine_iso"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def save_orders(df: pd.DataFrame):
    df2 = df.copy()

    for col in ["inizio_iso", "fine_iso"]:
        if col in df2.columns:
            # 1) forza il tipo datetime (accetta mixed/strings/NaT)
            s = pd.to_datetime(df2[col], errors="coerce")

            # 2) se è timezone-aware, converti a Europe/Rome e rimuovi tz
            try:
                if s.dt.tz is not None:
                    s = s.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
            except AttributeError:
                # s.dt non disponibile se tutta la colonna è NaT → ok
                pass

            df2[col] = s

    # 3) salva in formato stringa uniforme
    for col in ["inizio_iso", "fine_iso"]:
        if col in df2.columns:
            df2[col] = df2[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    df2.to_csv(ORDERS_CSV, index=False)


def next_work_start(dt: datetime) -> datetime:
    """Riposiziona dt all'inizio del prossimo slot lavorativo (oggi 08:00 se possibile, altrimenti prossimo giorno utile)."""
    dt_local = dt
    # weekend -> salta a lun 08:00
    while dt_local.weekday() not in WORK_DAYS:
        dt_local = (dt_local + timedelta(days=1)).replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    # fascia oraria
    start_today = dt_local.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    end_today = dt_local.replace(hour=WORK_END_HOUR, minute=0, second=0, microsecond=0)
    if dt_local < start_today:
        return start_today
    if dt_local >= end_today:
        # prossimo giorno lavorativo 08:00
        nd = dt_local + timedelta(days=1)
        while nd.weekday() not in WORK_DAYS:
            nd += timedelta(days=1)
        return nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    return dt_local

def add_work_hours(start: datetime, hours: float) -> datetime:
    """Somma 'hours' alle ore lavorative (lun–ven, 8–16)."""
    remaining = hours
    current = next_work_start(start)

    while remaining > 1e-9:
        # bordo giorno corrente
        work_end = current.replace(hour=WORK_END_HOUR, minute=0, second=0, microsecond=0)
        # ore disponibili oggi
        avail = (work_end - current).total_seconds() / 3600.0
        if avail <= 0:
            # salta a prossimo giorno lavorativo 08:00
            nd = current + timedelta(days=1)
            while nd.weekday() not in WORK_DAYS:
                nd += timedelta(days=1)
            current = nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
            continue

        if remaining <= avail + 1e-9:
            return current + timedelta(hours=remaining)

        # consuma tutta la giornata
        remaining -= avail
        # vai al prossimo giorno lavorativo 08:00
        nd = current + timedelta(days=1)
        while nd.weekday() not in WORK_DAYS:
            nd += timedelta(days=1)
        current = nd.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)

    return current

def compute_estimate(materiale: str, ml: float) -> float:
    coeff = MATERIAL_COEFF.get(materiale.lower(), list(MATERIAL_COEFF.values())[0])
    return round(coeff * ml, 2), coeff

def human(dt: datetime) -> str:
    if pd.isna(pd.Timestamp(dt)):
        return "-"
    return dt.strftime("%d/%m/%Y %H:%M")

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
        # preserva nome
        target = os.path.join(d, f.name)
        with open(target, "wb") as out:
            out.write(f.getbuffer())


def combine_date_time(d: date, t: time, tzinfo) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)  # niente tz




# =============== STATE INIT ===============
ensure_storage()
if "view" not in st.session_state:
    st.session_state.view = "ordini"   # "ordini" | "aggiungi"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

# =============== SIDEBAR ===============

# =============== SIDEBAR ===============
st.sidebar.title("📁 Ordini")

df = load_orders().sort_values("inizio_iso", ascending=False)

# Opzioni della sidebar: "new" per aggiunta
options = ["new"] + [str(int(x)) for x in df["id"]] if not df.empty else ["new"]
labels = ["➕ Aggiungi nuovo…"] + [
    f"#{int(row.id)} — {row.titolo} | {row.ore_stimate}h • fine: {human(row.fine_iso)}"
    for _, row in df.iterrows()
]

selected_str = st.sidebar.radio("Seleziona", options=options, format_func=lambda v: labels[options.index(v)])

if selected_str == "new":
    st.session_state.view = "aggiungi"
    st.session_state.selected_id = None
else:
    st.session_state.view = "ordini"
    st.session_state.selected_id = int(selected_str)

# filtro/ricerca veloce (facoltativo)
with st.sidebar.expander("🔎 Filtra"):
    q = st.text_input("Cerca per titolo/cliente...")
    if q:
        df = df[
            (df["titolo"].str.contains(q, case=False, na=False))
            | (df["cliente"].str.contains(q, case=False, na=False))
        ]

# =============== MAIN ===============
st.title("Gestionale Ordini")

if st.session_state.view == "aggiungi":
    st.subheader("➕ Nuovo ordine")
    with st.form("add_order", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            titolo = st.text_input("Titolo ordine*", placeholder="Davanzali lotto 12")
            cliente = st.text_input("Cliente*", placeholder="Impresa Rossi srl")
            materiale = st.selectbox("Materiale*", options=list(MATERIAL_COEFF.keys()))
        with col2:
            ml = st.number_input("Metri lineari*", min_value=0.0, step=0.5)
            d_inizio = st.date_input("Data inizio lavori*", value=date.today())
            t_inizio = st.time_input("Ora inizio lavori*", value=time(8, 0))
            note = st.text_area("Note", placeholder="Annotazioni interne...")

        files = st.file_uploader("Allega file (opzionale)", accept_multiple_files=True)

        # >>>>>>  IMPORTANTISSIMO: pulsante di submit dentro il form  <<<<<<
        submitted = st.form_submit_button("Crea ordine", use_container_width=True)

        if submitted:
            if not titolo or not cliente or ml <= 0:
                st.error("Compila i campi obbligatori e inserisci metri lineari > 0.")
            else:
                start_dt = combine_date_time(d_inizio, t_inizio, TZ)
                ore, coeff = compute_estimate(materiale, ml)
                end_dt = add_work_hours(start_dt, ore)

                df_all = load_orders()
                new_id = 1 if df_all.empty else int(df_all["id"].max()) + 1

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
                    "note": note,
                }
                df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)
                save_orders(df_all)

                save_uploaded_files(new_id, files)

                st.success(f"Ordine #{new_id} creato. Fine prevista: {human(end_dt)}")
                st.session_state.view = "ordini"
                st.session_state.selected_id = new_id
                st.rerun()

else:
    # Vista elenco + dettaglio
    st.subheader("📋 Elenco ordini")
    if df.empty:
        st.info("Nessun ordine presente. Aggiungine uno dalla sidebar.")
    else:
        # Tabella compatta
        show = df.copy()
        show["inizio"] = show["inizio_iso"].apply(human)
        show["fine_prevista"] = show["fine_iso"].apply(human)
        show = show[["id","titolo","cliente","materiale","metri_lineari","ore_stimate","inizio","fine_prevista"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("---")
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
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"### #{int(r['id'])} — {r['titolo']}")
                st.write(f"**Cliente:** {r['cliente']}")
                st.write(f"**Materiale:** {r['materiale']}  |  **ML:** {r['metri_lineari']}")
                st.write(f"**Ore stimate:** {r['ore_stimate']} (coeff: {r['coeff']} h/ml)")
                st.write(f"**Inizio lavori:** {human(r['inizio_iso'])}")
                st.write(f"**Fine prevista:** {human(r['fine_iso'])}")
                if isinstance(r.get("note", ""), str) and r["note"].strip():
                    st.write("**Note:**")
                    st.write(r["note"])
            with c2:
                st.write("**Azioni**")
                # Ricalcola fine (es. se cambi coeff tabella in futuro)
                if st.button("Ricalcola fine con regole attuali", use_container_width=True):
                    start_dt = pd.to_datetime(r["inizio_iso"]).to_pydatetime().replace(tzinfo=TZ)
                    ore = float(r["ore_stimate"])
                    new_end = add_work_hours(start_dt, ore)
                    odf.loc[odf["id"] == r["id"], "fine_iso"] = new_end
                    save_orders(odf)
                    st.success(f"Fine aggiornata: {human(new_end)}")
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
