# app.py
# Dashboard Streamlit pour le projet de detection d'anomalies IoT (LSTM)
# Consomme l'API FastAPI (main.py) qui doit tourner sur http://127.0.0.1:8000
#
# Habillage visuel inspire d'une maquette fintech (fond gris clair, cartes
# blanches arrondies, accent orange corail, typographie geometrique).
# Seuls la couleur, la typographie et la mise en page ont change : la
# logique metier et les appels API sont identiques a la version precedente.
#
# Pour lancer :
#   1) demarrer l'API      : uvicorn api.main:app --reload
#   2) demarrer le dashboard: python -m streamlit run app.py

import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# ----------------------------------------------------------------------------
# Configuration generale
# ----------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 260  # secondes

st.set_page_config(
    page_title="ThermoWatch — Surveillance IoT",
    page_icon="◐",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Design tokens (palette / typographie / rayon des cartes)
# ----------------------------------------------------------------------------

COLOR_BG = "#0D0D10"
COLOR_CARD = "#18181C"
COLOR_SURFACE_2 = "#212127"
COLOR_SIDEBAR_FROM = "#111114"
COLOR_SIDEBAR_TO = "#08080A"
COLOR_INK = "#F5F5F6"
COLOR_INK_SOFT = "#96969E"
COLOR_ACCENT = "#FF6B3D"
COLOR_ACCENT_DEEP = "#D8420F"
COLOR_ACCENT_SOFT = "rgba(255,107,61,0.16)"
COLOR_SUCCESS = "#34D399"
COLOR_SUCCESS_SOFT = "rgba(52,211,153,0.14)"
COLOR_DANGER = "#FF6B6E"
COLOR_DANGER_SOFT = "rgba(255,107,110,0.14)"
COLOR_BORDER = "rgba(255,255,255,0.08)"
RADIUS = "18px"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    color: {COLOR_INK};
}}

h1, h2, h3, h4 {{
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
    color: {COLOR_INK} !important;
}}

.kpi-value {{
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    font-variant-numeric: tabular-nums;
    color: {COLOR_INK} !important;
}}

/* garde-fou global : les titres et le texte principal ne doivent jamais
   heriter d'un blanc/clair sur fond blanc (bug de contraste corrige) */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label {{
    color: {COLOR_INK};
}}
.kpi-card:not(.accent) .kpi-value {{
    color: {COLOR_INK} !important;
}}

/* fond general de l'app : leger halo chaleureux en haut de page, pour
   eviter l'aplat gris uniforme */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1100px 420px at 12% -8%, {COLOR_ACCENT_SOFT} 0%, rgba(255,107,61,0) 60%),
        {COLOR_BG};
}}
[data-testid="stHeader"] {{
    background-color: transparent;
}}
[data-testid="stMain"] .block-container {{
    padding-top: 2.2rem;
}}

/* accroche de page : titre + trait degrade signature sous l'eyebrow */
.page-eyebrow {{
    color: {COLOR_INK_SOFT};
    font-weight: 500;
    margin-top: -6px;
    margin-bottom: 4px;
}}
[data-testid="stMain"] h1:first-of-type {{
    margin-bottom: 0.15rem;
}}
.page-eyebrow::after {{
    content: "";
    display: block;
    width: 46px;
    height: 3px;
    margin-top: 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, {COLOR_ACCENT}, {COLOR_ACCENT_DEEP});
}}

/* barre laterale : rail noir avec degrade subtil, plus de relief */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLOR_SIDEBAR_FROM} 0%, {COLOR_SIDEBAR_TO} 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{
    color: #F4F4F4 !important;
}}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{
    color: #85858C !important;
}}

/* navigation en pilules dans la sidebar, avec halo au survol/actif */
[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 11px 16px !important;
    margin-bottom: 8px;
    transition: background-color 0.15s ease, transform 0.15s ease;
    border: 1px solid rgba(255,255,255,0.05);
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background-color: rgba(255,255,255,0.10);
    transform: translateX(2px);
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(135deg, {COLOR_ACCENT}, {COLOR_ACCENT_DEEP});
    border-color: transparent;
    box-shadow: 0 6px 16px rgba(255,107,61,0.4);
}}

/* cartes generiques (st.container(border=True)) : relief + accroche degrade */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {COLOR_CARD};
    border-radius: {RADIUS};
    border: 1px solid {COLOR_BORDER};
    padding: 10px 8px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 10px 24px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {COLOR_ACCENT}, {COLOR_ACCENT_DEEP});
    opacity: 0.85;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 16px 32px rgba(0,0,0,0.55);
}}

/* boutons */
.stButton > button {{
    border-radius: 999px;
    font-weight: 600;
    border: none;
    padding: 0.6em 1.6em;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLOR_SURFACE_2}, #2B2B31);
    color: #FFFFFF;
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, {COLOR_ACCENT}, {COLOR_ACCENT_DEEP});
    color: #FFFFFF;
    box-shadow: 0 8px 20px rgba(255,107,61,0.35);
    transform: translateY(-1px);
}}

/* champs de saisie */
div[data-baseweb="select"] > div, .stNumberInput input {{
    border-radius: 12px !important;
    background-color: {COLOR_SURFACE_2} !important;
    border: 1px solid {COLOR_BORDER} !important;
    color: {COLOR_INK} !important;
}}
div[data-baseweb="select"] > div:focus-within {{
    border-color: {COLOR_ACCENT} !important;
    box-shadow: 0 0 0 3px {COLOR_ACCENT_SOFT} !important;
}}
div[data-baseweb="popover"] li {{
    background-color: {COLOR_SURFACE_2} !important;
    color: {COLOR_INK} !important;
}}
div[data-baseweb="popover"] li:hover {{
    background-color: {COLOR_ACCENT_SOFT} !important;
}}

/* tableau de donnees : en-tete teinte pour ancrer visuellement les colonnes */
[data-testid="stDataFrame"] {{
    border-radius: {RADIUS};
    overflow: hidden;
    border: 1px solid {COLOR_BORDER};
}}
[data-testid="stDataFrame"] [data-testid="stHeader"],
[data-testid="stDataFrame"] thead tr th {{
    background-color: {COLOR_ACCENT_SOFT} !important;
    color: {COLOR_INK} !important;
    font-weight: 600 !important;
}}

/* alertes streamlit natives : on assombrit le fond pastel par defaut pour
   rester dans le langage visuel noir, tout en gardant le texte colore natif */
div[data-testid="stAlert"] {{
    border-radius: 14px;
    border: 1px solid {COLOR_BORDER};
    background-color: {COLOR_SURFACE_2} !important;
}}
div[data-testid="stAlert"] p {{
    color: {COLOR_INK} !important;
}}

/* cartes KPI custom : chip icone + accroche degrade + relief au survol */
.kpi-card {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS};
    padding: 22px 22px 20px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 10px 22px rgba(0,0,0,0.45);
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.kpi-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {COLOR_ACCENT}, {COLOR_ACCENT_DEEP});
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 30px rgba(0,0,0,0.55);
}}
.kpi-card.accent {{
    background: linear-gradient(150deg, {COLOR_ACCENT} 0%, {COLOR_ACCENT_DEEP} 100%);
    border: none;
    color: #FFFFFF;
}}
.kpi-card.accent::before {{
    background: rgba(255,255,255,0.35);
}}
.kpi-card.accent .kpi-label, .kpi-card.accent .kpi-delta {{
    color: #FFE6DA !important;
}}
.kpi-icon {{
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background-color: {COLOR_ACCENT_SOFT};
    color: {COLOR_ACCENT_DEEP};
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 14px;
}}
.kpi-card.accent .kpi-icon {{
    background-color: rgba(255,255,255,0.18);
    color: #FFFFFF;
}}
.kpi-label {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {COLOR_INK_SOFT};
    text-transform: none;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 2.05rem;
    line-height: 1.1;
    margin-bottom: 6px;
}}
.kpi-delta {{
    font-size: 0.8rem;
    font-weight: 600;
}}
.kpi-delta.positive {{ color: {COLOR_SUCCESS}; }}
.kpi-delta.negative {{ color: {COLOR_DANGER}; }}
.kpi-card.accent .kpi-delta.positive, .kpi-card.accent .kpi-delta.negative {{
    color: #FFE6DA;
}}

/* banniere de statut (normal / anomalie), avec chip iconique sans emoji */
.status-banner {{
    border-radius: {RADIUS};
    padding: 16px 20px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    font-size: 1.05rem;
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid transparent;
}}
.status-banner::before {{
    content: "";
    flex: 0 0 auto;
    width: 10px;
    height: 10px;
    border-radius: 999px;
}}
.status-banner.ok {{
    background-color: {COLOR_SUCCESS_SOFT};
    color: #0F7A54;
    border-color: rgba(31,169,113,0.25);
}}
.status-banner.ok::before {{
    background-color: {COLOR_SUCCESS};
    box-shadow: 0 0 0 4px rgba(31,169,113,0.18);
}}
.status-banner.ko {{
    background-color: {COLOR_DANGER_SOFT};
    color: #B4232A;
    border-color: rgba(229,72,77,0.25);
}}
.status-banner.ko::before {{
    background-color: {COLOR_DANGER};
    box-shadow: 0 0 0 4px rgba(229,72,77,0.18);
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# template plotly assorti a la palette
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color=COLOR_INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[COLOR_ACCENT, COLOR_INK],
        xaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
        yaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
    )
)


def kpi_card(label, value, delta_text=None, delta_positive=True, accent=False):
    """Construit le HTML d'une carte KPI dans le style de la maquette."""
    delta_html = ""
    if delta_text:
        sign_class = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="kpi-delta {sign_class}">{arrow} {delta_text}</div>'
    card_class = "kpi-card accent" if accent else "kpi-card"
    return f"""
    <div class="{card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# ----------------------------------------------------------------------------
# Fonctions d'appel a l'API, avec gestion propre des erreurs
# ----------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_devices():
    """Recupere la liste des devices. Mise en cache pour eviter des appels
    repetes a chaque interaction utilisateur."""
    try:
        response = requests.get(f"{API_BASE_URL}/devices", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def get_alerts(method="lstm_anomaly"):
    """Recupere l'historique des alertes (non mis en cache : on veut des
    donnees fraiches sur la page d'accueil)."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/alerts",
            params={"method": method},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def get_alert_episodes(method="lstm_anomaly"):
    """Recupere les anomalies regroupees en EPISODES (blocs de minutes
    consecutives), plutot qu'une ligne par minute flag individuelle.
    Un episode de 100 minutes d'affile compte pour 1, pas pour 100 —
    c'est ce chiffre-la qui doit s'afficher sur la carte KPI, pas le
    decompte brut (qui gonfle a plusieurs milliers a cause du bruit
    statistique du seuil percentile 90, cf notes projet)."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/alerts/episodes",
            params={"method": method},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def get_history(device_id, limit=300):
    """Recupere l'historique d'un device donne."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/devices/{device_id}/history",
            params={"limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def post_predict_demo(device_id):
    """Declenche une prediction demo pour un device."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/devices/{device_id}/predict_demo",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def api_unreachable_message(error_detail):
    st.error(
        "Impossible de contacter l'API FastAPI sur "
        f"`{API_BASE_URL}`.\n\n"
        "Verifiez qu'elle est bien lancee (`uvicorn api.main:app --reload`)."
    )
    with st.expander("Details techniques de l'erreur"):
        st.code(error_detail)


# ----------------------------------------------------------------------------
# Page 1 : Vue Globale & Alertes
# ----------------------------------------------------------------------------

def page_vue_globale():
    st.title("Vue globale")
    st.markdown('<p class="page-eyebrow">Etat du parc de capteurs et derniers episodes d\'anomalie detectes.</p>', unsafe_allow_html=True)

    devices, devices_err = get_devices()
    if devices_err:
        api_unreachable_message(devices_err)
        return

    # episodes = anomalies regroupees en incidents (plutot que le
    # decompte brut de minutes, qui gonfle a plusieurs milliers)
    episodes, episodes_err = get_alert_episodes(method="lstm_anomaly")
    if episodes_err:
        api_unreachable_message(episodes_err)
        return

    # Sécurité au cas où l'API renvoie null
    devices = devices or []
    episodes = episodes or []

    nb_devices = len(devices)
    nb_episodes = len(episodes)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(kpi_card("Appareils surveilles", nb_devices), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Episodes d'anomalie detectes", nb_episodes, accent=True), unsafe_allow_html=True)

    st.caption(
        "Un episode regroupe des minutes flag consecutives (meme incident) en un seul "
        "evenement, plutot que de compter chaque minute individuellement."
    )

    st.write("")
    with st.container(border=True):
        st.subheader("Derniers episodes d'anomalie")

        if nb_episodes == 0:
            st.info("Aucun episode d'anomalie detecte pour le moment.")
            return

        episodes_df = pd.DataFrame(episodes)

        colonnes_dispo = [
            c for c in ["device_id", "debut", "fin", "duree_minutes", "temperature_max", "temperature_min"]
            if c in episodes_df.columns
        ]
        episodes_df = episodes_df[colonnes_dispo].copy()

        rename_map = {
            "device_id": "Appareil",
            "debut": "Début",
            "fin": "Fin",
            "duree_minutes": "Durée (min)",
            "temperature_max": "Temp. max (°C)",
            "temperature_min": "Temp. min (°C)",
        }
        episodes_df.rename(columns=rename_map, inplace=True)

        if "Début" in episodes_df.columns:
            episodes_df["Début"] = pd.to_datetime(episodes_df["Début"])
            episodes_df = episodes_df.sort_values("Début", ascending=False)
        if "Fin" in episodes_df.columns:
            episodes_df["Fin"] = pd.to_datetime(episodes_df["Fin"])
        if "Temp. max (°C)" in episodes_df.columns:
            episodes_df["Temp. max (°C)"] = episodes_df["Temp. max (°C)"].round(1)
        if "Temp. min (°C)" in episodes_df.columns:
            episodes_df["Temp. min (°C)"] = episodes_df["Temp. min (°C)"].round(1)

        st.dataframe(episodes_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Page 2 : Analyse Historique
# ----------------------------------------------------------------------------

def page_analyse_historique():
    st.title("Analyse historique")
    st.markdown('<p class="page-eyebrow">Historique de temperature d\'un appareil, anomalies LSTM en surbrillance.</p>', unsafe_allow_html=True)

    devices, devices_err = get_devices()
    if devices_err:
        api_unreachable_message(devices_err)
        return

    device_ids = [d["device_id"] for d in devices]
    if not device_ids:
        st.warning("Aucun appareil disponible.")
        return

    with st.container(border=True):
        col_select, col_limit = st.columns([2, 1])
        with col_select:
            selected_device = st.selectbox("Choisir un appareil", device_ids)
        with col_limit:
            limit = st.number_input("Nombre de points (limit)", min_value=50, max_value=5000, value=300, step=50)

    history, history_err = get_history(selected_device, limit=int(limit))
    if history_err:
        api_unreachable_message(history_err)
        return

    if not history:
        st.warning(f"Pas de donnees disponibles pour l'appareil {selected_device}.")
        return

    df = pd.DataFrame(history)

    if "Time" not in df.columns or "Temperature" not in df.columns:
        st.error("La reponse de l'API ne contient pas les colonnes attendues (Time, Temperature).")
        return

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Temperature"],
            mode="lines",
            name="Temperature",
            line=dict(color=COLOR_INK, width=1.8),
        )
    )

    if "lstm_anomaly" in df.columns:
        anomalies_df = df[df["lstm_anomaly"] == 1]
        fig.add_trace(
            go.Scatter(
                x=anomalies_df["Time"],
                y=anomalies_df["Temperature"],
                mode="markers",
                name="Anomalie LSTM",
                marker=dict(color=COLOR_ACCENT, size=9, symbol="circle", line=dict(color="#FFFFFF", width=1)),
            )
        )
        nb_anomalies_device = len(anomalies_df)
    else:
        st.warning("La colonne 'lstm_anomaly' est absente de la reponse de l'API : pas de marqueurs affiches.")
        nb_anomalies_device = 0

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Temperature — {selected_device}",
        xaxis_title="Temps",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{nb_anomalies_device} anomalie(s) LSTM detectee(s) sur les {len(df)} points affiches.")


# ----------------------------------------------------------------------------
# Page 3 : Demo Temps Reel
# ----------------------------------------------------------------------------

def page_demo_temps_reel():
    st.title("Démo temps réel")
    st.markdown('<p class="page-eyebrow">Simule une prediction a partir des 60 dernieres minutes d\'historique.</p>', unsafe_allow_html=True)

    devices, devices_err = get_devices()
    if devices_err:
        api_unreachable_message(devices_err)
        return

    device_ids = [d["device_id"] for d in devices]
    if not device_ids:
        st.warning("Aucun appareil disponible.")
        return

    with st.container(border=True):
        selected_device = st.selectbox("Choisir un appareil", device_ids)
        launch = st.button("Lancer la prediction", type="primary")

    if not launch:
        return

    with st.spinner("Appel du modele en cours..."):
        result, result_err = post_predict_demo(selected_device)

    if result_err:
        api_unreachable_message(result_err)
        return

    st.write("")
    st.subheader("Résultat de la prédiction")

    ecart = result["prediction_temperature"] - result["valeur_reelle"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(kpi_card("Prediction (°C)", f"{result['prediction_temperature']:.2f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Valeur reelle (°C)", f"{result['valeur_reelle']:.2f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(
            kpi_card("Ecart brut (°C)", f"{ecart:+.2f}", delta_positive=(ecart <= 0)),
            unsafe_allow_html=True,
        )

    col4, col5 = st.columns(2)
    with col4:
        st.markdown(kpi_card("Erreur normalisee", f"{result['erreur_normalisee']:.4f}"), unsafe_allow_html=True)
    with col5:
        st.markdown(kpi_card("Seuil du device", f"{result['seuil_device']:.4f}", accent=True), unsafe_allow_html=True)

    st.write("")
    if result["anomaly"]:
        st.markdown(f'<div class="status-banner ko">Anomalie détectée sur {selected_device}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-banner ok">Comportement normal pour {selected_device}</div>', unsafe_allow_html=True)

    with st.expander("Reponse brute de l'API"):
        st.json(result)


# ----------------------------------------------------------------------------
# Navigation laterale
# ----------------------------------------------------------------------------

def main():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:24px;">
                <div style="background:{COLOR_ACCENT}; width:34px; height:34px; border-radius:10px;
                            display:flex; align-items:center; justify-content:center;
                            color:#FFFFFF; font-weight:600; font-size:15px;">TW</div>
                <div style="font-family:-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
                            font-weight:600; font-size:1.05rem; letter-spacing:-0.01em;">
                    ThermoWatch
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = {
            "Vue Globale & Alertes": page_vue_globale,
            "Analyse Historique": page_analyse_historique,
            "Démo Temps Réel": page_demo_temps_reel,
        }
        selection = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        st.caption(f"API : {API_BASE_URL}")
        st.caption("Detection d'anomalies IoT — LSTM")

    pages[selection]()


if __name__ == "__main__":
    main()