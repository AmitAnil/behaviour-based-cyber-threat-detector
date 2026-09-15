"""
Behaviour-Based Cyber Threat Detector - Streamlit dashboard
PS07 (Software / Applied Blockchain & Cybersecurity) - MeitY / CERT-In

Run locally:
    pip install -r requirements.txt
    python src/generate_data.py     # only needed once, or to regenerate
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detection_engine import run_pipeline, get_supporting_events, BASELINE_DAYS  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (validated categorical + status roles - see dataviz design notes)
# ---------------------------------------------------------------------------
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
INK = "#0b0b0b"
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

st.set_page_config(page_title="Behaviour-Based Cyber Threat Detector", page_icon="\U0001F6E1️", layout="wide")

PAGE_ICONS = {
    "Overview Dashboard": "\U0001F6E1️",
    "Alert Drill-Down": "\U0001F50E",
    "User Baseline Profiles": "\U0001F464",
    "Normal vs Anomalous Demo": "\U0001F3AD",
    "Methodology": "\U0001F4D8",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }

/* Hero banner (Overview page) */
.hero-banner {
  background: linear-gradient(135deg, #2a78d6 0%, #184f95 100%);
  border-radius: 18px;
  padding: 2rem 2.3rem;
  margin-bottom: 1.8rem;
  box-shadow: 0 10px 28px rgba(24, 79, 149, 0.22);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(255,255,255,0.16); color: #ffffff;
  padding: 0.3rem 0.85rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;
  text-transform: uppercase; margin-bottom: 0.9rem;
}
.hero-banner h1 { color: #ffffff; font-size: 2.15rem; font-weight: 800; margin: 0 0 0.5rem 0; letter-spacing: -0.02em; }
.hero-banner p { color: rgba(255,255,255,0.90); font-size: 1.03rem; margin: 0; max-width: 820px; line-height: 1.55; }

/* Plain page header (all other pages) */
.page-header { display: flex; align-items: flex-start; gap: 0.85rem; margin-bottom: 1.6rem; }
.page-header .icon {
  font-size: 1.6rem; background: #eaf1fb; border-radius: 12px;
  width: 48px; height: 48px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.page-header h1 { font-size: 1.65rem; font-weight: 800; margin: 0; letter-spacing: -0.01em; color:#0b0b0b; }
.page-header p { margin: 0.25rem 0 0 0; color: #52514e; font-size: 0.96rem; max-width: 820px; line-height:1.5; }

/* Stat tile */
.stat-card {
  background: #ffffff; border: 1px solid #e1e0d9; border-top: 4px solid var(--accent, #2a78d6);
  border-radius: 12px; padding: 1.05rem 1.15rem 0.95rem 1.15rem;
  box-shadow: 0 1px 3px rgba(11,11,11,0.045);
  transition: box-shadow .15s ease, transform .15s ease;
}
.stat-card:hover { box-shadow: 0 8px 18px rgba(11,11,11,0.09); transform: translateY(-1px); }
.stat-label { font-size: 0.76rem; color: #52514e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
.stat-value { font-size: 2.1rem; font-weight: 800; color: #0b0b0b; line-height: 1; }
.stat-sub { font-size: 0.78rem; color: #898781; margin-top: 0.35rem; }

/* Bordered section containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px !important; box-shadow: 0 1px 4px rgba(11,11,11,0.05); }
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #f6f5f2; border-right: 1px solid #e1e0d9; }
section[data-testid="stSidebar"] label { font-weight: 500; }

/* Headings + metrics */
h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }
div[data-testid="stMetricValue"] { font-weight: 800; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def stat_card(label, value, accent="#2a78d6", sub=None):
    sub_html = f"<div class='stat-sub'>{sub}</div>" if sub else ""
    html = (
        f'<div class="stat-card" style="--accent:{accent}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'{sub_html}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def page_header(page_name, subtitle):
    html = (
        '<div class="page-header">'
        f'<div class="icon">{PAGE_ICONS.get(page_name, "")}</div>'
        f'<div><h1>{page_name}</h1><p>{subtitle}</p></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner="Ingesting logs, learning baselines, scoring deviations...")
def load():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(os.path.join(data_dir, "auth_logs.csv")):
        raise FileNotFoundError("Synthetic logs not found - run `python src/generate_data.py` first.")
    return run_pipeline(data_dir)


def base_layout(fig, height=380, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRIDLINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def severity_badge(sev):
    color = STATUS[sev]
    return f"<span style='background:{color}22;color:{color};border:1px solid {color}66;padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.85em'>{sev}</span>"


# ---------------------------------------------------------------------------
try:
    data = load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

auth, files, net, users_df = data["auth"], data["files"], data["net"], data["users"]
alerts = data["alerts"].copy()
baselines = data["baselines"]
cutoff, window_end = data["cutoff"], data["window_end"]
alerts["date_str"] = alerts["date"].dt.strftime("%Y-%m-%d")
user_lookup = users_df.set_index("user_id")[["name", "department", "home_city", "home_country"]].to_dict("index")
alerts["name"] = alerts["user_id"].map(lambda u: user_lookup.get(u, {}).get("name", u))
alerts["department"] = alerts["user_id"].map(lambda u: user_lookup.get(u, {}).get("department", "-"))

st.sidebar.markdown(
    "<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.1rem'>"
    "<span style='font-size:1.5rem'>\U0001F6E1️</span>"
    "<span style='font-size:1.25rem;font-weight:800'>Threat Detector</span></div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("PS07 — Behaviour-Based Cyber Threat Detector · MeitY / CERT-In")
page = st.sidebar.radio("Navigate", [
    "Overview Dashboard", "Alert Drill-Down", "User Baseline Profiles", "Normal vs Anomalous Demo", "Methodology",
], format_func=lambda p: f"{PAGE_ICONS.get(p, '')}  {p}")
st.sidebar.divider()
st.sidebar.markdown(
    "<div style='background:#ffffff;border:1px solid #e1e0d9;border-radius:12px;padding:0.9rem 1rem;font-size:0.85rem;line-height:1.7'>"
    f"<b>Baseline period</b><br>{auth['date'].min().date()} → {(cutoff - pd.Timedelta(days=1)).date()}<br>"
    f"<b>Monitoring window</b><br>{cutoff.date()} → {window_end.date()}<br>"
    f"<b>Users monitored</b><br>{users_df.shape[0]}"
    "</div>",
    unsafe_allow_html=True,
)

# ===========================================================================
if page == "Overview Dashboard":
    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-badge">\U0001F6E1️ PS07 · MeitY / CERT-In</div>'
        '<h1>Behaviour-Based Cyber Threat Detector</h1>'
        '<p>Statistical + rule-based baseline-and-deviation engine over synthetic authentication, '
        'file-access, and network-activity logs. Every alert is fully explainable.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stat_card("Total alerts", len(alerts), accent="#2a78d6")
    with c2:
        stat_card("Critical", int((alerts["severity"] == "Critical").sum()), accent=STATUS["Critical"])
    with c3:
        stat_card("High", int((alerts["severity"] == "High").sum()), accent=STATUS["High"])
    with c4:
        stat_card("Medium", int((alerts["severity"] == "Medium").sum()), accent=STATUS["Medium"])
    with c5:
        stat_card("Low", int((alerts["severity"] == "Low").sum()), accent=STATUS["Low"])

    st.write("")
    fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 2])
    sev_filter = fcol1.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    user_filter = fcol2.multiselect("User", sorted(alerts["user_id"].unique()), default=[])
    min_score = fcol3.slider("Minimum anomaly score", 0, 100, 0, 1)

    view = alerts[alerts["severity"].isin(sev_filter) & (alerts["anomaly_score"] >= min_score)]
    if user_filter:
        view = view[view["user_id"].isin(user_filter)]
    view = view.sort_values("anomaly_score", ascending=False)

    st.write("")
    left, right = st.columns([1.6, 1])
    with left:
        with st.container(border=True):
            st.subheader("Alerts")
            show = view[["user_id", "name", "department", "date_str", "anomaly_score", "severity", "num_reasons", "devices"]].rename(
                columns={"user_id": "User", "name": "Name", "department": "Dept", "date_str": "Date",
                         "anomaly_score": "Score", "severity": "Severity", "num_reasons": "# Signals", "devices": "Device(s)"}
            )

            def style_sev(v):
                c = STATUS.get(v, MUTED)
                return f"background-color:{c}22;color:{c};font-weight:600"

            st.dataframe(
                show.style.map(style_sev, subset=["Severity"]).format({"Score": "{:.1f}"}),
                use_container_width=True, height=430, hide_index=True,
            )
            st.caption(f"Showing {len(view)} of {len(alerts)} user-day alerts (score ≥ 12 shown; near-zero-deviation days are suppressed).")

    with right:
        with st.container(border=True):
            st.subheader("Alert score timeline")
            fig = go.Figure()
            for sev in SEVERITY_ORDER:
                d = view[view["severity"] == sev]
                fig.add_trace(go.Scatter(
                    x=d["date_str"], y=d["anomaly_score"], mode="markers", name=sev,
                    marker=dict(size=11, color=STATUS[sev], line=dict(width=1, color="#ffffff")),
                    text=d["user_id"], hovertemplate="%{text} — %{y:.1f}<br>%{x}<extra>" + sev + "</extra>",
                ))
            fig.update_yaxes(title="Anomaly score", range=[0, 105])
            fig.update_xaxes(title="Date")
            base_layout(fig, height=397)
            st.plotly_chart(fig, use_container_width=True)

    st.write("")
    with st.container(border=True):
        st.subheader("Severity mix by user")
        pivot = alerts.groupby(["user_id", "severity"]).size().unstack(fill_value=0)
        for s in SEVERITY_ORDER:
            if s not in pivot.columns:
                pivot[s] = 0
        pivot = pivot[SEVERITY_ORDER]
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig2 = go.Figure()
        for sev in SEVERITY_ORDER:
            fig2.add_trace(go.Bar(x=pivot.index, y=pivot[sev], name=sev, marker_color=STATUS[sev]))
        fig2.update_layout(barmode="stack")
        fig2.update_xaxes(title="User")
        fig2.update_yaxes(title="Alert-days")
        base_layout(fig2, height=320)
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
elif page == "Alert Drill-Down":
    page_header("Alert Drill-Down", "Pick an alert to see exactly which raw events triggered it — full supporting evidence, not a black-box score.")

    if alerts.empty:
        st.info("No alerts in the monitoring window.")
        st.stop()

    options = alerts.sort_values("anomaly_score", ascending=False).apply(
        lambda r: f"{r['anomaly_score']:.1f} [{r['severity']}] — {r['user_id']} ({r['name']}) on {r['date_str']}", axis=1
    ).tolist()
    idx_map = alerts.sort_values("anomaly_score", ascending=False).index.tolist()
    choice = st.selectbox("Alert", options)
    row = alerts.loc[idx_map[options.index(choice)]]

    user_id, date = row["user_id"], row["date"]
    bl = baselines[user_id]

    st.markdown(f"### {row['name']} ({user_id}) — {row['department']} — {row['date_str']} &nbsp; {severity_badge(row['severity'])}", unsafe_allow_html=True)

    card_col, score_col = st.columns([2.3, 1])
    with card_col:
        with st.container(border=True):
            st.subheader("Why this was flagged")
            for r in row["reasons"]:
                st.markdown(f"- {r}")
    with score_col:
        stat_card("Anomaly score", f"{row['anomaly_score']:.1f} <span style='font-size:1.1rem;color:#898781'>/ 100</span>", accent=STATUS[row["severity"]], sub=f"Severity: {row['severity']}")

    a, f, n = get_supporting_events(auth, files, net, user_id, date)

    st.write("")
    with st.container(border=True):
        st.subheader("Baseline vs. this day")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Logins", len(a))
        m2.metric("File events", len(f), delta=f"{len(f) - bl['daily_file_count_mean']:.0f} vs baseline")
        m3.metric("File volume (MB)", f"{f['file_size_mb'].sum():.0f}", delta=f"{f['file_size_mb'].sum() - bl['daily_file_size_mean']:.0f} vs baseline")
        m4.metric("Network egress (MB)", f"{n['bytes_mb'].sum():.0f}", delta=f"{n['bytes_mb'].sum() - bl['daily_net_mb_mean']:.0f} vs baseline")

        comp_fig = go.Figure()
        metrics = ["File events", "File volume (MB)", "Network egress (MB)"]
        baseline_vals = [bl["daily_file_count_mean"], bl["daily_file_size_mean"], bl["daily_net_mb_mean"]]
        actual_vals = [len(f), f["file_size_mb"].sum(), n["bytes_mb"].sum()]
        comp_fig.add_trace(go.Bar(name="Personal baseline (avg/day)", x=metrics, y=baseline_vals, marker_color=SEQ_BLUE[2]))
        comp_fig.add_trace(go.Bar(name="This day", x=metrics, y=actual_vals, marker_color=STATUS[row["severity"]]))
        comp_fig.update_layout(barmode="group")
        base_layout(comp_fig, height=340)
        st.plotly_chart(comp_fig, use_container_width=True)

    st.write("")
    st.subheader("Supporting events")
    t1, t2, t3 = st.tabs([f"Authentication ({len(a)})", f"File access ({len(f)})", f"Network activity ({len(n)})"])
    with t1:
        if a.empty:
            st.caption("No authentication events this day.")
        else:
            st.dataframe(a[["timestamp", "device_id", "city", "country", "source_ip"]], use_container_width=True, hide_index=True)
            known = bl["known_devices"]
            unknown_devs = set(a["device_id"]) - known
            if unknown_devs:
                st.warning(f"Unrecognized device(s) not seen during baseline: {', '.join(unknown_devs)}")
            known_locs = bl["known_locations"]
            unknown_locs = set(zip(a["city"], a["country"])) - known_locs
            if unknown_locs:
                st.warning(f"Location(s) never seen during baseline: {', '.join(f'{c}, {k}' for c, k in unknown_locs)}")
    with t2:
        if f.empty:
            st.caption("No file-access events this day.")
        else:
            fdisp = f[["timestamp", "device_id", "resource", "action", "file_size_mb"]].copy()
            fdisp["new_resource"] = ~fdisp["resource"].isin(bl["known_resources"])
            st.dataframe(fdisp, use_container_width=True, hide_index=True)
            n_new = int(fdisp["new_resource"].sum())
            if n_new:
                st.warning(f"{n_new} of these {len(fdisp)} events touched a resource never seen during baseline.")
    with t3:
        if n.empty:
            st.caption("No network events this day.")
        else:
            ndisp = n[["timestamp", "device_id", "dest_domain", "bytes_mb", "protocol"]].copy()
            ndisp["new_domain"] = ~ndisp["dest_domain"].isin(bl["known_domains"])
            st.dataframe(ndisp, use_container_width=True, hide_index=True)
            n_new = int(ndisp["new_domain"].sum())
            if n_new:
                st.warning(f"{n_new} connection(s) went to a domain never seen during baseline.")

# ===========================================================================
elif page == "User Baseline Profiles":
    page_header("User Baseline Profiles", f"The learned 'normal' for each user, built only from the first {BASELINE_DAYS} days of logs — this is what every later day is compared against.")

    uid = st.selectbox("User", sorted(baselines.keys()), format_func=lambda u: f"{u} — {user_lookup.get(u,{}).get('name',u)} ({user_lookup.get(u,{}).get('department','-')})")
    bl = baselines[uid]
    info = user_lookup.get(uid, {})

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Typical login hour", f"{bl['login_hour_mean']:.1f}h", sub=f"± {bl['login_hour_std']:.1f}h std. dev.")
    with c2:
        stat_card("Typical file volume/day", f"{bl['daily_file_count_mean']:.0f}", sub=f"events · {bl['daily_file_size_mean']:.0f} MB")
    with c3:
        stat_card("Typical network egress/day", f"{bl['daily_net_mb_mean']:.0f}", sub="MB")

    st.write("")
    with st.container(border=True):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Known devices**")
            st.write(", ".join(sorted(bl["known_devices"])) or "—")
            st.markdown("**Known locations**")
            st.write(", ".join(f"{c}, {k}" for c, k in sorted(bl["known_locations"])) or "—")
        with colB:
            st.markdown("**Known network domains**")
            st.write(", ".join(sorted(bl["known_domains"])) or "—")
            st.markdown(f"**Home base:** {info.get('home_city','-')}, {info.get('home_country','-')} ({info.get('department','-')} dept.)")

        hist = auth[(auth["user_id"] == uid) & (auth["date"] < cutoff)]
        hours = (hist["timestamp"].dt.hour + hist["timestamp"].dt.minute / 60.0)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=hours, nbinsx=24, marker_color=SEQ_BLUE[3], name="Baseline logins"))
        fig.update_xaxes(title="Hour of day", range=[0, 24])
        fig.update_yaxes(title="Login count")
        base_layout(fig, height=300, legend=False)
        st.subheader("Login-hour distribution (baseline period)")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
elif page == "Normal vs Anomalous Demo":
    page_header("Normal vs Anomalous Demo", "Two contrasting, deliberately injected scenarios that showcase both halves of the detection method.")

    gt_path = os.path.join(os.path.dirname(__file__), "data", "ground_truth.csv")
    gt = pd.read_csv(gt_path, parse_dates=["timestamp"]) if os.path.exists(gt_path) else pd.DataFrame()

    tabA, tabB = st.tabs(["Scenario A — Impossible Travel + Exfiltration", "Scenario B — Gradual Behavioural Drift"])

    with tabA:
        st.markdown(
            "**Story:** an account is compromised. Minutes after a legitimate morning login, the same "
            "account logs in again — this time from a city on another continent, on a device never seen "
            "before. Within 25 minutes it downloads dozens of files it has never touched, several of them "
            "sensitive, and pushes an 800MB+ transfer to a domain nobody in the company has ever contacted. "
            "**Every one of these is a hard rule violation on its own; together they are unambiguous.**"
        )
        a_rows = alerts[alerts["severity"] == "Critical"].sort_values("anomaly_score", ascending=False)
        if not a_rows.empty:
            top = a_rows.iloc[0]
            st.success(f"Detected: **{top['user_id']} ({top['name']})** on **{top['date_str']}** — score **{top['anomaly_score']:.0f}/100**, severity **{top['severity']}**")
            for r in top["reasons"]:
                st.markdown(f"- {r}")
        if not gt.empty:
            st.markdown("**Injected ground truth events:**")
            st.dataframe(gt[gt["scenario"] == "A"][["timestamp", "type", "detail"]], use_container_width=True, hide_index=True)
        st.info("**Detection method used:** rule-based (unrecognized device, impossible travel speed via haversine "
                "distance/time, never-seen domain, never-seen resources) reinforced by statistical z-scores on "
                "file-count / file-volume / network-volume, all against that user's own 30-day baseline.")

    with tabB:
        st.markdown(
            "**Story:** no new device, no new country, no single 'smoking gun' event. Instead, over four "
            "consecutive nights, the same user's login time creeps later (10pm → 1am → 2am → 3am+) and "
            "file-access volume climbs (1.4x → 1.9x → 2.6x → 3.4x normal). **A static rule engine keyed to "
            "specific thresholds or new locations would miss this entirely** — it only stands out as a "
            "statistical deviation from *this specific user's* personal baseline, accumulating night over night."
        )
        b_uid = gt[gt["scenario"] == "B"]["user_id"].iloc[0] if not gt.empty else None
        if b_uid:
            b_rows = alerts[alerts["user_id"] == b_uid].sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=b_rows["date_str"], y=b_rows["anomaly_score"], mode="lines+markers",
                                      line=dict(color=CATEGORICAL[1], width=3), marker=dict(size=10, color=[STATUS[s] for s in b_rows["severity"]]),
                                      name=f"{b_uid} anomaly score"))
            fig.update_yaxes(title="Anomaly score", range=[0, 105])
            fig.update_xaxes(title="Date")
            base_layout(fig, height=320, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                b_rows[["date_str", "anomaly_score", "severity", "num_reasons"]].rename(
                    columns={"date_str": "Date", "anomaly_score": "Score", "severity": "Severity", "num_reasons": "# Signals"}),
                use_container_width=True, hide_index=True,
            )
            latest = b_rows.sort_values("date").iloc[-1]
            st.markdown("**Signals on the worst day of the drift:**")
            for r in latest["reasons"]:
                st.markdown(f"- {r}")
        if not gt.empty:
            st.markdown("**Injected ground truth events:**")
            st.dataframe(gt[gt["scenario"] == "B"][["timestamp", "type", "detail"]], use_container_width=True, hide_index=True)
        st.info("**Detection method used:** purely statistical (z-score of login hour and daily file/network volume "
                "against the user's own baseline mean/std) — no rule fires here, illustrating why baseline-and-"
                "deviation catches what static thresholds cannot.")

# ===========================================================================
elif page == "Methodology":
    page_header("Methodology", "How the baseline-and-deviation engine turns raw logs into an explainable anomaly score.")
    st.markdown(f"""
### 1. Ingest
Three synthetic log streams are ingested: **authentication** (who logged in, when, from where, on which
device), **file-access** (which resource, what action, how large), and **network-activity** (destination
domain, bytes transferred). See *Data & Schema* in the README for exact columns.

### 2. Establish a per-user/device baseline
For each user, the **first {BASELINE_DAYS} days** of logs are treated as a training/baseline period (never
touched by scoring, so a baseline can't be poisoned by the very anomaly it should catch). From it we learn:

- typical login-hour **mean and standard deviation**
- the **set** of previously-used devices, locations (city/country), file resources, and network domains
- typical **daily volumes** (file-access count, total file MB, total network MB) as mean and standard deviation

### 3. Score deviations for every user, every day after the baseline
Two complementary signal families are combined into a single **0-100 anomaly score**:

| Signal type | Examples | Why |
|---|---|---|
| Rule-based (hard) | unrecognized device, physically-impossible travel speed (haversine distance / time between two logins &gt; commercial flight speed), traffic to a never-seen domain, access to resources never touched before | These should essentially never happen for a genuine, un-compromised user — any occurrence is inherently suspicious regardless of how "big" it is statistically |
| Statistical (soft) | z-score of the day's login hour, file-access count, file-access volume (MB), and network egress (MB) against that user's own baseline mean/std | Catches gradual drift, unusual-for-this-person timing, and volume spikes that no fixed threshold would generalize across a diverse user population |

Each signal contributes a capped number of points; the total is clipped to [0, 100]. Severity bands:
**Critical ≥ 75, High ≥ 50, Medium ≥ 25, Low &gt; 0** (below ~12 total points, a day is treated as
normal and not surfaced as an alert at all, to keep the queue analyst-usable).

### 4. Explainability + supporting events
Every alert carries a **plain-English list of exactly which signals fired** (not just a number), and the
*Alert Drill-Down* page lets you pull the **raw log rows** behind any alert — the actual logins, file
accesses, and network connections from that user on that day — with the specific unrecognized
device/location/resource/domain highlighted.

### 5. Why per-user baselines instead of one global rule set?
A night-shift IT admin logging in at 2am is normal *for them*; the same login for a 9-to-5 marketing
analyst is a strong anomaly. Baselining **per individual** (not per role or globally) is what lets the
engine catch subtle, personalized deviations — see the "gradual drift" scenario in the demo page —
while a shared static rule set would need to be so loose it misses that case, or so strict it constantly
false-positives on legitimate night-shift workers.

### Edge cases handled
- **New user with no baseline yet** → excluded from scoring until they accumulate history (no false alarms from insufficient data).
- **Weekend / low-activity days** → modelled with lower activity probability during data generation; an inactive day produces no alert (nothing to score) rather than a spurious "volume drop" anomaly.
- **Users with multiple registered devices** → all registered devices are part of the baseline "known set"; only a genuinely new device ID is flagged.
- **Occasional legitimate secondary location** → domestic business travel is included in the synthetic baseline generation as low-probability noise, so the model isn't a naive "one location only" rule.
""")
    st.caption("Synthetic data only — no real user, network, or organizational data is used anywhere in this system.")
