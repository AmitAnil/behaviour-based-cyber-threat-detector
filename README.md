# Behaviour-Based Cyber Threat Detector

**PS07 · Software / Applied Blockchain & Cybersecurity**
Host Ministry: Ministry of Electronics & IT (MeitY) · CERT-In

Individual login or file-access events can look completely normal on their own, but
taken together across a user or device they can reveal an account compromise that
static rule alerts miss. This project ingests simulated authentication, file-access,
and network-activity logs, learns a personal "normal" baseline for every user/device,
scores deviations from that baseline with a hybrid statistical + rule-based engine,
and presents every alert — with an anomaly score, a severity level, and the exact
supporting events behind it — in an interactive Streamlit dashboard.

> All data in this project is **synthetically generated**. No real user, network, or
> organizational data is used anywhere.

---

## 1. Architecture

```
src/generate_data.py   ──►  data/*.csv   ──►  src/detection_engine.py  ──►  app.py (Streamlit)
  (synthetic logs)         (auth / file /       (baseline + scoring)         (dashboard)
                             network / users)
```

1. **`src/generate_data.py`** — synthesizes 35 days of authentication, file-access,
   and network-activity logs for 15 users/devices with realistic per-user baselines
   (working hours, home location, typical volumes), then injects two demo compromise
   scenarios (see §4).
2. **`src/detection_engine.py`** — learns a per-user baseline from the first 30 days,
   then scores every remaining user-day against it, producing an explainable
   0–100 anomaly score, a severity label, and a plain-English reasons list.
3. **`app.py`** — a Streamlit dashboard with five views: Overview, Alert Drill-Down,
   User Baseline Profiles, a narrated Normal-vs-Anomalous demo, and a Methodology page.

## 2. Data & schema

| File | Columns |
|---|---|
| `data/users_devices.csv` | `user_id, name, department, home_city, home_country, devices` |
| `data/auth_logs.csv` | `timestamp, user_id, device_id, source_ip, city, country, lat, lon, auth_result` |
| `data/file_access_logs.csv` | `timestamp, user_id, device_id, resource, action, file_size_mb` |
| `data/network_logs.csv` | `timestamp, user_id, device_id, dest_domain, bytes_mb, protocol` |
| `data/ground_truth.csv` | the injected anomaly events, for demo narration only (never used by the engine) |

15 synthetic users across Finance, Engineering, HR, Sales, and IT, each with their own
login-hour pattern (including a couple of legitimate night-shift IT users), home city,
device set, and typical file/network volume, over **30 baseline days + 5 monitored
days**.

## 3. Detection methodology

For each user, the **first 30 days** are treated as a training/baseline period, never
touched by scoring. From it the engine learns: login-hour mean/std, the set of known
devices/locations/file-resources/network-domains, and daily-volume mean/std for file
count, file MB, and network MB.

Every subsequent user-day is scored by combining two signal families into one
**0–100 anomaly score**:

- **Rule-based (hard) signals** — things that should essentially never happen for a
  genuine user: login from an unrecognized device, a **physically impossible travel
  speed** between two logins (haversine distance ÷ time > commercial-flight speed),
  traffic to a never-seen network domain, or access to file resources never touched
  in the baseline.
- **Statistical (soft) signals** — z-scores of the day's login hour, file-access
  count, file-access volume, and network egress against **that specific user's own**
  baseline mean/std. This is what catches gradual drift and volume creep that no
  fixed global threshold would generalize across a diverse workforce.

Severity bands: **Critical ≥ 75, High ≥ 50, Medium ≥ 25, Low > 0** (below ~12 points a
day is treated as normal noise and not surfaced). Every alert carries the exact list
of signals that fired, and the dashboard's Alert Drill-Down page pulls back the raw
log rows behind it.

## 4. Demo scenario (normal vs. anomalous)

Two contrasting scenarios are injected into the synthetic data and are walked through
on the dashboard's "Normal vs Anomalous Demo" page:

- **Scenario A — Impossible travel + mass exfiltration.** A legitimate morning login
  is followed two hours later by a login from another continent, on a never-seen
  device, followed by a 25-minute burst of 48 downloads (including files never
  touched before) and an 800MB+ transfer to a domain nobody has ever contacted.
  Every signal is a hard rule violation; together it scores **100/100, Critical**.
- **Scenario B — Gradual behavioural drift.** Same device, same home location, no
  single alarming event — but over four nights the login time creeps from ~10pm to
  past 3am and file-access volume climbs from 1.4× to 3.4× normal. No static rule
  fires; the escalating **statistical** z-scores alone carry the user from Low to
  High severity night over night, which is exactly the class of subtle,
  individually-unremarkable-but-collectively-suspicious behaviour static alerting
  misses and this baseline approach is built to catch.

## 5. Running locally

```bash
pip install -r requirements.txt
python src/generate_data.py     # writes data/*.csv (re-run to regenerate with a new seed)
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`).

## 6. Deploying

The app is a standard Streamlit app with no external services or secrets, so it
deploys directly to **Streamlit Community Cloud**: push this repo to GitHub, go to
[share.streamlit.io](https://share.streamlit.io), point it at the repo with `app.py`
as the entrypoint, and deploy. It also runs unmodified on any platform that can run
`streamlit run app.py` (Render, Railway, a container, etc.).

## 7. Project structure

```
threat-detector/
├── app.py                     # Streamlit dashboard (5 pages)
├── requirements.txt
├── .streamlit/config.toml     # theme
├── src/
│   ├── generate_data.py       # synthetic log generator + scenario injection
│   └── detection_engine.py    # baseline builder + anomaly scoring engine
├── data/                      # generated CSVs (auth/file/network logs, users, ground truth)
└── README.md
```

## 8. Design notes / edge cases handled

- **No baseline yet** — a user with no history is excluded from scoring rather than
  producing false alarms from insufficient data.
- **Per-user, not per-role, baselines** — a night-shift IT admin logging in at 2am is
  normal *for them*; the same login for a 9-to-5 analyst is a strong anomaly. Global
  or role-based rules would need to be too loose (missing Scenario B) or too strict
  (false-positiving on legitimate night-shift work); personal baselines avoid both.
- **Weekends / inactive days** — lower activity is modelled as a lower daily-active
  probability rather than a fixed schedule, and an inactive day produces no alert
  (nothing to score) instead of a spurious "volume dropped to zero" anomaly.
- **Multiple registered devices per user** — all of a user's registered devices sit
  in their baseline "known set"; only a genuinely new device ID is flagged.
- **Occasional legitimate travel** — low-probability secondary-location logins are
  included in baseline generation so the model isn't a naive "one location only" rule.

## 9. Possible extensions

- Persist baselines/alerts to a database and support streaming/incremental scoring.
- Add an isolation-forest or autoencoder model as a third, ML-based signal alongside
  the rule-based and z-score signals, and compare precision/recall on labelled data.
- Alert triage workflow (acknowledge / escalate / false-positive feedback loop that
  retrains the baseline).
- Real log-source connectors (SIEM/EDR export formats, Windows Event Log, Okta/Azure
  AD sign-in logs) in place of the synthetic generator.
