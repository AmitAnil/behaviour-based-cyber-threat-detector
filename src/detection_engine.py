"""
Baseline-and-deviation detection engine for the Behaviour-Based Cyber Threat
Detector (PS07).

Method (hybrid statistical + rule-based, entirely offline / no ML training
required so it is transparent and explainable to a SOC analyst):

  1. BASELINE (per user, learned only from the first `BASELINE_DAYS` days):
       - login-hour distribution (mean, std)
       - set of previously-seen devices
       - set of previously-seen (city, country) locations, + home lat/lon
       - set of previously-seen file resources
       - set of previously-seen network destination domains
       - daily volume distributions: file count, total file size (MB),
         total network egress (MB)  -> mean & std per user

  2. SCORING (per user, per day, over the "monitoring window"):
       Two families of signals are combined into one 0-100 anomaly score:

       a. Rule-based signals (hard, binary "this should never normally
          happen") - unrecognized device, physically-impossible travel
          between two logins (haversine distance / time > commercial-flight
          speed), access to resources/domains never seen in the baseline.

       b. Statistical signals (soft, "this is unusually far from this
          user's own normal") - z-score of the day's login hour, file
          count, file volume (MB) and network egress (MB) against that
          user's personal baseline mean/std. This is what catches gradual
          drift that no single static rule would trip.

       Score = sum of capped per-signal point contributions, clipped to
       [0, 100]. Severity = Low / Medium / High / Critical by threshold.

  3. EXPLAINABILITY: every alert carries a plain-English list of the exact
     signals that fired ("reasons"), and the raw log rows behind it can be
     pulled back out by (user_id, date) for the "supporting events" drill-down.
"""

import math
import numpy as np
import pandas as pd

BASELINE_DAYS = 30           # days 0..29 used purely to learn "normal"
IMPOSSIBLE_TRAVEL_KMH = 900  # faster than a commercial flight => physically implausible

SEVERITY_THRESHOLDS = [
    (75, "Critical"),
    (50, "High"),
    (25, "Medium"),
    (0, "Low"),
]


def severity_for(score):
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_logs(data_dir="data"):
    auth = pd.read_csv(f"{data_dir}/auth_logs.csv", parse_dates=["timestamp"])
    files = pd.read_csv(f"{data_dir}/file_access_logs.csv", parse_dates=["timestamp"])
    net = pd.read_csv(f"{data_dir}/network_logs.csv", parse_dates=["timestamp"])
    users = pd.read_csv(f"{data_dir}/users_devices.csv")
    for df in (auth, files, net):
        df["date"] = df["timestamp"].dt.floor("D")
    return auth, files, net, users


def build_baselines(auth, files, net, baseline_days=BASELINE_DAYS):
    """Learn a per-user 'normal' profile from the first `baseline_days` days only."""
    cutoff = auth["date"].min() + pd.Timedelta(days=baseline_days)

    b_auth = auth[auth["date"] < cutoff]
    b_files = files[files["date"] < cutoff]
    b_net = net[net["date"] < cutoff]

    baselines = {}
    for user_id, g in b_auth.groupby("user_id"):
        hours = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
        locs = set(zip(g["city"], g["country"]))
        devices = set(g["device_id"])
        home = g[["lat", "lon"]].mode().iloc[0] if not g.empty else pd.Series({"lat": 0, "lon": 0})

        f = b_files[b_files["user_id"] == user_id]
        daily_file_count = f.groupby("date").size()
        daily_file_size = f.groupby("date")["file_size_mb"].sum()

        n = b_net[b_net["user_id"] == user_id]
        daily_net = n.groupby("date")["bytes_mb"].sum()

        baselines[user_id] = {
            "login_hour_mean": float(hours.mean()) if len(hours) else 10.0,
            "login_hour_std": max(float(hours.std(ddof=0)) if len(hours) > 1 else 1.5, 0.75),
            "known_devices": devices,
            "known_locations": locs,
            "home_lat": float(home["lat"]), "home_lon": float(home["lon"]),
            "known_resources": set(f["resource"]),
            "known_domains": set(n["dest_domain"]),
            "daily_file_count_mean": float(daily_file_count.mean()) if len(daily_file_count) else 5.0,
            "daily_file_count_std": max(float(daily_file_count.std(ddof=0)) if len(daily_file_count) > 1 else 3.0, 2.0),
            "daily_file_size_mean": float(daily_file_size.mean()) if len(daily_file_size) else 10.0,
            "daily_file_size_std": max(float(daily_file_size.std(ddof=0)) if len(daily_file_size) > 1 else 5.0, 3.0),
            "daily_net_mb_mean": float(daily_net.mean()) if len(daily_net) else 80.0,
            "daily_net_mb_std": max(float(daily_net.std(ddof=0)) if len(daily_net) > 1 else 40.0, 20.0),
        }
    return baselines, cutoff


def _zscore(value, mean, std):
    return (value - mean) / std if std > 0 else 0.0


def score_window(auth, files, net, baselines, window_start, window_end):
    """Score every (user, day) pair in [window_start, window_end] inclusive."""
    alerts = []

    # Precompute implied travel speed between every consecutive pair of logins per user
    # (across the whole timeline, so the first login of the monitoring window can be
    # compared against the last login before it, even if that was in the baseline).
    travel_speed_by_login = {}
    for user_id, g in auth.sort_values("timestamp").groupby("user_id"):
        g = g.reset_index()
        for i in range(1, len(g)):
            prev, cur = g.loc[i - 1], g.loc[i]
            dt_hours = (cur["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue
            dist_km = haversine_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
            speed = dist_km / dt_hours
            travel_speed_by_login[cur["index"]] = {
                "speed_kmh": speed, "dist_km": dist_km, "dt_hours": dt_hours,
                "from_city": prev["city"], "from_country": prev["country"],
                "to_city": cur["city"], "to_country": cur["country"],
            }

    days = pd.date_range(window_start, window_end, freq="D")
    all_users = sorted(set(auth["user_id"]) | set(files["user_id"]) | set(net["user_id"]))

    for user_id in all_users:
        bl = baselines.get(user_id)
        if bl is None:
            continue
        u_auth_all = auth[auth["user_id"] == user_id].sort_values("timestamp").reset_index()
        u_files = files[files["user_id"] == user_id]
        u_net = net[net["user_id"] == user_id]

        for day in days:
            day_auth = u_auth_all[u_auth_all["date"] == day]
            day_files = u_files[u_files["date"] == day]
            day_net = u_net[u_net["date"] == day]

            if day_auth.empty and day_files.empty and day_net.empty:
                continue  # inactive that day, nothing to score

            score = 0.0
            reasons = []
            devices_seen = set(day_auth["device_id"]) if not day_auth.empty else set()

            # --- Rule-based signals ---
            new_devices = devices_seen - bl["known_devices"]
            if new_devices:
                score += 20
                reasons.append(f"Login from unrecognized device(s): {', '.join(sorted(new_devices))}")

            new_locations = set(zip(day_auth["city"], day_auth["country"])) - bl["known_locations"] if not day_auth.empty else set()
            if new_locations:
                score += 15
                loc_str = ", ".join(f"{c}, {k}" for c, k in sorted(new_locations))
                reasons.append(f"Login from new/unseen location: {loc_str}")

            max_speed = 0.0
            travel_detail = None
            for idx in day_auth["index"]:
                info = travel_speed_by_login.get(idx)
                if info and info["speed_kmh"] > max_speed:
                    max_speed = info["speed_kmh"]
                    travel_detail = info
            if travel_detail and max_speed > IMPOSSIBLE_TRAVEL_KMH:
                score += 35
                reasons.append(
                    f"Impossible travel: {travel_detail['from_city']} -> {travel_detail['to_city']} "
                    f"({travel_detail['dist_km']:.0f} km in {travel_detail['dt_hours']:.1f}h, "
                    f"implied speed {max_speed:.0f} km/h)"
                )

            new_domains = set(day_net["dest_domain"]) - bl["known_domains"] if not day_net.empty else set()
            if new_domains:
                score += min(15 * len(new_domains), 30)
                reasons.append(f"Traffic to new/unseen domain(s): {', '.join(sorted(new_domains))}")

            new_resources = set(day_files["resource"]) - bl["known_resources"] if not day_files.empty else set()
            if len(new_resources) >= 3:
                score += 20
                reasons.append(f"{len(new_resources)} distinct resources accessed that were never touched in the baseline period")
            elif new_resources:
                score += 8
                reasons.append(f"Accessed {len(new_resources)} resource(s) not seen in the baseline period")

            # --- Statistical signals (z-scores vs this user's own baseline) ---
            if not day_auth.empty:
                hours = day_auth["timestamp"].dt.hour + day_auth["timestamp"].dt.minute / 60.0
                z_hour = max(
                    abs(_zscore(h, bl["login_hour_mean"], bl["login_hour_std"])) for h in hours
                )
                pts = min(z_hour / 3.0, 1.0) * 15
                if pts >= 6:
                    score += pts
                    reasons.append(f"Login time unusual for this user (z={z_hour:.1f} vs personal baseline)")

            file_count = len(day_files)
            z_fc = _zscore(file_count, bl["daily_file_count_mean"], bl["daily_file_count_std"])
            pts = min(max(z_fc, 0) / 3.0, 1.0) * 15
            if pts >= 6:
                score += pts
                reasons.append(f"File-access volume {file_count} vs typical {bl['daily_file_count_mean']:.0f}/day (z={z_fc:.1f})")

            file_size = day_files["file_size_mb"].sum() if not day_files.empty else 0.0
            z_fs = _zscore(file_size, bl["daily_file_size_mean"], bl["daily_file_size_std"])
            pts = min(max(z_fs, 0) / 3.0, 1.0) * 15
            if pts >= 6:
                score += pts
                reasons.append(f"File data volume {file_size:.0f} MB vs typical {bl['daily_file_size_mean']:.0f} MB/day (z={z_fs:.1f})")

            net_mb = day_net["bytes_mb"].sum() if not day_net.empty else 0.0
            z_net = _zscore(net_mb, bl["daily_net_mb_mean"], bl["daily_net_mb_std"])
            pts = min(max(z_net, 0) / 3.0, 1.0) * 20
            if pts >= 8:
                score += pts
                reasons.append(f"Network egress {net_mb:.0f} MB vs typical {bl['daily_net_mb_mean']:.0f} MB/day (z={z_net:.1f})")

            score = float(np.clip(score, 0, 100))
            if not reasons or score < 12:
                continue  # perfectly normal / negligible-deviation day, don't clutter the alert list

            alerts.append({
                "user_id": user_id,
                "date": day,
                "devices": ", ".join(sorted(devices_seen)) if devices_seen else (u_auth_all["device_id"].iloc[-1] if not u_auth_all.empty else "n/a"),
                "anomaly_score": round(score, 1),
                "severity": severity_for(score),
                "num_reasons": len(reasons),
                "reasons": reasons,
                "logins": len(day_auth),
                "file_events": len(day_files),
                "file_mb": round(float(file_size), 1),
                "net_mb": round(float(net_mb), 1),
            })

    return pd.DataFrame(alerts).sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def get_supporting_events(auth, files, net, user_id, date):
    """Raw log rows behind a given (user, day) alert, for the dashboard drill-down."""
    day = pd.Timestamp(date)
    a = auth[(auth["user_id"] == user_id) & (auth["date"] == day)].sort_values("timestamp")
    f = files[(files["user_id"] == user_id) & (files["date"] == day)].sort_values("timestamp")
    n = net[(net["user_id"] == user_id) & (net["date"] == day)].sort_values("timestamp")
    return a, f, n


def run_pipeline(data_dir="data"):
    auth, files, net, users = load_logs(data_dir)
    baselines, cutoff = build_baselines(auth, files, net)
    window_end = auth["date"].max()
    alerts_df = score_window(auth, files, net, baselines, cutoff, window_end)
    return {
        "auth": auth, "files": files, "net": net, "users": users,
        "baselines": baselines, "cutoff": cutoff, "window_end": window_end,
        "alerts": alerts_df,
    }


if __name__ == "__main__":
    result = run_pipeline()
    alerts = result["alerts"]
    print(f"Baseline learned from days before {result['cutoff'].date()}")
    print(f"Scored monitoring window through {result['window_end'].date()}")
    print(f"Total alerts (user-days with >=1 signal): {len(alerts)}")
    print()
    print(alerts[["user_id", "date", "anomaly_score", "severity", "num_reasons"]].head(20).to_string(index=False))
    print()
    top = alerts.iloc[0]
    print(f"Top alert: {top['user_id']} on {top['date'].date()} - score {top['anomaly_score']} ({top['severity']})")
    for r in top["reasons"]:
        print(f"  - {r}")
