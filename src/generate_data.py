"""
Synthetic log generator for the Behaviour-Based Cyber Threat Detector (PS07).

Generates three log streams (authentication, file-access, network-activity) for a
population of synthetic users/devices over a 35-day window:

  - Days 0-29:  pure "training" baseline behaviour (normal, with natural noise).
  - Days 30-34: continued normal behaviour for almost everyone, PLUS two
                deliberately injected anomalous scenarios used as the demo:

      Scenario A - "Impossible travel + mass exfiltration"
          A single, loud, multi-signal account-takeover: a login from a
          location the user could not physically have reached, from an
          unrecognized device, followed by a burst of downloads of files the
          user has never touched, followed by a large transfer to a domain
          never seen before. Designed to be catchable by simple rules AND by
          statistics.

      Scenario B - "Gradual behavioural drift"
          A quieter, creeping anomaly: the same user/device/location, but
          login times drift later each night and file-access volume climbs
          day over day. No single event looks alarming in isolation - only a
          statistical baseline-and-deviation approach catches it.

Output: CSV files under ./data/
  - users_devices.csv   (ground-truth user/device profile metadata)
  - auth_logs.csv
  - file_access_logs.csv
  - network_logs.csv
  - ground_truth.csv    (which events/days are "should be flagged", for demo narration only)
"""

import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = "data"
NUM_BASELINE_DAYS = 30
NUM_DEMO_DAYS = 5
TOTAL_DAYS = NUM_BASELINE_DAYS + NUM_DEMO_DAYS
START_DATE = datetime(2026, 8, 1)

# ---------------------------------------------------------------------------
# Reference locations: (city, country, lat, lon)
# ---------------------------------------------------------------------------
HOME_CITIES = [
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Bangalore", "India", 12.9716, 77.5946),
    ("Hyderabad", "India", 17.3850, 78.4867),
    ("Chennai", "India", 13.0827, 80.2707),
    ("Pune", "India", 18.5204, 73.8567),
    ("Kolkata", "India", 22.5726, 88.3639),
    ("Ahmedabad", "India", 23.0225, 72.5714),
]

# Locations only ever used for injected attacker logins (never appear as a
# legitimate home city), representing plausible attacker infrastructure.
ATTACKER_LOCATIONS = [
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Bucharest", "Romania", 44.4268, 26.1025),
    ("Hanoi", "Vietnam", 21.0278, 105.8342),
]

DEPARTMENTS = {
    "Finance": {
        "resources": [
            "finance/expense_reports/", "finance/vendor_invoices/",
            "finance/budget_2026.xlsx", "finance/general_ledger.xlsx",
        ],
        "sensitive_resources": [
            "finance/payroll_master.xlsx", "finance/bank_account_details.db",
            "finance/employee_ssn_records.db",
        ],
        "domains": ["erp.corp.local", "sap.corp.local", "mail.corp.local"],
    },
    "Engineering": {
        "resources": [
            "eng/repo/backend/", "eng/repo/frontend/", "eng/design_docs/",
            "eng/ci_logs/",
        ],
        "sensitive_resources": [
            "eng/prod_credentials.env", "eng/infra/terraform_state.tfstate",
        ],
        "domains": ["github.corp.local", "jira.corp.local", "mail.corp.local"],
    },
    "HR": {
        "resources": ["hr/onboarding/", "hr/policies/", "hr/leave_records.xlsx"],
        "sensitive_resources": ["hr/employee_records.db", "hr/salary_bands.xlsx"],
        "domains": ["workday.corp.local", "mail.corp.local"],
    },
    "Sales": {
        "resources": ["sales/pipeline.xlsx", "sales/decks/", "sales/contracts/"],
        "sensitive_resources": ["sales/client_contracts_confidential/"],
        "domains": ["salesforce.corp.local", "mail.corp.local"],
    },
    "IT": {
        "resources": ["it/tickets/", "it/asset_inventory.xlsx", "it/scripts/"],
        "sensitive_resources": ["it/admin_credentials.vault", "it/network_diagram.vsdx"],
        "domains": ["adminportal.corp.local", "mail.corp.local"],
    },
}

FIRST_NAMES = ["Aarav", "Priya", "Rohit", "Sneha", "Karan", "Ananya", "Vikram",
               "Divya", "Arjun", "Neha", "Rahul", "Pooja", "Sanjay", "Meera", "Ishaan"]

NUM_USERS = 15


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def make_users():
    users = []
    depts = list(DEPARTMENTS.keys())
    for i in range(NUM_USERS):
        uid = f"u{i+1:02d}"
        name = FIRST_NAMES[i % len(FIRST_NAMES)]
        dept = depts[i % len(depts)]
        home = HOME_CITIES[i % len(HOME_CITIES)]
        # Most users work standard daytime hours; a couple are night-shift IT/ops.
        night_shift = dept == "IT" and i % 3 == 0
        login_hour_mean = 22.0 if night_shift else random.uniform(8.5, 10.5)
        login_hour_std = 1.3
        primary_device = f"{uid}-laptop"
        devices = [primary_device]
        if i % 4 == 0:  # some users also have a registered mobile device
            devices.append(f"{uid}-mobile")
        users.append({
            "user_id": uid,
            "name": name,
            "department": dept,
            "home_city": home[0],
            "home_country": home[1],
            "home_lat": home[2],
            "home_lon": home[3],
            "devices": devices,
            "login_hour_mean": login_hour_mean,
            "login_hour_std": login_hour_std,
            "logins_per_day_mean": random.uniform(2.0, 4.5),
            "files_per_day_mean": random.uniform(12, 30),
            "file_size_mean_mb": random.uniform(1.0, 3.5),
            "network_mb_per_day_mean": random.uniform(80, 220),
        })
    return users


def gen_ip_for_location(city):
    # Deterministic-ish fake IP per city so the same location "looks" consistent.
    base = abs(hash(city)) % 200
    return f"10.{base}.{random.randint(0,255)}.{random.randint(1,254)}"


def gen_normal_day(user, day_idx, date, auth_rows, file_rows, net_rows):
    dept = DEPARTMENTS[user["department"]]
    is_weekend = date.weekday() >= 5
    active_prob = 0.35 if is_weekend else 0.97
    if random.random() > active_prob:
        return

    n_logins = max(1, int(round(np.random.poisson(max(user["logins_per_day_mean"] * (0.4 if is_weekend else 1.0), 0.5)))))
    device = random.choice(user["devices"])
    city, country, lat, lon = user["home_city"], user["home_country"], user["home_lat"], user["home_lon"]

    login_times = []
    for _ in range(n_logins):
        hour = np.clip(np.random.normal(user["login_hour_mean"], user["login_hour_std"]), 0, 23.98)
        ts = date + timedelta(hours=float(hour), minutes=random.uniform(0, 59))
        login_times.append(ts)
    login_times.sort()

    for ts in login_times:
        ip = gen_ip_for_location(city)
        auth_rows.append({
            "timestamp": ts, "user_id": user["user_id"], "device_id": device,
            "source_ip": ip, "city": city, "country": country,
            "lat": lat, "lon": lon, "auth_result": "success",
        })

        n_files = max(0, int(round(np.random.poisson(user["files_per_day_mean"] / max(n_logins, 1)))))
        for _ in range(n_files):
            fts = ts + timedelta(minutes=random.uniform(1, 90))
            resource = random.choice(dept["resources"])
            # extremely rare organic touch of a sensitive resource (e.g. quarterly audit)
            if random.random() < 0.01:
                resource = random.choice(dept["sensitive_resources"])
            size = max(0.05, np.random.lognormal(mean=math.log(user["file_size_mean_mb"]), sigma=0.6))
            action = random.choices(["read", "write", "download"], weights=[0.7, 0.2, 0.1])[0]
            file_rows.append({
                "timestamp": fts, "user_id": user["user_id"], "device_id": device,
                "resource": resource, "action": action, "file_size_mb": round(size, 2),
            })

        n_conns = max(1, int(round(np.random.poisson(3))))
        total_mb = max(1.0, np.random.lognormal(mean=math.log(user["network_mb_per_day_mean"] / n_logins), sigma=0.5))
        per_conn = total_mb / n_conns
        for _ in range(n_conns):
            nts = ts + timedelta(minutes=random.uniform(1, 90))
            domain = random.choice(dept["domains"])
            net_rows.append({
                "timestamp": nts, "user_id": user["user_id"], "device_id": device,
                "dest_domain": domain, "bytes_mb": round(max(0.1, np.random.normal(per_conn, per_conn * 0.3)), 2),
                "protocol": "https",
            })


def main():
    users = make_users()
    auth_rows, file_rows, net_rows = [], [], []
    ground_truth = []

    # --- Baseline + continued-normal generation for every user, every day ---
    target_a = users[2]   # Scenario A victim (index fixed for reproducibility)
    target_b = users[8]   # Scenario B victim

    for day_idx in range(TOTAL_DAYS):
        date = START_DATE + timedelta(days=day_idx)
        for user in users:
            # Skip normal generation on the exact injected days for our two
            # target users on the days we override below (still want *some*
            # normal activity that day for scenario A, so only B suppresses it).
            if user is target_b and day_idx in (31, 32, 33, 34):
                continue  # replaced entirely by the drifting-anomaly generator
            gen_normal_day(user, day_idx, date, auth_rows, file_rows, net_rows)

    # --- Scenario A: single-day loud takeover on day 32 ---
    day = START_DATE + timedelta(days=32)
    dept = DEPARTMENTS[target_a["department"]]
    home_city, home_country = target_a["home_city"], target_a["home_country"]

    # Legit morning login (normal) already generated by the loop above for day 32.
    # ~2 hours after a plausible legit login, inject the attacker login.
    attacker_city, attacker_country, alat, alon = random.choice(ATTACKER_LOCATIONS)
    attack_ts = day.replace(hour=13, minute=10)
    unknown_device = f"unknown-device-{random.randint(1000,9999)}"
    auth_rows.append({
        "timestamp": attack_ts, "user_id": target_a["user_id"], "device_id": unknown_device,
        "source_ip": gen_ip_for_location(attacker_city), "city": attacker_city, "country": attacker_country,
        "lat": alat, "lon": alon, "auth_result": "success",
    })
    ground_truth.append({"scenario": "A", "type": "impossible_travel_login", "user_id": target_a["user_id"],
                          "timestamp": attack_ts, "detail": f"Login from {attacker_city}, {attacker_country} on unrecognized device {unknown_device}"})

    # Mass download burst of files never normally touched, all within ~25 minutes.
    all_sensitive = dept["sensitive_resources"]
    for i in range(48):
        fts = attack_ts + timedelta(minutes=random.uniform(1, 25))
        resource = random.choice(all_sensitive) if i % 2 == 0 else random.choice(dept["resources"])
        size = max(2.0, np.random.lognormal(mean=math.log(14.0), sigma=0.4))
        file_rows.append({
            "timestamp": fts, "user_id": target_a["user_id"], "device_id": unknown_device,
            "resource": resource, "action": "download", "file_size_mb": round(size, 2),
        })
    ground_truth.append({"scenario": "A", "type": "mass_download_burst", "user_id": target_a["user_id"],
                          "timestamp": attack_ts, "detail": "48 downloads (incl. sensitive files) within 25 minutes"})

    # Large exfil transfer to a never-seen-before external domain.
    exfil_ts = attack_ts + timedelta(minutes=30)
    net_rows.append({
        "timestamp": exfil_ts, "user_id": target_a["user_id"], "device_id": unknown_device,
        "dest_domain": "file-drop-ex4.net", "bytes_mb": 812.0, "protocol": "https",
    })
    ground_truth.append({"scenario": "A", "type": "exfil_transfer", "user_id": target_a["user_id"],
                          "timestamp": exfil_ts, "detail": "812 MB transfer to previously-unseen external domain"})

    # --- Scenario B: gradual drift over days 31-34 ---
    drift_hours = [1.0, 2.2, 3.0, 3.6]         # creeping later each night (vs ~10am baseline)
    drift_file_multiplier = [1.4, 1.9, 2.6, 3.4]  # file volume creeping up each day
    for offset, (hr, mult) in enumerate(zip(drift_hours, drift_file_multiplier)):
        day_idx = 31 + offset
        day = START_DATE + timedelta(days=day_idx)
        ts = day.replace(hour=int(hr), minute=random.randint(0, 59))
        device = target_b["devices"][0]
        auth_rows.append({
            "timestamp": ts, "user_id": target_b["user_id"], "device_id": device,
            "source_ip": gen_ip_for_location(target_b["home_city"]), "city": target_b["home_city"],
            "country": target_b["home_country"], "lat": target_b["home_lat"], "lon": target_b["home_lon"],
            "auth_result": "success",
        })
        dept_b = DEPARTMENTS[target_b["department"]]
        n_files = int(round(target_b["files_per_day_mean"] * mult))
        for _ in range(n_files):
            fts = ts + timedelta(minutes=random.uniform(1, 120))
            resource = random.choice(dept_b["resources"])
            if offset == 3 and random.random() < 0.15:
                resource = random.choice(dept_b["sensitive_resources"])
            size = max(0.05, np.random.lognormal(mean=math.log(target_b["file_size_mean_mb"]), sigma=0.6))
            file_rows.append({
                "timestamp": fts, "user_id": target_b["user_id"], "device_id": device,
                "resource": resource, "action": random.choices(["read", "write", "download"], weights=[0.5, 0.2, 0.3])[0],
                "file_size_mb": round(size, 2),
            })
        total_mb = target_b["network_mb_per_day_mean"] * mult
        for _ in range(4):
            nts = ts + timedelta(minutes=random.uniform(1, 120))
            net_rows.append({
                "timestamp": nts, "user_id": target_b["user_id"], "device_id": device,
                "dest_domain": random.choice(dept_b["domains"]), "bytes_mb": round(total_mb / 4, 2),
                "protocol": "https",
            })
        ground_truth.append({"scenario": "B", "type": "gradual_drift_day", "user_id": target_b["user_id"],
                              "timestamp": ts, "detail": f"Login hour drifted to {hr:.1f}h past midnight; file volume x{mult}"})

    # --- Assemble & persist ---
    auth_df = pd.DataFrame(auth_rows).sort_values("timestamp").reset_index(drop=True)
    file_df = pd.DataFrame(file_rows).sort_values("timestamp").reset_index(drop=True)
    net_df = pd.DataFrame(net_rows).sort_values("timestamp").reset_index(drop=True)
    gt_df = pd.DataFrame(ground_truth).sort_values("timestamp").reset_index(drop=True)

    users_df = pd.DataFrame([{
        "user_id": u["user_id"], "name": u["name"], "department": u["department"],
        "home_city": u["home_city"], "home_country": u["home_country"],
        "devices": ";".join(u["devices"]),
    } for u in users])

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    auth_df.to_csv(f"{OUT_DIR}/auth_logs.csv", index=False)
    file_df.to_csv(f"{OUT_DIR}/file_access_logs.csv", index=False)
    net_df.to_csv(f"{OUT_DIR}/network_logs.csv", index=False)
    users_df.to_csv(f"{OUT_DIR}/users_devices.csv", index=False)
    gt_df.to_csv(f"{OUT_DIR}/ground_truth.csv", index=False)

    print(f"Users: {len(users_df)}")
    print(f"Auth logs: {len(auth_df)} rows")
    print(f"File-access logs: {len(file_df)} rows")
    print(f"Network logs: {len(net_df)} rows")
    print(f"Injected ground-truth events: {len(gt_df)}")
    print(f"Scenario A victim: {target_a['user_id']} ({target_a['name']}, {target_a['department']})")
    print(f"Scenario B victim: {target_b['user_id']} ({target_b['name']}, {target_b['department']})")


if __name__ == "__main__":
    main()
