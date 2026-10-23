"""Журнал событий (SQLite — без отдельного сервера; для прома меняется на PostgreSQL)."""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    camera_name TEXT,
    area        TEXT,
    zone        TEXT,
    type        TEXT NOT NULL,
    title       TEXT,
    severity    TEXT,
    track_id    INTEGER,
    info        TEXT,
    onset_ts    REAL,           -- когда признак появился в кадре
    alert_ts    REAL,           -- когда система подняла тревогу
    end_ts      REAL,           -- когда нарушение закончилось
    snapshot    TEXT,           -- кадр-доказательство
    clip        TEXT,           -- фрагмент (GIF)
    status      TEXT DEFAULT 'new',   -- new | confirmed | false | forwarded
    status_by   TEXT,
    status_ts   REAL,
    comment     TEXT
);
CREATE INDEX IF NOT EXISTS ix_ev_ts ON events(alert_ts);
CREATE INDEX IF NOT EXISTS ix_ev_cam ON events(camera_id);
"""


class Storage:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def add(self, **kw) -> int:
        cols = ",".join(kw)
        q = ",".join("?" * len(kw))
        with self.lock:
            cur = self.db.execute(f"INSERT INTO events ({cols}) VALUES ({q})", list(kw.values()))
            self.db.commit()
            return int(cur.lastrowid)

    def update(self, eid: int, **kw):
        if not kw:
            return
        s = ",".join(f"{k}=?" for k in kw)
        with self.lock:
            self.db.execute(f"UPDATE events SET {s} WHERE id=?", [*kw.values(), eid])
            self.db.commit()

    def get(self, eid: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            r = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        return dict(r) if r else None

    def query(self, camera=None, type=None, zone=None, status=None, severity=None,
              ts_from=None, ts_to=None, limit=200, offset=0) -> List[Dict[str, Any]]:
        where, args = [], []
        for col, val in (("camera_id", camera), ("type", type), ("zone", zone), ("status", status),
                         ("severity", severity)):
            if val:
                vals = val.split(",")
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                args += vals
        if ts_from:
            where.append("alert_ts >= ?")
            args.append(float(ts_from))
        if ts_to:
            where.append("alert_ts <= ?")
            args.append(float(ts_to))
        w = ("WHERE " + " AND ".join(where)) if where else ""
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM events {w} ORDER BY alert_ts DESC LIMIT ? OFFSET ?",
                                   [*args, limit, offset]).fetchall()
        return [dict(r) for r in rows]

    def stats(self, ts_from: Optional[float] = None) -> Dict[str, Any]:
        ts_from = ts_from or (time.time() - 30 * 86400)
        with self.lock:
            def q(sql):
                return [dict(r) for r in self.db.execute(sql, (ts_from,)).fetchall()]
            base = "FROM events WHERE alert_ts >= ? AND status != 'false'"
            res = {
                "total": q(f"SELECT COUNT(*) n {base}")[0]["n"],
                "by_type": q(f"SELECT type, title, COUNT(*) n {base} GROUP BY type ORDER BY n DESC"),
                "by_camera": q(f"SELECT camera_id, camera_name, COUNT(*) n {base} GROUP BY camera_id ORDER BY n DESC"),
                "by_zone": q(f"SELECT COALESCE(NULLIF(zone,''),'—') zone, COUNT(*) n {base} GROUP BY zone ORDER BY n DESC"),
                "by_hour": q(f"SELECT CAST(strftime('%H', alert_ts, 'unixepoch', 'localtime') AS INT) h, COUNT(*) n {base} GROUP BY h"),
                "by_day": q(f"SELECT strftime('%Y-%m-%d', alert_ts, 'unixepoch', 'localtime') d, COUNT(*) n {base} GROUP BY d ORDER BY d"),
                "by_status": [dict(r) for r in self.db.execute(
                    "SELECT status, COUNT(*) n FROM events WHERE alert_ts >= ? GROUP BY status", (ts_from,)).fetchall()],
                "latency": q(f"SELECT type, AVG(alert_ts-onset_ts) avg_s, MAX(alert_ts-onset_ts) max_s {base} GROUP BY type"),
            }
        # точность по отметкам оператора: confirmed / (confirmed + false)
        st = {r["status"]: r["n"] for r in res["by_status"]}
        judged = st.get("confirmed", 0) + st.get("forwarded", 0) + st.get("false", 0)
        res["precision"] = (st.get("confirmed", 0) + st.get("forwarded", 0)) / judged if judged else None
        with self.lock:
            pt = self.db.execute(
                "SELECT type, SUM(status IN ('confirmed','forwarded')) ok, SUM(status='false') bad "
                "FROM events WHERE alert_ts >= ? GROUP BY type", (ts_from,)).fetchall()
        res["precision_by_type"] = [
            {"type": r["type"], "ok": r["ok"], "bad": r["bad"],
             "precision": (r["ok"] / (r["ok"] + r["bad"])) if (r["ok"] + r["bad"]) else None} for r in pt]
        return res
