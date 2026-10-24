"""Веб-сервер: API + WebSocket-тревоги + раздача интерфейса оператора.

Запуск:  python -m sv.server --config config/config.yaml --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import platform
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Set

import yaml
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .notify import make_notifiers
from .pipeline import CameraWorker, Models
from .rules.base import REGISTRY, Rule
from .storage import Storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger("server")


class App:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        os.chdir(ROOT)
        self.data_dir = os.path.join(ROOT, "data")
        self.media_dir = os.path.join(self.data_dir, "media")
        os.makedirs(self.media_dir, exist_ok=True)
        self.db = Storage(os.path.join(self.data_dir, "events.db"))
        self.models = Models(self.cfg)
        self.notifiers = make_notifiers(self.cfg.get("notify"))
        self.workers: Dict[str, CameraWorker] = {}
        self.clients: Set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.cfg_lock = threading.Lock()
        self.started = time.time()

    # ---------------------------------------------------------- камеры
    def start(self):
        for cam in self.cfg.get("cameras", []):
            w = CameraWorker(cam, self.models, self.on_open, self.on_close, self.media_dir)
            self.workers[cam["id"]] = w
            w.start()

    def save_config(self):
        with self.cfg_lock:
            tmp = self.config_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self.config_path)

    # ---------------------------------------------------------- события (вызываются из потоков камер)
    def on_open(self, worker: CameraWorker, rule: Rule, sig, ctx, ev) -> int:
        cam = worker.cam
        title = rule.title
        if rule.type == "fall" and sig.info == "человек лежит":
            title = "Человек лежит на полу"
        # сначала доказательства (кадр, фрагмент), потом запись — в журнале не бывает событий без картинки
        import uuid as _uuid
        frame = getattr(worker, "_cur_frame", None)
        snap = clip = None
        if frame is not None:
            snap, clip = worker.save_evidence(f"{cam['id']}_{int(ev.alert_t)}_{_uuid.uuid4().hex[:6]}", frame, sig.det, title)
        zone = sig.zone
        if not zone and frame is not None:   # падение, бег и т.п. тоже привязываем к зоне, если человек в ней
            H, W = frame.shape[:2]
            bx = sig.det.box
            for z in getattr(worker, "zones", []):
                try:
                    if z.contains(float((bx[0] + bx[2]) / 2), float(bx[3]), W, H):
                        zone = z.name
                        break
                except Exception:
                    pass
        eid = self.db.add(camera_id=cam["id"], camera_name=cam.get("name", cam["id"]), area=cam.get("area", ""),
                          zone=zone, type=rule.type, title=title, severity=rule.severity,
                          track_id=sig.det.track_id, info=sig.info, onset_ts=ev.onset_t, alert_ts=ev.alert_t,
                          snapshot=os.path.basename(snap) if snap else None,
                          clip=os.path.basename(clip) if clip else None)
        rec = self.db.get(eid)
        rec["snapshot_path"] = snap
        for n in self.notifiers:
            n.send(rec)
        self.broadcast({"type": "event_open", "event": self._pub(rec)})
        log.info("СОБЫТИЕ #%s %s [%s] %s", eid, title, cam["id"], sig.info)
        return eid

    def on_close(self, worker: CameraWorker, ev):
        if ev.id:
            self.db.update(ev.id, end_ts=ev.closed_t)
            self.broadcast({"type": "event_close", "id": ev.id, "end_ts": ev.closed_t})

    @staticmethod
    def _pub(rec: dict) -> dict:
        rec = dict(rec)
        rec.pop("snapshot_path", None)
        rec["snapshot_url"] = f"/media/{rec['snapshot']}" if rec.get("snapshot") else None
        rec["clip_url"] = f"/media/{rec['clip']}" if rec.get("clip") else None
        return rec

    def broadcast(self, msg: dict):
        if not self.loop:
            return
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    async def _send(self, ws: WebSocket, data: str):
        try:
            await ws.send_text(data)
        except Exception:
            self.clients.discard(ws)

    def metrics(self) -> dict:
        cams = {}
        for cid, w in self.workers.items():
            m = dict(w.metrics)
            cams[cid] = m
        fps = [m["fps"] for m in cams.values()]
        return {"backend": self.models.kind, "cameras": cams, "uptime_s": time.time() - self.started,
                "host": platform.node(), "total_fps": sum(fps)}


def create_app(config_path: str) -> FastAPI:
    core = App(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        core.loop = asyncio.get_running_loop()
        core.start()

        async def ticker():
            while True:
                await asyncio.sleep(2)
                core.broadcast({"type": "metrics", "metrics": core.metrics()})

        task = asyncio.create_task(ticker())
        yield
        task.cancel()
        for w in core.workers.values():
            w.stop()

    app = FastAPI(title="LinQ Safety Vision", lifespan=lifespan)
    app.state.core = core
    app.mount("/media", StaticFiles(directory=core.media_dir), name="media")
    app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(ROOT, "web", "index.html"))

    # ------------------------------------------------------------ камеры
    @app.get("/api/zones")
    def zones():
        names = {z.get("name") for c in core.cfg.get("cameras", []) for z in c.get("zones", []) if z.get("name")}
        names |= {r["zone"] for r in core.db.query(None, None, None, None, None, None, None, 5000, 0) if r.get("zone")}
        return sorted(names)

    @app.get("/api/cameras")
    def cameras():
        out = []
        for cam in core.cfg.get("cameras", []):
            w = core.workers.get(cam["id"])
            out.append({**cam, "metrics": w.metrics if w else None})
        return out

    @app.get("/api/cameras/{cid}/stream")
    def stream(cid: str):
        w = core.workers.get(cid)
        if not w:
            raise HTTPException(404)

        def gen():
            last = None
            while not w.stopped:
                j = w.jpeg
                if j is not None and j is not last:
                    last = j
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n"
                time.sleep(0.04)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/cameras/{cid}/frame.jpg")
    def frame(cid: str, raw: int = 0):
        w = core.workers.get(cid)
        if not w:
            raise HTTPException(404)
        j = w.raw_jpeg if raw else w.jpeg
        if j is None:
            raise HTTPException(503, "кадр ещё не готов")
        return Response(j, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.put("/api/cameras/{cid}/config")
    def put_config(cid: str, body: dict = Body(...)):
        cams = core.cfg.get("cameras", [])
        cam = next((c for c in cams if c["id"] == cid), None)
        if not cam:
            raise HTTPException(404)
        for k in ("name", "area", "zones", "rules", "detect_every"):
            if k in body:
                cam[k] = body[k]
        for r in (cam.get("rules") or {}):
            if r not in REGISTRY:
                raise HTTPException(400, f"неизвестное правило {r}")
        core.workers[cid].apply_config(cam)
        core.save_config()
        return cam

    @app.get("/api/rules")
    def rules():
        return [{"type": t, "title": c.title, "severity": c.severity, "doc": (c.__doc__ or "").strip(),
                 "defaults": {**Rule.defaults, **c.defaults}} for t, c in REGISTRY.items()]

    # ------------------------------------------------------------ события
    @app.get("/api/events")
    def events(camera: str = None, type: str = None, zone: str = None, status: str = None, severity: str = None,
               ts_from: float = None, ts_to: float = None, limit: int = 200, offset: int = 0):
        rows = core.db.query(camera, type, zone, status, severity, ts_from, ts_to, limit, offset)
        return [core._pub(r) for r in rows]

    @app.get("/api/events.csv")
    def events_csv(camera: str = None, type: str = None, zone: str = None, status: str = None,
                   ts_from: float = None, ts_to: float = None):
        rows = core.db.query(camera, type, zone, status, None, ts_from, ts_to, 100000, 0)
        buf = io.StringIO()
        wr = csv.writer(buf, delimiter=";")
        wr.writerow(["№", "Время тревоги", "Начало нарушения", "Окончание", "Камера", "Участок", "Зона", "Нарушение",
                     "Критичность", "Трек", "Статус", "Кто отметил", "Комментарий", "Кадр"])
        f = lambda ts: datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M:%S") if ts else ""
        for r in rows:
            wr.writerow([r["id"], f(r["alert_ts"]), f(r["onset_ts"]), f(r["end_ts"]), r["camera_name"], r["area"],
                         r["zone"], r["title"], r["severity"], r["track_id"], r["status"], r["status_by"] or "",
                         r["comment"] or "", r["snapshot"] or ""])
        data = "﻿" + buf.getvalue()  # BOM — чтобы Excel открыл кириллицу
        return Response(data, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=journal.csv"})

    @app.get("/api/events/{eid}")
    def event(eid: int):
        r = core.db.get(eid)
        if not r:
            raise HTTPException(404)
        return core._pub(r)

    @app.patch("/api/events/{eid}")
    def patch_event(eid: int, body: dict = Body(...)):
        if not core.db.get(eid):
            raise HTTPException(404)
        upd = {}
        if "status" in body:
            if body["status"] not in ("new", "confirmed", "false", "forwarded"):
                raise HTTPException(400, "status")
            upd.update(status=body["status"], status_ts=time.time(), status_by=body.get("by", "оператор"))
        if "comment" in body:
            upd["comment"] = body["comment"]
        core.db.update(eid, **upd)
        rec = core._pub(core.db.get(eid))
        core.broadcast({"type": "event_update", "event": rec})
        return rec

    @app.post("/api/notify/test")
    def notify_test():
        if not core.notifiers:
            raise HTTPException(400, "Telegram не настроен: config/secrets.yaml (token, chat_id)")
        for n in core.notifiers:
            n.send({"id": 0, "type": "test", "title": "Проверка связи: ДОЗОР на связи", "severity": "warning",
                    "camera_name": "—", "camera_id": "—", "area": "", "zone": "", "alert_ts": time.time(),
                    "info": "Так будут выглядеть тревоги для мастера смены."})
        return {"ok": True, "chats": sum(len(n.chats) for n in core.notifiers)}

    @app.get("/api/notify")
    def notify_status():
        return {"telegram": bool(core.notifiers), "chats": sum(len(n.chats) for n in core.notifiers)}

    @app.get("/report")
    def report(hours: float = 12, camera: str = None):
        """Печатный отчёт по охране труда за смену (Ctrl+P → PDF)."""
        import html as H
        t1 = time.time()
        t0 = t1 - hours * 3600
        rows = core.db.query(camera, None, None, None, None, t0, t1, 5000, 0)
        real = [r for r in rows if r["status"] != "false"]
        f = lambda ts: datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if ts else "—"
        by = {}
        for r in real:
            by.setdefault(r["title"], []).append(r)
        zones = {}
        for r in real:
            k = f'{r["camera_name"]}{" / " + r["zone"] if r["zone"] else ""}'
            zones[k] = zones.get(k, 0) + 1
        crit = sum(1 for r in real if r["severity"] == "critical")
        open_n = sum(1 for r in real if r["status"] == "new")
        lat = [r["alert_ts"] - r["onset_ts"] for r in real if r["onset_ts"]]
        st = {"new": "не обработано", "confirmed": "подтверждено", "forwarded": "передано мастеру", "false": "ложное"}
        trs = "".join(
            f'<tr><td>{r["id"]}</td><td>{f(r["alert_ts"])}</td><td>{H.escape(r["title"])}</td><td>{H.escape(r["camera_name"])}'
            f'{"<br><small>" + H.escape(r["zone"]) + "</small>" if r["zone"] else ""}</td><td>{st.get(r["status"], r["status"])}</td>'
            f'<td>{H.escape(r["comment"] or "")}</td><td>{"<img src=/media/" + r["snapshot"] + ">" if r["snapshot"] else ""}</td></tr>'
            for r in real[:300])
        types = "".join(f"<li><b>{H.escape(k)}</b> — {len(v)}</li>" for k, v in sorted(by.items(), key=lambda x: -len(x[1])))
        top = "".join(f"<li>{H.escape(k)} — {n}</li>" for k, n in sorted(zones.items(), key=lambda x: -x[1])[:5])
        doc = f"""<!doctype html><html lang=ru><meta charset=utf-8><title>Отчёт ДОЗОР за смену</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#111}}h1{{margin:0}}.k{{display:flex;gap:16px;margin:16px 0}}
.k div{{border:1px solid #ccc;border-radius:8px;padding:10px 16px}}.k b{{font-size:26px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}}img{{width:120px}}small{{color:#666}}
@media print{{button{{display:none}}}}</style>
<button onclick="print()">Печать / сохранить в PDF</button>
<h1>Отчёт по нарушениям ТБ за смену</h1>
<p>Период: {f(t0)} — {f(t1)}{" · камера " + H.escape(camera) if camera else ""} · сформирован системой ДОЗОР</p>
<div class=k><div><b>{len(real)}</b>нарушений</div><div><b>{crit}</b>критичных</div><div><b>{open_n}</b>без реакции</div>
<div><b>{(sum(lat) / len(lat) if lat else 0):.1f} с</b>среднее время до тревоги</div><div><b>{len(rows) - len(real)}</b>ложных (исключены)</div></div>
<h3>По типам</h3><ul>{types or "<li>нарушений нет</li>"}</ul><h3>Проблемные места</h3><ol>{top or "<li>—</li>"}</ol>
<h3>Журнал</h3><table><tr><th>№</th><th>Время</th><th>Нарушение</th><th>Камера / зона</th><th>Реакция</th><th>Комментарий</th><th>Кадр</th></tr>{trs}</table>
<p><small>Подпись ответственного: ____________________</small></p></html>"""
        return Response(doc, media_type="text/html; charset=utf-8")

    @app.get("/api/stats")
    def stats(days: float = 30):
        return core.db.stats(time.time() - days * 86400)

    @app.get("/api/metrics")
    def metrics():
        return core.metrics()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        core.clients.add(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "metrics", "metrics": core.metrics()}, ensure_ascii=False))
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            core.clients.discard(websocket)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="открыть браузер, когда сервер будет готов")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    if args.open:
        def _open_when_ready():
            import socket
            import webbrowser
            for _ in range(240):
                try:
                    socket.create_connection(("127.0.0.1", args.port), timeout=0.5).close()
                    time.sleep(1.0)
                    webbrowser.open(f"http://localhost:{args.port}")
                    return
                except OSError:
                    time.sleep(0.5)

        threading.Thread(target=_open_when_ready, daemon=True).start()
    print(f"ДОЗОР запускается: http://localhost:{args.port}  (остановить — Ctrl+C или закрыть окно)")
    uvicorn.run(create_app(os.path.abspath(args.config)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
