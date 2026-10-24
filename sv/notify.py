"""Оповещения во внешние каналы. Сейчас — Telegram-бот мастеру смены.

Включается в config/config.yaml:
  notify:
    telegram: {token: "123:ABC", chat_id: "-100..."}
или переменными окружения SV_TG_TOKEN / SV_TG_CHAT.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime

log = logging.getLogger("notify")

SEV_ICON = {"critical": "🔴", "warning": "🟠"}


class TelegramNotifier:
    def __init__(self, token: str, chat_id, base_url: str = "", types=None):
        self.token = token
        self.chats = [str(c) for c in (chat_id if isinstance(chat_id, (list, tuple)) else str(chat_id).split(","))]
        self.types = set(types) if types else None   # каким нарушениям слать (None — всем)
        self.base_url = base_url.rstrip("/")
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=100)
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, ev: dict):
        if self.types and ev.get("type") not in self.types and ev.get("type") != "test":
            return
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            pass

    def _run(self):
        import requests

        api = f"https://api.telegram.org/bot{self.token}"
        while True:
            ev = self.q.get()
            try:
                ts = datetime.fromtimestamp(ev["alert_ts"]).strftime("%H:%M:%S")
                text = (f"{SEV_ICON.get(ev['severity'], '⚠️')} <b>{ev['title']}</b>\n"
                        f"Камера: {ev['camera_name']} ({ev['camera_id']})\n"
                        f"Участок: {ev.get('area') or '—'}{(' / ' + ev['zone']) if ev.get('zone') else ''}\n"
                        f"Время: {ts}\n{ev.get('info') or ''}")
                if self.base_url:
                    text += f"\n<a href=\"{self.base_url}/#event={ev['id']}\">Открыть событие #{ev['id']}</a>"
                path = ev.get("snapshot_path")
                for chat in self.chats:
                    if path and os.path.exists(path):
                        with open(path, "rb") as f:
                            requests.post(f"{api}/sendPhoto", data={"chat_id": chat, "caption": text,
                                                                   "parse_mode": "HTML"}, files={"photo": f}, timeout=15)
                    else:
                        requests.post(f"{api}/sendMessage", data={"chat_id": chat, "text": text,
                                                                 "parse_mode": "HTML"}, timeout=15)
            except Exception as e:  # оповещение не должно ронять конвейер
                log.warning("telegram: %s", e)


def load_secrets() -> dict:
    """config/secrets.yaml — токены и id чатов; файл в .gitignore и не попадает в репозиторий."""
    import yaml

    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "secrets.yaml")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def make_notifiers(cfg: dict):
    out = []
    tg = dict((cfg or {}).get("telegram") or {})
    tg.update({k: v for k, v in (load_secrets().get("telegram") or {}).items() if v})
    token = tg.get("token") or os.environ.get("SV_TG_TOKEN")
    chat = tg.get("chat_id") or os.environ.get("SV_TG_CHAT")
    if token and chat:
        out.append(TelegramNotifier(token, chat, tg.get("base_url", ""), tg.get("types")))
    return out
