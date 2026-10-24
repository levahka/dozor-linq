"""Заполняет журнал ДЕМО-историей за прошлые дни, чтобы показать аналитику (графики, проблемные зоны, точность).

Все события помечаются в поле info как «[демо]», а камеры — «(демо)», чтобы их нельзя было спутать с реальными.
    python tools/seed_demo.py --days 30          # добавить
    python tools/seed_demo.py --clear            # удалить все демо-события
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sv.storage import Storage  # noqa: E402

CAMS = [("demo_sklad", "Склад — зона отгрузки (демо)", "Склад", ["Проезд погрузчика", "Стеллажи"]),
        ("demo_ceh1", "Цех 1 — пролёт А (демо)", "Цех 1", ["У пресса", ""]),
        ("demo_ceh2", "Цех 2 — конвейер (демо)", "Цех 2", ["Под конвейером", ""]),
        ("demo_dvor", "Двор — проезд техники (демо)", "Территория", ["Проезд", ""])]
TYPES = [("zone", "Вход в опасную зону", "critical", 0.34), ("running", "Бег по цеху", "warning", 0.2),
         ("proximity", "Опасное сближение с техникой", "critical", 0.16), ("phone", "Использование телефона", "warning", 0.22),
         ("fall", "Падение человека", "critical", 0.06), ("immobile", "Неподвижность, возможна потеря сознания", "critical", 0.02)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    db = Storage(os.path.join(ROOT, "data", "events.db"))
    if a.clear:
        with db.lock:
            n = db.db.execute("DELETE FROM events WHERE info LIKE '[демо]%'").rowcount
            db.db.commit()
        print("удалено демо-событий:", n)
        return
    rnd = random.Random(42)
    now = time.time()
    n = 0
    for day in range(a.days, 0, -1):
        base = now - day * 86400
        # тренд: после «инструктажа» на 15-й день нарушений на складе меньше
        for _ in range(rnd.randint(8, 16)):
            cam = rnd.choices(CAMS, weights=[5 if day > 15 else 2, 3, 2, 2])[0]
            t = rnd.choices(TYPES, weights=[x[3] for x in TYPES])[0]
            # больше нарушений в первые часы смен (8-10) и в ночную (0-3) на складе
            hour = rnd.choices(range(24), weights=[3, 3, 2, 1, 1, 1, 1, 2, 6, 6, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 2, 2, 2, 3])[0]
            ts = base - (base % 86400) + hour * 3600 + rnd.randint(0, 3599) - 3 * 3600
            zone = rnd.choice(cam[3]) if t[0] == "zone" else ""
            if t[0] == "zone" and not zone:
                zone = cam[3][0]
            delay = rnd.uniform(0.9, 2.2)
            dur = rnd.uniform(3, 60)
            st = rnd.choices(["confirmed", "forwarded", "false", "new"], weights=[60, 20, 9, 11])[0]
            db.add(camera_id=cam[0], camera_name=cam[1], area=cam[2], zone=zone, type=t[0], title=t[1], severity=t[2],
                   track_id=rnd.randint(1, 400), info="[демо] сгенерировано для показа аналитики",
                   onset_ts=ts - delay, alert_ts=ts, end_ts=ts + dur, status=st,
                   status_by="оператор" if st != "new" else None, status_ts=ts + 40 if st != "new" else None)
            n += 1
    print("добавлено демо-событий:", n)


if __name__ == "__main__":
    main()
