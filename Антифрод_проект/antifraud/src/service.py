"""
Сервис скоринга транзакций в реальном времени (упрощённо, без веб-фреймворка).
В продакшене это был бы обработчик REST/gRPC-запроса от процессинга;
здесь — класс, который принимает одну транзакцию и отвечает за миллисекунды.
"""
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from features import make_features

RAW_FIELDS = ["amount", "hour", "mcc", "is_new_device", "distance_km",
              "tx_count_1h", "is_foreign", "account_age_days", "client_avg_amount"]


class ScoringService:
    def __init__(self, model_path, threshold, protected=False,
                 max_declines=5, window_sec=3600):
        self.clf = joblib.load(model_path)
        self.thr = threshold
        self.protected = protected
        self.max_declines = max_declines
        self.window = window_sec
        self.declines = defaultdict(deque)   # card_id -> время отказов
        self.blocked = set()

    def _check(self, tx: dict):
        # простая валидация входа: отрицательные суммы, чужие категории и т.п.
        if tx["amount"] <= 0 or not 0 <= tx["hour"] <= 23 or not 0 <= tx["mcc"] <= 7:
            raise ValueError("некорректная транзакция")

    def handle(self, tx: dict, card_id="card", now=None):
        now = time.time() if now is None else now
        self._check(tx)
        if self.protected and card_id in self.blocked:
            return {"decision": "blocked"}

        X = make_features(pd.DataFrame([{k: tx[k] for k in RAW_FIELDS}]))[self.clf.used_features_]
        p = float(self.clf.predict_proba(X)[0, 1])
        decision = "decline" if p >= self.thr else "approve"

        if not self.protected:
            # открытый режим: наружу уходит и сам скор — этим пользуется атака
            return {"decision": decision, "risk_score": round(p, 4)}

        if decision == "decline":
            q = self.declines[card_id]
            q.append(now)
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_declines:
                self.blocked.add(card_id)
        return {"decision": decision}


def benchmark(service, raw, n=1000):
    """Замер задержки на одиночных запросах (как в реальном потоке)."""
    rows = raw.sample(n, random_state=1)[RAW_FIELDS].to_dict("records")
    times = []
    for i, tx in enumerate(rows):
        t0 = time.perf_counter()
        service.handle(tx, card_id=f"c{i}")
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2),
            "p99_ms": round(float(np.percentile(t, 99)), 2)}
