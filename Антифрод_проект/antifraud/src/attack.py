"""
Атака уклонения (evasion attack) на модель скоринга.

Сценарий: у злоумышленника есть данные чужой карты. Перед тем как
вывести деньги, он «пробует» операции и смотрит на реакцию системы.
Если API отдаёт числовой риск-скор (так часто делают для партнёров
и мобильного приложения), атакующий видит, какие изменения снижают
риск, и подгоняет операцию под порог.

Атакующий меняет только то, что ему доступно (см. features.CONTROLLABLE).
Реализован жадный поиск: на каждом шаге перебираем возможные правки,
оставляем ту, что сильнее всего снизила скор. Каждая проверка = 1 запрос.
"""
import numpy as np
import pandas as pd

DAY_HOURS = [10, 13, 16, 19]
SAFE_MCC = [0, 1, 2]          # продукты, кафе, транспорт
MIN_AMOUNT_SHARE = 0.3        # дробить сумму сильнее невыгодно
MIN_DISTANCE = 1.0


def _candidates(cur: pd.DataFrame, orig_amount: np.ndarray):
    """Все варианты «одной правки» для каждой строки."""
    out = []
    # 1) уменьшить сумму на 20%
    c = cur.copy()
    c["amount"] = np.maximum(c["amount"] * 0.8, orig_amount * MIN_AMOUNT_SHARE)
    out.append(c)
    # 2) перенести операцию на дневное время
    for h in DAY_HOURS:
        c = cur.copy(); c["hour"] = h; out.append(c)
    # 3) сменить категорию продавца
    for m in SAFE_MCC:
        c = cur.copy(); c["mcc"] = m; out.append(c)
    # 4) делать операции реже
    c = cur.copy(); c["tx_count_1h"] = np.maximum(c["tx_count_1h"] - 1, 1); out.append(c)
    # 5) прокси ближе к городу владельца карты
    c = cur.copy(); c["distance_km"] = np.maximum(c["distance_km"] / 3, MIN_DISTANCE); out.append(c)
    return out


def greedy_attack(score_fn, raw, thr, max_steps=12):
    """Атака с доступом к скору. Векторизована по всем транзакциям сразу.
    Возвращает изменённые транзакции и статистику."""
    cur = raw.copy().reset_index(drop=True)
    orig_amount = cur["amount"].values.copy()
    s = score_fn(cur)
    queries = np.ones(len(cur), dtype=int)
    active = s >= thr                      # атакуем только то, что блокируется

    for _ in range(max_steps):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        sub = cur.iloc[idx]
        cands = _candidates(sub, orig_amount[idx])
        scores = np.vstack([score_fn(c) for c in cands])   # (n_cand, n_active)
        queries[idx] += len(cands)
        best = scores.argmin(axis=0)
        best_s = scores[best, np.arange(len(idx))]
        improved = best_s < s[idx] - 1e-6
        for b in range(len(cands)):
            sel = np.where(improved & (best == b))[0]
            if len(sel):
                cur.iloc[idx[sel]] = cands[b].iloc[sel].values
        s[idx[improved]] = best_s[improved]
        # кто прошёл порог или застрял — выходит из атаки
        active[idx[~improved]] = False
        active[s < thr] = False

    return cur, _stats(raw, cur, s, thr, queries)


def label_only_attack(label_fn, raw, max_queries=5, seed=0):
    """Атака, когда API отвечает только «одобрено/отклонено» и после
    max_queries отказов карта блокируется. Атакующий перебирает
    случайные комбинации правок вслепую."""
    rng = np.random.default_rng(seed)
    raw = raw.reset_index(drop=True)
    best = raw.copy()
    passed = np.zeros(len(raw), dtype=bool)
    queries = np.zeros(len(raw), dtype=int)

    for q in range(max_queries):
        todo = np.where(~passed)[0]
        if len(todo) == 0:
            break
        c = raw.iloc[todo].copy()
        if q > 0:  # первая попытка — исходная операция
            share = rng.uniform(MIN_AMOUNT_SHARE, 1.0, len(c))
            c["amount"] = c["amount"] * share
            c["hour"] = rng.choice(DAY_HOURS, len(c))
            c["mcc"] = rng.choice(SAFE_MCC, len(c))
            c["tx_count_1h"] = np.maximum(c["tx_count_1h"] - rng.integers(0, 4, len(c)), 1)
            c["distance_km"] = np.maximum(c["distance_km"] / rng.uniform(1, 20, len(c)), MIN_DISTANCE)
        ok = label_fn(c) == 0
        queries[todo] += 1
        best.iloc[todo[ok]] = c[ok].values
        passed[todo[ok]] = True

    return best, {
        "n": int(len(raw)),
        "success_rate": round(float(passed.mean()), 4),
        "mean_queries": round(float(queries.mean()), 2),
        "money_passed_share": round(float(best.loc[passed, "amount"].sum() / raw["amount"].sum()), 4),
    }


def _stats(raw, adv, s, thr, queries):
    raw = raw.reset_index(drop=True)
    success = s < thr
    return {
        "n": int(len(raw)),
        "success_rate": round(float(success.mean()), 4),
        "mean_queries": round(float(queries.mean()), 1),
        # какую долю задуманной суммы удалось провести
        "money_passed_share": round(float(adv.loc[success, "amount"].sum() / raw["amount"].sum()), 4),
        "mean_amount_kept": round(float((adv.loc[success, "amount"] / raw.loc[success, "amount"]).mean()), 3)
        if success.any() else 0.0,
    }
