"""
Запуск всего проекта:  python src/main.py
1) генерация данных  2) базовая модель  3) атака на неё
4) защищённая модель  5) повтор атак  6) замер задержки  7) графики
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, os.path.dirname(__file__))
from data import generate, MCC_NAMES
from features import time_split
from model import train_model, score, pick_threshold, metrics
from attack import greedy_attack, label_only_attack
from defense import adversarial_training
from service import ScoringService, benchmark

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
PLOTS = "results/plots"
for folder in ["data", "models", PLOTS]:
    os.makedirs(folder, exist_ok=True)
res = {}

# ---------- 1. данные ----------
df = generate()
df.to_csv("data/transactions.csv", index=False)
train_full, test = time_split(df, 0.25)
train, val = time_split(train_full, 0.2)
print(f"всего {len(df)}, train {len(train)}, val {len(val)}, test {len(test)}, "
      f"fraud {df.is_fraud.mean():.2%}")
res["data"] = {"n": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
               "fraud_rate": round(float(df.is_fraud.mean()), 4),
               "n_fraud": int(df.is_fraud.sum())}

# ---------- 2. базовая модель ----------
print("\n[1] базовая модель")
base = train_model(train)
thr_b = pick_threshold(base, val)
res["baseline"] = metrics(base, test, thr_b)
print(res["baseline"])
joblib.dump(base, "models/baseline.joblib")

# важность признаков перестановкой: на что опирается базовая модель
from sklearn.inspection import permutation_importance
from features import make_features
Xt = make_features(test)[base.used_features_]
imp = permutation_importance(base, Xt, test.is_fraud, scoring="average_precision",
                             n_repeats=3, random_state=0)
res["baseline_importance"] = {f: round(float(v), 3) for f, v in
                              sorted(zip(Xt.columns, imp.importances_mean), key=lambda t: -t[1])}
print("важность признаков:", res["baseline_importance"])

# ---------- 3. атака на базовую модель ----------
print("\n[2] атака на базовую модель (API отдаёт скор)")
fraud_test = test[test.is_fraud == 1]
det_b = fraud_test[score(base, fraud_test) >= thr_b]
adv_b, st = greedy_attack(lambda d: score(base, d), det_b, thr_b)
res["attack_baseline_score"] = st
print(st)
res["attack_baseline_label5"] = label_only_attack(
    lambda d: (score(base, d) >= thr_b).astype(int), det_b, max_queries=5)[1]

# пример одной атакованной транзакции для отчёта
s_after = score(base, adv_b)
i = int(np.argmax((s_after < thr_b) * score(base, det_b)))  # самая «уверенная» из обойдённых
ex_before, ex_after = det_b.iloc[i], adv_b.iloc[i]
res["example"] = {
    "before": {k: (round(float(ex_before[k]), 2)) for k in
               ["amount", "hour", "mcc", "tx_count_1h", "distance_km", "is_new_device"]},
    "after": {k: (round(float(ex_after[k]), 2)) for k in
              ["amount", "hour", "mcc", "tx_count_1h", "distance_km", "is_new_device"]},
    "score_before": round(float(score(base, det_b.iloc[[i]])[0]), 4),
    "score_after": round(float(s_after[i]), 4),
}

# ---------- 4. защищённая модель ----------
print("\n[3] адверсариальное обучение")
prot, thr_p = adversarial_training(train, val, rounds=3)
res["protected"] = metrics(prot, test, thr_p)
print(res["protected"])
joblib.dump(prot, "models/protected.joblib")

# ---------- 5. атаки на защищённую модель ----------
print("\n[4] атаки на защищённую модель")
det_p = fraud_test[score(prot, fraud_test) >= thr_p]
# а) худший случай: скор каким-то образом утёк, атакующий адаптивный
_, st = greedy_attack(lambda d: score(prot, d), det_p, thr_p)
res["attack_protected_score"] = st
# б) реальный режим сервиса: только решение + блокировка после 5 отказов
res["attack_protected_label5"] = label_only_attack(
    lambda d: (score(prot, d) >= thr_p).astype(int), det_p, max_queries=5)[1]
print(res["attack_protected_score"], res["attack_protected_label5"])

# проверка через сам сервис на 200 картах (лимит отказов работает «вживую»)
svc = ScoringService("models/protected.joblib", thr_p, protected=True, max_declines=5)
blocked = 0
for k, (_, tx) in enumerate(det_p.head(200).iterrows()):
    for attempt in range(10):
        t = tx.to_dict()
        t["amount"] *= 0.8 ** attempt
        t["hour"] = 13
        r = svc.handle(t, card_id=f"card{k}", now=attempt * 60)
        if r["decision"] != "decline":
            break
    blocked += r["decision"] == "blocked"
res["service_check"] = {"cards": 200, "blocked": blocked}

# ---------- 6. задержка ----------
res["latency_open"] = benchmark(ScoringService("models/baseline.joblib", thr_b), test)
res["latency_protected"] = benchmark(
    ScoringService("models/protected.joblib", thr_p, protected=True), test)
print("\nзадержка:", res["latency_open"], res["latency_protected"])

with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------- 7. графики ----------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#3b6ea5", "#d1603d", "#7a9a3a"

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(np.log10(df[df.is_fraud == 0].amount), bins=60, alpha=.7, density=True, label="обычные", color=C1)
ax[0].hist(np.log10(df[df.is_fraud == 1].amount), bins=60, alpha=.7, density=True, label="мошенничество", color=C2)
ax[0].set_xlabel("log10(сумма, руб.)"); ax[0].set_title("Распределение сумм"); ax[0].legend()
rate = df.groupby("hour").is_fraud.mean() * 100
ax[1].bar(rate.index, rate.values, color=C2)
ax[1].set_xlabel("час операции"); ax[1].set_ylabel("% мошенничества"); ax[1].set_title("Доля мошенничества по часам")
plt.tight_layout(); plt.savefig(f"{PLOTS}/data_overview.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
r = df.groupby("mcc").is_fraud.mean() * 100
ax.barh([MCC_NAMES[i] for i in r.index], r.values, color=C1)
ax.set_xlabel("% мошенничества"); ax.set_title("Доля мошенничества по категориям продавцов")
plt.tight_layout(); plt.savefig(f"{PLOTS}/mcc.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
y = test.is_fraud.values
for clf, name, c in [(base, "базовая", C1), (prot, "защищённая", C3)]:
    s = score(clf, test)
    fpr, tpr, _ = roc_curve(y, s); ax[0].plot(fpr, tpr, label=name, color=c)
    p, rc, _ = precision_recall_curve(y, s); ax[1].plot(rc, p, label=name, color=c)
ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=.8)
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC-кривая"); ax[0].legend()
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("PR-кривая"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{PLOTS}/roc_pr.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.2))
labels = ["Базовая\nскор открыт", "Базовая\n5 попыток", "Защищённая\nскор открыт", "Защищённая\n5 попыток"]
keys = ["attack_baseline_score", "attack_baseline_label5", "attack_protected_score", "attack_protected_label5"]
asr = [res[k]["success_rate"] * 100 for k in keys]
money = [res[k]["money_passed_share"] * 100 for k in keys]
x = np.arange(4)
b1 = ax.bar(x - .2, asr, .4, label="успешных обходов, %", color=C2)
b2 = ax.bar(x + .2, money, .4, label="проведено денег, % от задуманного", color=C1)
ax.bar_label(b1, fmt="%.0f"); ax.bar_label(b2, fmt="%.0f")
ax.set_xticks(x, labels); ax.set_ylim(0, 105); ax.legend(); ax.set_title("Результат атаки уклонения")
plt.tight_layout(); plt.savefig(f"{PLOTS}/attack_results.png", dpi=150); plt.close()

print("\nготово, результаты в results/")
