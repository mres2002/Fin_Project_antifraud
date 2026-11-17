"""
Генерация синтетического набора платёжных транзакций.

Настоящие банковские данные в открытом доступе почти не встречаются
(в датасете Kaggle Credit Card Fraud признаки обезличены через PCA,
поэтому по ним нельзя понять, чем управляет злоумышленник). Мы сделали
свой генератор по мотивам PaySim: признаки понятные, и для каждого
известно, может ли мошенник его поменять.
"""
import numpy as np
import pandas as pd

# категории продавцов (MCC укрупнённо)
MCC_NAMES = ["продукты", "кафе", "транспорт", "одежда",
             "электроника", "путешествия", "p2p_перевод", "игры_крипто"]

# насколько категория «любима» мошенниками (подобрано вручную)
MCC_FRAUD_WEIGHT = np.array([0.2, 0.3, 0.2, 0.6, 1.5, 1.0, 1.8, 2.0])


def generate(n=120_000, fraud_rate=0.02, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    # ---------- легитимные операции ----------
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=7.0, sigma=1.0, size=n_legit),  # ~1100 руб медиана
        "hour": rng.choice(24, size=n_legit, p=_day_profile()),
        "mcc": rng.choice(8, size=n_legit, p=[0.3, 0.17, 0.15, 0.1, 0.06, 0.04, 0.15, 0.03]),
        "is_new_device": rng.binomial(1, 0.03, n_legit),
        "distance_km": rng.exponential(8, n_legit),
        "tx_count_1h": rng.poisson(0.8, n_legit) + 1,
        "is_foreign": rng.binomial(1, 0.02, n_legit),
        "account_age_days": rng.integers(30, 3650, n_legit),
    })
    legit["client_avg_amount"] = legit["amount"] * rng.lognormal(0, 0.5, n_legit)

    # ---------- мошеннические операции ----------
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=8.6, sigma=1.1, size=n_fraud),  # заметно крупнее
        "hour": rng.choice(24, size=n_fraud, p=_night_profile()),
        "mcc": rng.choice(8, size=n_fraud, p=MCC_FRAUD_WEIGHT / MCC_FRAUD_WEIGHT.sum()),
        "is_new_device": rng.binomial(1, 0.55, n_fraud),
        "distance_km": rng.exponential(120, n_fraud),
        "tx_count_1h": rng.poisson(3.0, n_fraud) + 1,
        "is_foreign": rng.binomial(1, 0.15, n_fraud),
        # 40% мошенничества идёт через свежие «счета-дропы»
        "account_age_days": np.where(rng.random(n_fraud) < 0.4,
                                     rng.integers(1, 60, n_fraud),
                                     rng.integers(30, 3650, n_fraud)),
    })
    # средний чек клиента не зависит от мошенника, берём «обычный»
    fraud["client_avg_amount"] = rng.lognormal(7.0, 1.0, n_fraud)

    legit["is_fraud"] = 0
    fraud["is_fraud"] = 1
    df = pd.concat([legit, fraud], ignore_index=True)

    # немного шума в метках: часть мошенничества выглядит как обычная покупка
    # (в жизни так и бывает, плюс модель не должна получаться идеальной)
    mask = (df.is_fraud == 1) & (rng.random(len(df)) < 0.15)
    df.loc[mask, ["amount", "hour", "distance_km", "tx_count_1h", "is_new_device"]] = \
        legit.sample(mask.sum(), random_state=seed)[
            ["amount", "hour", "distance_km", "tx_count_1h", "is_new_device"]].values

    df["amount"] = df["amount"].round(2)
    df["distance_km"] = df["distance_km"].round(1)
    df["timestamp"] = _timestamps(len(df), rng)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "tx_id", np.arange(len(df)))
    return df


def _day_profile():
    p = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 4, 5, 5, 6,
                  7, 6, 6, 6, 6, 7, 8, 8, 7, 5, 3, 2], dtype=float)
    return p / p.sum()


def _night_profile():
    p = np.array([5, 6, 6, 5, 4, 3, 2, 2, 2, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5], dtype=float)
    return p / p.sum()


def _timestamps(n, rng):
    # 60 дней операций
    start = pd.Timestamp("2026-08-01")
    secs = rng.integers(0, 60 * 24 * 3600, n)
    return start + pd.to_timedelta(secs, unit="s")


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(df.shape, "доля мошенничества:", round(df.is_fraud.mean(), 4))
