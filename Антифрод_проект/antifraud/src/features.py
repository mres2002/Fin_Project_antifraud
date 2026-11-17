"""Признаки модели и правила, что из них может менять злоумышленник."""
import numpy as np
import pandas as pd

FEATURES = ["log_amount", "amount_to_avg", "hour_sin", "hour_cos", "is_night",
            "mcc", "is_new_device", "log_distance", "tx_count_1h",
            "is_foreign", "log_account_age"]

# признаки для защищённой модели: убираем те, что злоумышленник
# меняет бесплатно и мгновенно (время операции и категория продавца)
ROBUST_FEATURES = [f for f in FEATURES
                   if f not in ("hour_sin", "hour_cos", "is_night", "mcc")]

# сырые поля, которые мошенник реально контролирует:
#  amount       — может дробить сумму, но не меньше 30% от задуманной
#  hour         — может выбрать время операции
#  mcc          — может выбрать магазин/способ вывода
#  tx_count_1h  — может делать операции реже
#  distance_km  — может подменить геолокацию через прокси (уменьшить)
# остальное (возраст счёта, средний чек клиента) ему недоступно
CONTROLLABLE = ["amount", "hour", "mcc", "tx_count_1h", "distance_km"]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=raw.index)
    X["log_amount"] = np.log1p(raw["amount"])
    X["amount_to_avg"] = raw["amount"] / (raw["client_avg_amount"] + 1)
    X["hour_sin"] = np.sin(2 * np.pi * raw["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * raw["hour"] / 24)
    X["is_night"] = ((raw["hour"] <= 5) | (raw["hour"] >= 23)).astype(int)
    X["mcc"] = raw["mcc"].astype(int)
    X["is_new_device"] = raw["is_new_device"].astype(int)
    X["log_distance"] = np.log1p(raw["distance_km"])
    X["tx_count_1h"] = raw["tx_count_1h"]
    X["is_foreign"] = raw["is_foreign"].astype(int)
    X["log_account_age"] = np.log1p(raw["account_age_days"])
    return X[FEATURES]


def time_split(df, test_share=0.25):
    """Делим по времени, а не случайно: модель учится на прошлом,
    проверяется на будущем — так честнее для антифрода."""
    cut = int(len(df) * (1 - test_share))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
