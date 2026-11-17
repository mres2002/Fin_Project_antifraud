"""Обучение модели скоринга, подбор порога и метрики."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

from features import make_features, FEATURES



def train_model(raw_train, extra_raw=None, seed=0, features=FEATURES):
    """raw_train — сырые транзакции с колонкой is_fraud.
    extra_raw — дополнительные примеры (для адверсариального обучения)."""
    data = raw_train if extra_raw is None else \
        __import__("pandas").concat([raw_train, extra_raw], ignore_index=True)
    X = make_features(data)[features]
    y = data["is_fraud"].values

    # мошенничества ~2%, поэтому даём классу 1 больший вес
    w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    w = np.sqrt(w)  # полный вес даёт слишком много ложных срабатываний

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, categorical_features=[f == "mcc" for f in features],
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    clf.fit(X, y, sample_weight=w)
    clf.used_features_ = list(features)   # запоминаем, на каких признаках учились
    return clf


def score(clf, raw):
    return clf.predict_proba(make_features(raw)[clf.used_features_])[:, 1]


def pick_threshold(clf, raw_val, max_fpr=0.01):
    """Порог, при котором блокируется не больше 1% честных операций.
    Для банка ложная блокировка — это недовольный клиент и звонок в поддержку,
    поэтому ограничиваем именно FPR."""
    s = score(clf, raw_val[raw_val.is_fraud == 0])
    return float(np.quantile(s, 1 - max_fpr))


def metrics(clf, raw, thr):
    y = raw["is_fraud"].values
    s = score(clf, raw)
    pred = (s >= thr).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, s), 4),
        "pr_auc": round(average_precision_score(y, s), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "fpr": round(float(pred[y == 0].mean()), 4),
        "threshold": round(thr, 4),
    }
