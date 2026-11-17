"""
Защита модели. Используем четыре меры, которые дополняют друг друга:

0. Отбор устойчивых признаков: защищённая модель не смотрит на час
   операции и категорию продавца — их атакующий меняет бесплатно.
   Остаются признаки, которые подделать дорого или нельзя
   (устройство, возраст счёта, сумма относительно привычной и т.д.).
1. Адверсариальное обучение: атакуем модель на обучающих данных,
   найденные «обходные» транзакции добавляем в обучение с меткой fraud
   и переобучаем. Делаем несколько раундов.
2. Скрытие скора: наружу отдаётся только решение (approve / decline),
   поэтому жадный поиск по скору становится невозможен.
3. Лимит отказов: после N отклонённых операций за час карта
   блокируется — у атакующего остаётся всего несколько попыток.
Меры 2 и 3 реализованы в service.py.
"""
import pandas as pd

from model import train_model, score, pick_threshold
from attack import greedy_attack
from features import ROBUST_FEATURES


def adversarial_training(raw_train, raw_val, rounds=3, max_fpr=0.01, log=print,
                         features=ROBUST_FEATURES):
    clf = train_model(raw_train, features=features)
    thr = pick_threshold(clf, raw_val, max_fpr)
    adv_pool = []

    for r in range(1, rounds + 1):
        fraud = raw_train[raw_train.is_fraud == 1]
        detected = fraud[score(clf, fraud) >= thr]
        adv, st = greedy_attack(lambda d: score(clf, d), detected, thr, max_steps=12)
        adv["is_fraud"] = 1
        adv_pool.append(adv)
        log(f"  раунд {r}: атака на train прошла у {st['success_rate']:.1%}, "
            f"добавлено {len(adv)} примеров")

        extra = pd.concat(adv_pool, ignore_index=True)
        clf = train_model(raw_train, extra_raw=extra, seed=r, features=features)
        thr = pick_threshold(clf, raw_val, max_fpr)

    return clf, thr
