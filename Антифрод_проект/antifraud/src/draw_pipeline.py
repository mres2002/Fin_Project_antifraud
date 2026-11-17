import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6)
steps = [("Транзакция\nот процессинга", "#e8eef6"), ("Валидация\nвхода", "#e8eef6"),
         ("Расчёт\nпризнаков", "#e8eef6"), ("Модель\n(бустинг)", "#dfe9d3"),
         ("Порог\nFPR ≤ 1%", "#dfe9d3"), ("Ответ:\napprove /\ndecline", "#f6e2da")]
w, h, y = 1.6, 1.3, 1.9
for i, (t, c) in enumerate(steps):
    x = 0.2 + i * 1.97
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#555"))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=11)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.97, y + h/2), xytext=(x + w + 0.05, y + h/2),
                    arrowprops=dict(arrowstyle="->", color="#333"))
ax.add_patch(FancyBboxPatch((6.1, 0.2), 5.6, 1.1, boxstyle="round,pad=0.05", fc="#fff6e0", ec="#b08a2e", ls="--"))
ax.text(8.9, 0.75, "Защита: устойчивые признаки + адверсариальное обучение,\nскор не отдаётся наружу, блокировка после 5 отказов",
        ha="center", va="center", fontsize=10)
ax.text(0.2, 0.75, "Офлайн: генерация данных → разбиение по времени\n→ обучение → подбор порога на валидации → тест",
        ha="left", va="center", fontsize=10, color="#333")
plt.tight_layout(); plt.savefig("results/plots/pipeline.png", dpi=150)
