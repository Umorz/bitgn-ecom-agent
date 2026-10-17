# -*- coding: utf-8 -*-
"""Статистика прогонов BitGN: разброс, доверительные интервалы, парные тесты.

   python stats.py A=<маска> [B=<маска>] [SET=analysis|holdout]
   Маска - шаблон папок серий в папке прогонов (BITGN_RUNS), например "*_qwen_e*_PROD".
   SET - считать только по половине prod из prod_split.json.

   Для группы: балл каждого прогона, среднее ± ст.откл., 95% ДИ бутстрепом по
   задачам, доля успеха по задачам, нестабильные задачи (решаются через раз).
   Для двух групп: разница долей успеха по задачам и её значимость -
   перестановочный тест со сменой знака (повторы учитываются как доли), а при
   одном прогоне в каждой группе - ещё точный тест Макнемара.
   Зачем: один прогон даёт разброс в несколько задач, и без этого случайность
   легко принять за эффект правки."""
import glob, json, math, os, random, sys
from statistics import mean, pstdev

# папка прогонов: BITGN_RUNS или runs/ в корне проекта (в git не попадает)
RUNS = os.environ.get("BITGN_RUNS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
for s in (sys.stdout,):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass


from task_types import is_broken

SUBSET = None   # SET=analysis|holdout - считать только по половине prod из prod_split.json


def load_group(pattern):
    """Прогоны по маске: (имя папки, {задача: балл}). Сломанные прогоны пропускаются.

    Берутся все задачи, а не только done: невыполненная задача - это балл 0, как у BitGN."""
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, pattern))):
        for rp in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            trials = json.load(open(rp, encoding="utf-8"))["trials"]
            if is_broken(trials):
                print(f"   пропущен сломанный прогон {os.path.basename(d)}")
                continue
            tasks = {t["task"]: t["score"] for t in trials if SUBSET is None or t["task"] in SUBSET}
            if tasks:
                runs.append((os.path.basename(d), tasks))
    return runs


def run_score(tasks):
    """Балл прогона как у BitGN: средний балл по задачам (dev - 55, prod - 100)."""
    return sum(tasks.values()) / len(tasks)


def pass_rates(runs):
    """Доля успеха по задаче: среднее балла задачи по прогонам группы."""
    tasks = sorted({t for _, r in runs for t in r}, key=lambda x: int(x[1:]))
    return {t: mean(r.get(t, 0.0) for _, r in runs) for t in tasks}


def bootstrap_ci(values, n=10000, seed=1):
    rnd = random.Random(seed)
    k = len(values)
    means = sorted(mean(rnd.choice(values) for _ in range(k)) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def describe(name, runs):
    scores = [run_score(r) for _, r in runs]
    print(f"\n== {name}: {len(runs)} прогон(ов)")
    for (label, r), s in zip(runs, scores):
        print(f"   {label:55s} балл {s:.3f}  решено {sum(v >= 1 for v in r.values())}/{len(r)}")
    pr = pass_rates(runs)
    lo, hi = bootstrap_ci(list(pr.values()))
    sd = pstdev(scores) if len(scores) > 1 else float("nan")
    print(f"   среднее {mean(scores):.3f} ± {sd:.3f} (ст.откл. между прогонами); "
          f"95% ДИ по задачам [{lo:.3f}; {hi:.3f}]")
    unstable = [t for t, v in pr.items() if 0 < v < 1]
    solid = [t for t, v in pr.items() if v >= 1]
    print(f"   всегда решены: {len(solid)}; нестабильные ({len(unstable)}): "
          + ", ".join(f"{t}={pr[t]:.2f}" for t in unstable))
    return pr


def sign_flip_test(diffs, n=20000, seed=2):
    """Перестановочный тест: случайно меняем знак разниц по задачам."""
    rnd = random.Random(seed)
    obs = abs(sum(diffs))
    hits = sum(abs(sum(d if rnd.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def mcnemar_exact(b, c):
    """b - решено только в A, c - только в B. Двусторонний точный биномиальный тест."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a_runs, b_runs):
    pa, pb = pass_rates(a_runs), pass_rates(b_runs)
    tasks = sorted(set(pa) & set(pb), key=lambda x: int(x[1:]))
    diffs = [pb[t] - pa[t] for t in tasks]
    print(f"\n== Сравнение B - A по {len(tasks)} общим задачам")
    print(f"   средняя доля успеха: A {mean(pa[t] for t in tasks):.3f}, B {mean(pb[t] for t in tasks):.3f}, "
          f"разница {mean(diffs):+.3f} (≈ {sum(diffs):+.1f} задачи)")
    print(f"   перестановочный тест: p = {sign_flip_test(diffs):.3f}")
    if len(a_runs) == 1 and len(b_runs) == 1:
        b = sum(1 for t in tasks if pa[t] >= 1 and pb[t] < 1)
        c = sum(1 for t in tasks if pb[t] >= 1 and pa[t] < 1)
        print(f"   Макнемар: только в A {b}, только в B {c}, p = {mcnemar_exact(b, c):.3f}")
    better = [t for t in tasks if pb[t] - pa[t] >= 0.5]
    worse = [t for t in tasks if pa[t] - pb[t] >= 0.5]
    print(f"   заметно лучше в B: {better}\n   заметно хуже в B: {worse}")
    print("   p < 0.05 - разница вряд ли случайна; p > 0.05 - от шума не отличить")


if __name__ == "__main__":
    groups = dict(a.split("=", 1) for a in sys.argv[1:])
    if "SET" in groups:
        part = groups.pop("SET")
        SUBSET = set(json.load(open(os.path.join(os.path.dirname(__file__), "prod_split.json"), encoding="utf-8"))[part])
        print(f"считаю только по части {part}: {len(SUBSET)} задач")
    loaded = {k: load_group(v) for k, v in groups.items()}
    for k, runs in loaded.items():
        if not runs:
            sys.exit(f"группа {k}: не найдено прогонов по маске {groups[k]}")
        describe(k, runs)
    if "A" in loaded and "B" in loaded:
        compare(loaded["A"], loaded["B"])
