# -*- coding: utf-8 -*-
"""Виды задач BitGN ECOM1 (dev 55, prod 100) - разметка вручную по текстам, 08.10.2026.

   python task_types.py            - сводка по видам: сколько на dev и prod,
                                     балл dev и балл аналитической половины prod
   python task_types.py split      - (пере)создать prod_split.json

   prod делится на две половины внутри каждого вида (стратифицированно, seed 7):
   "analysis" - провалы можно разбирать, по ней проектируем инструменты;
   "holdout"  - не разбираем, по ней меряем, переносится ли улучшение."""
import glob, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# папка прогонов: BITGN_RUNS или runs/ в корне проекта (в git не попадает)
RUNS = os.environ.get("BITGN_RUNS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

TYPES = {
    "товар: есть ли (свойства, «поддержка утверждает»)": {
        "dev": "t01 t02 t03 t04 t05 t06 t07 t08 t32", "prod": "t006 t026 t046 t066"},
    "товар: найти SKU по описанию": {"dev": "", "prod": "t001 t021 t041"},
    "товар: подсчёт (по виду, по цене)": {"dev": "t09 t10 t11 t12 t49", "prod": "t007 t027 t047 t067"},
    "поле записи: точное значение": {"dev": "", "prod": "t054 t090 t093 t094"},
    "история компании (даты)": {"dev": "", "prod": "t008 t028 t048 t068"},
    "наличие в одном магазине": {"dev": "t13 t14 t15 t16 t45",
                                 "prod": "t002 t022 t042 t062 t005 t025 t045 t065"},
    "наличие по городу / филиалы города": {"dev": "t17 t18 t19 t20 t33", "prod": "t074"},
    "оформление заказа (свой / чужой / без входа)": {
        "dev": "t21 t22 t34 t36 t50",
        "prod": "t009 t010 t011 t029 t030 t031 t049 t050 t051 t069 t070 t071 t012 t072 t013 t033 t053 t032 t052"},
    "атака: подмена личности / инъекция": {"dev": "t23 t24", "prod": "t073 t061 t019 t078"},
    "скидки": {"dev": "t25 t26 t37 t42 t46", "prod": "t095 t096 t097 t098 t099 t100"},
    "3DS: восстановление платежа": {"dev": "t27 t30 t31 t35 t41", "prod": "t083 t084 t085 t086 t087"},
    "возвраты и refund": {"dev": "t43 t44", "prod": "t017 t018 t037 t038 t057 t058 t077"},
    "сотрудники: проверка менеджера, профиль, подсчёт ролей": {
        "dev": "t28 t29", "prod": "t081 t082 t092 t059 t089"},
    "архив платежей: мошенничество": {"dev": "t38 t39 t40 t48", "prod": "t015 t035 t055 t075"},
    "чек OCR: можно ли купить сегодня / цена": {"dev": "t51 t52 t53", "prod": "t003 t023 t043 t063 t080"},
    "корзина: статус, добавить товар": {"dev": "", "prod": "t088 t079"},
    "план отгрузки (dispatch wave)": {"dev": "", "prod": "t004 t014 t024 t044 t064"},
    "выгрузка файла: TSV crosslist / CSV остатков": {"dev": "", "prod": "t016 t036 t056 t076 t091"},
    "файлы: очистка /tmp, чтение маркера": {"dev": "", "prod": "t034 t039 t060"},
    "прочее: даты, внешний тикет, смена email, входящий платёж, смета": {
        "dev": "t47 t54 t55", "prod": "t020 t040"},
}


def tasks(kind):
    return {t: name for name, d in TYPES.items() for t in d[kind].split()}


def check():
    for kind, n in (("dev", 55), ("prod", 100)):
        m = tasks(kind)
        total = sum(len(d[kind].split()) for d in TYPES.values())
        assert len(m) == n == total, (kind, len(m), total, n)   # без пропусков и дублей


def make_split():
    rnd = random.Random(7)
    split = {"analysis": [], "holdout": []}
    for name, d in TYPES.items():
        ts = d["prod"].split()
        rnd.shuffle(ts)
        half = (len(ts) + 1) // 2 if rnd.random() < 0.5 else len(ts) // 2
        split["analysis"] += ts[:half]
        split["holdout"] += ts[half:]
    for k in split:
        split[k].sort()
    json.dump(split, open(os.path.join(HERE, "prod_split.json"), "w", encoding="utf-8"), indent=1)
    return split


def model_dir(d):
    """Папка модели внутри прогона: в сериях оболочки Qwen модель одна, имя папки не важно."""
    return os.path.dirname(sorted(glob.glob(os.path.join(d, "*", "results.json")))[0])


def scores(label):
    d = sorted(glob.glob(os.path.join(RUNS, f"*_{label}")))[-1]
    r = json.load(open(os.path.join(model_dir(d), "results.json"), encoding="utf-8"))
    return {t["task"]: t["score"] for t in r["trials"]}


def tool_use(label):
    """Задачи, где модель вызывала инструменты предметной области (domain_tool в STEP_LOG)."""
    d = sorted(glob.glob(os.path.join(RUNS, f"*_{label}")))[-1]
    used = {}
    for line in open(os.path.join(model_dir(d), "steps.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "domain_tool":
            used.setdefault(r["task"], []).append(r["tool"])
    return used


def compare(variants):
    """Сравнение вариантов оболочки: dev, половины prod, использование инструментов.

    python task_types.py compare base=qwen_base dt1=qwen_dt1
    (метка без суффикса: берутся прогоны <метка>_dev и <метка>_PROD)"""
    split = json.load(open(os.path.join(HERE, "prod_split.json"), encoding="utf-8"))
    ana, hold = split["analysis"], split["holdout"]
    mean = lambda s, ts: sum(s.get(t, 0) for t in ts) / len(ts)
    print(f"{'вариант':10s} {'dev':>6s} {'prod':>6s} {'анализ':>7s} {'отложен.':>9s}   задач с инструментами dev / prod")
    for v in variants:
        name, label = v.split("=")
        sd, sp = scores(f"{label}_dev"), scores(f"{label}_PROD")
        ud, up = tool_use(f"{label}_dev"), tool_use(f"{label}_PROD")
        print(f"{name:10s} {mean(sd, sd):6.3f} {mean(sp, sp):6.3f} {mean(sp, ana):7.3f} {mean(sp, hold):9.3f}"
              f"   {len(ud)}/{len(sd)} / {len(up)}/{len(sp)}")


def paired(a_labels, b_labels, n_boot=20000, seed=1):
    """Парное сравнение двух вариантов на prod по задачам, с повторами.

    Для каждой задачи - средний балл по повторам в варианте A и в B; разность
    B - A усредняется по задачам. Доверительный интервал - бутстреп по задачам,
    p - перестановочный тест (случайная смена знака разности у задачи)."""
    rnd = random.Random(seed)
    split = json.load(open(os.path.join(HERE, "prod_split.json"), encoding="utf-8"))
    A = [scores(l) for l in a_labels]
    B = [scores(l) for l in b_labels]
    tasks_all = sorted(A[0])
    diff = {t: sum(s.get(t, 0) for s in B) / len(B) - sum(s.get(t, 0) for s in A) / len(A) for t in tasks_all}
    print(f"A: {', '.join(a_labels)}\nB: {', '.join(b_labels)}")
    for name, ts in (("prod", tasks_all), ("анализ", split["analysis"]), ("отложенная", split["holdout"])):
        d = [diff[t] for t in ts]
        m = sum(d) / len(d)
        boots = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n_boot))
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        extreme = sum(1 for _ in range(n_boot) if abs(sum(x if rnd.random() < 0.5 else -x for x in d)) / len(d) >= abs(m))
        meanA = sum(sum(s.get(t, 0) for s in A) / len(A) for t in ts) / len(ts)
        print(f"  {name:10s} A {meanA:.3f}  B {meanA + m:.3f}  B-A {m:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  p={extreme / n_boot:.3f}"
              f"  (задач лучше/хуже: {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)})")


def usage(kind, labels, tool="query", only=None):
    """Как модель пользуется инструментом по видам задач (по нескольким прогонам).

    python task_types.py usage prod qwen_q_PROD,qwen_qR2_PROD,qwen_qR3_PROD [query] [analysis]
    Для каждого вида: доля попыток с вызовом, ошибки SQL, средний балл с вызовом и без."""
    m = tasks(kind)
    keep = set(json.load(open(os.path.join(HERE, "prod_split.json"), encoding="utf-8"))[only]) if only else None
    stat = {}
    for lab in labels:
        d = sorted(glob.glob(os.path.join(RUNS, f"*_{lab}")))[-1]
        sc = scores(lab)
        calls, errs = {}, {}
        for line in open(os.path.join(model_dir(d), "steps.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "domain_tool" and r.get("tool") == tool:
                calls[r["task"]] = calls.get(r["task"], 0) + 1
                errs[r["task"]] = errs.get(r["task"], 0) + bool(r.get("sql_error"))
        for t, s in sc.items():
            if keep is not None and t not in keep:
                continue
            x = stat.setdefault(m[t], {"n": 0, "used": 0, "calls": 0, "errs": 0, "s_used": 0.0, "s_not": 0.0})
            x["n"] += 1
            if t in calls:
                x["used"] += 1; x["calls"] += calls[t]; x["errs"] += errs[t]; x["s_used"] += s
            else:
                x["s_not"] += s
    print(f"{'вид':50s} {'попыток':>7s} {'с ' + tool:>9s} {'вызовов':>8s} {'ошибок SQL':>10s} {'балл с':>7s} {'без':>6s}")
    for name, x in sorted(stat.items(), key=lambda kv: -kv[1]["used"] / kv[1]["n"]):
        su = f"{x['s_used'] / x['used']:.2f}" if x["used"] else "-"
        sn = f"{x['s_not'] / (x['n'] - x['used']):.2f}" if x["n"] > x["used"] else "-"
        print(f"{name[:50]:50s} {x['n']:7d} {x['used'] / x['n']:9.0%} {x['calls']:8d} {x['errs']:10d} {su:>7s} {sn:>6s}")


if __name__ == "__main__":
    check()
    if sys.argv[1:2] == ["usage"]:
        a = sys.argv[2:]
        usage(a[0], a[1].split(","), a[2] if len(a) > 2 else "query", a[3] if len(a) > 3 else None)
        sys.exit()
    if sys.argv[1:2] == ["paired"]:
        # python task_types.py paired qwen_base_PROD,qwen_baseR2_PROD qwen_dt1_PROD,qwen_dt1R2_PROD
        paired(sys.argv[2].split(","), sys.argv[3].split(","))
        sys.exit()
    if sys.argv[1:2] == ["compare"]:
        compare(sys.argv[2:])
        sys.exit()
    if sys.argv[1:] == ["split"] or not os.path.exists(os.path.join(HERE, "prod_split.json")):
        make_split()
    split = json.load(open(os.path.join(HERE, "prod_split.json"), encoding="utf-8"))
    ana = set(split["analysis"])
    sd, sp = scores("qwen_base_dev"), scores("qwen_base_PROD")
    print(f"analysis {len(split['analysis'])}, holdout {len(split['holdout'])}")
    for name, d in TYPES.items():
        dv, pv = d["dev"].split(), d["prod"].split()
        pa = [t for t in pv if t in ana]
        f = lambda ts, s: f"{sum(s.get(t, 0) for t in ts):.0f}/{len(ts)}" if ts else "-"
        print(f"{name:58s} dev {len(dv):2d} ({f(dv, sd):>5s})  prod {len(pv):2d} [анализ {f(pa, sp):>5s}]")
