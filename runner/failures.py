# -*- coding: utf-8 -*-
"""Отчёт о провалах для разбора правил: что спросили, что агент делал и ответил,
на что сослался и что сказал BitGN.

   python failures.py <маска серий> [задачи...]

   По каждой проваленной задаче печатает: вопрос
   целиком, шаги агента (вызов и начало вывода, ошибки тоже), итоговый отчёт,
   ссылки и вердикт BitGN. Если задача прогонялась в нескольких сериях, выводит
   каждую попытку - видно, стабилен ли провал."""
import glob, json, os, re, sys

# папка прогонов: BITGN_RUNS или runs/ в корне проекта (в git не попадает)
RUNS = os.environ.get("BITGN_RUNS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
OUT_LIMIT = 220
for s in (sys.stdout,):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass


def steps(part):
    out = []
    for n, st in enumerate(re.split(r"\nNext step_\d+\.\.\. ", part)[1:], 1):
        call = re.search(r"\n  (.*)", st)
        call = call.group(1).strip() if call else ""
        if call.startswith("tool='report_completion'"):
            break
        res = re.search(r"\n(OUT|ERR[^:]*): (.*?)(?=\n(?:Next step_|AUTO|FORMAT|MASKED|CUT|LOOP|BLOCKED)|\Z)", st, re.S)
        kind, body = (res.group(1), res.group(2).strip()) if res else ("", "")
        if "<<'SQL'" in body.split("\n", 1)[0] and "\nSQL\n" in body:
            body = body.split("\nSQL\n", 1)[1]
        elif body.startswith(("/bin/", "cat ", "ls ", "tree", "rg ", "sed ")) and "\n" in body:
            body = body.split("\n", 1)[1]
        body = body.strip().replace("\n", " | ")
        flag = "!! " if kind.startswith("ERR") or "stderr:" in body or "[exit " in body else ""
        extra = " [refs rejected by harness]" if "REFERENCE CHECK FAILED" in st else ""
        out.append(f"  {n}. {call[:240]}\n     {flag}-> {body[:OUT_LIMIT]}{extra}")
    return out


if __name__ == "__main__":
    pattern, only = sys.argv[1], set(sys.argv[2:])
    for d in sorted(glob.glob(os.path.join(RUNS, pattern))):
        for rp in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            m = os.path.dirname(rp)
            res = {t["task"]: t for t in json.load(open(rp, encoding="utf-8"))["trials"] if t["done"]}
            text = re.sub(r"\x1b\[[0-9;]*m", "", open(os.path.join(m, "console.log"), encoding="utf-8", errors="replace").read())
            solved = sorted(k for k, t in res.items() if t["score"] >= 1 and (not only or k in only))
            print(f"\n######## {os.path.basename(d)}: решены {solved}")
            for part in re.split(r"=+ Starting task: ", text)[1:]:
                tid = part.split(" ", 1)[0]
                if (only and tid not in only) or tid not in res or res[tid]["score"] >= 1:
                    continue
                q = part.split("\n", 2)[1].strip()
                summ = re.findall(r"AGENT SUMMARY: (.*)", part)
                outcome = re.findall(r"agent (OUTCOME_\w+)", part)
                refs = re.findall(r"\n- (/\S+)", part.split("AGENT SUMMARY")[-1]) if summ else []
                print(f"\n=== {tid}  BitGN: {'; '.join(res[tid]['detail'])}")
                print(f"Q: {q}")
                print("\n".join(steps(part)))
                print(f"FINAL {outcome[-1] if outcome else '-'}: {(summ[-1] if summ else '-')[:400]}")
                print(f"REFS: {refs}")
