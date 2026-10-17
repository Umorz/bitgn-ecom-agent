# -*- coding: utf-8 -*-
"""Диспетчер серии прогонов BitGN ECOM без присмотра.

   Запуск:  python dispatcher.py <план.json>
   План — список моделей, задач и вариант оболочки (см. plan_screen.json).

   Что делает для каждой модели по очереди:
     1. предпроверка: Ollama и прокси живы, закачек нет, GPU свободен и остыл;
     2. замер скорости и того, сколько модели влезло в видеопамять;
     3. прогон main.py с лимитами на задачу и на модель;
     4. выгрузка модели и охлаждение перед следующей.
   В конце пишет summary.md и summary.json в папку серии.
   Пока работает, запрещает Windows засыпать (закрытие крышки это не отменяет)."""
import ctypes, datetime, json, os, re, statistics, subprocess, sys, time, urllib.parse, urllib.request

# корень проекта - папка над runner/; остальные пути относительно него
ROOT      = os.environ.get("BITGN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
# Python агента: окружение uv в agent/.venv; BITGN_AGENT_PY - чтобы взять другое
AGENT_PY  = os.environ.get("BITGN_AGENT_PY") or os.path.join(AGENT_DIR, ".venv", "Scripts", "python.exe")
ENVFILE   = os.path.join(ROOT, ".env")
RUNS_DIR  = os.environ.get("BITGN_RUNS") or os.path.join(ROOT, "runs")
RESTART   = f'"{sys.executable}" "{os.path.join(ROOT, "runner", "restart_llm.py")}"'
OLLAMA    = "http://127.0.0.1:11434"
PROXY     = "http://127.0.0.1:11435"

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception: pass

LOG = None


def say(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def http_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def alive(url) -> bool:
    try:
        http_json(url, timeout=5); return True
    except Exception:
        return False


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        mem, util, temp = (int(x) for x in out.strip().split(","))
        return {"mem_mb": mem, "util": util, "temp": temp}
    except Exception:
        return None


def pulls_running() -> bool:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='ollama.exe'\" | "
          "Where-Object { $_.CommandLine -match ' pull ' } | Measure-Object | Select-Object -ExpandProperty Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
        return int(out.strip() or 0) > 0
    except Exception:
        return False


def unload_all():
    try:
        for m in http_json(OLLAMA + "/api/ps", timeout=10).get("models", []):
            http_json(OLLAMA + "/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def preflight(max_wait=900) -> list:
    """Ждёт тихую систему до max_wait секунд. Возвращает список нарушений (пустой — всё чисто)."""
    subprocess.run(RESTART, shell=True, capture_output=True, timeout=180)
    unload_all()
    deadline = time.time() + max_wait
    while True:
        problems = []
        if not alive(OLLAMA + "/api/version"): problems.append("ollama down")
        if not alive(PROXY + "/api/version"):  problems.append("proxy down")
        if pulls_running():                     problems.append("ollama pull running")
        g = gpu_state()
        if g is None:                           problems.append("nvidia-smi failed")
        else:
            if g["mem_mb"] > 600: problems.append(f"GPU memory busy {g['mem_mb']} MB")
            if g["util"] > 25:    problems.append(f"GPU util {g['util']}%")
            if g["temp"] > 70:    problems.append(f"GPU temp {g['temp']}C")
        if not problems or time.time() > deadline:
            return problems
        time.sleep(30)


PROBE_TEXT = ("The warehouse keeps an inventory ledger for every product line, store and city. "
              "Each record lists availability, reservations and damaged units. ") * 60


def probe(model: str) -> dict:
    """Скорость на промпте ~1.5k токенов и доля модели в видеопамяти."""
    r = http_json(OLLAMA + "/api/generate", {
        "model": model, "stream": False, "keep_alive": "10m",
        "prompt": PROBE_TEXT + "\nSummarize the ledger rules in five sentences.",
        "options": {"num_predict": 200, "temperature": 0},
    }, timeout=1800)
    res = {
        "prompt_tokens": r.get("prompt_eval_count"),
        "prompt_tps": round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9), 1) if r.get("prompt_eval_duration") else None,
        "gen_tokens": r.get("eval_count"),
        "gen_tps": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1) if r.get("eval_duration") else None,
        "load_s": round(r.get("load_duration", 0) / 1e9, 1),
    }
    for m in http_json(OLLAMA + "/api/ps", timeout=10).get("models", []):
        if m["name"].split(":")[0] == model.split(":")[0]:
            res["ctx"] = m.get("context_length")
            res["vram_share"] = round(m["size_vram"] / m["size"], 2) if m.get("size") else None
            res["size_gib"] = round(m["size"] / 2**30, 2)
    return res


def git_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=AGENT_DIR, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=AGENT_DIR, capture_output=True, text=True).stdout.strip()
        return h + ("-DIRTY" if dirty else "")
    except Exception:
        return "unknown"


def read_env(name: str) -> str:
    for line in open(ENVFILE, encoding="utf-8"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"нет {name} в .env")


def read_key() -> str:
    return read_env("BITGN_API_KEY")


# Облачные провайдеры: OpenAI-совместимый адрес и имя ключа в .env.
# В облаке нет Ollama, прокси и видеокарты - их проверки пропускаются.
PROVIDERS = {
    "eden": ("https://api.edenai.run/v3", "EDENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # частный сервер: адрес и имя модели в коде не храним, они в QWEN_BASE_URL и QWEN_MODEL
    "qwen": ("", "QWEN_API_KEY"),
}


def setting(name: str) -> str:
    """Значение из переменной окружения, иначе из .env."""
    return os.environ.get(name) or read_env(name)


def provider(name: str, model: str) -> tuple[str, str, str]:
    """(адрес, имя ключа, имя модели для API) провайдера.

    Пустой адрес берётся из <ИМЯ>_BASE_URL. Модель, названная как сам провайдер (например, "qwen"),
    - псевдоним: настоящее имя на сервере берётся из <ИМЯ>_MODEL и не попадает в логи, папки
    и имя прогона на BitGN."""
    base, key_name = PROVIDERS[name]
    api_model = setting(f"{name.upper()}_MODEL") if model == name else model
    return base or setting(f"{name.upper()}_BASE_URL"), key_name, api_model


def hide(text, base: str) -> str:
    """Текст для лога без адреса провайдера: ни URL, ни хоста, ни порта (в т.ч. в текстах ошибок)."""
    out, u = str(text), urllib.parse.urlsplit(base)
    for s in filter(None, (base.rstrip("/"), u.netloc, u.hostname)):
        out = out.replace(s, "<адрес>")
    if u.port:
        out = re.sub(rf"(?:(?<=port=)|(?<=:)){u.port}\b", "<порт>", out)
    return out


def probe_cloud(base: str, key: str, model: str) -> dict:
    """Короткий вызов облачной модели: жива ли и сколько отвечает."""
    t = time.time()
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps({
        "model": model, "max_tokens": 20,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
    }).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return {"latency_s": round(time.time() - t, 1), "cost": d.get("cost"),
            "reply": ((d.get("choices") or [{}])[0].get("message") or {}).get("content")}


LLAMA_BASE = "http://127.0.0.1:8090"
START_LLAMA = os.path.join(ROOT, "runner", "start_llama.py")


def probe_llama() -> dict:
    """Скорость своего llama-server на том же тексте, что и probe() для Ollama."""
    # через чат: сырой /completion без шаблона модель обрывает через 2 токена
    r = http_json(LLAMA_BASE + "/v1/chat/completions", {
        "messages": [{"role": "user", "content": PROBE_TEXT + "\nSummarize the ledger rules in five sentences."}],
        "max_tokens": 200, "temperature": 0, "cache_prompt": False}, timeout=1800)
    t = r.get("timings", {})
    g = gpu_state() or {}
    return {"prompt_tokens": t.get("prompt_n"), "prompt_tps": round(t.get("prompt_per_second") or 0, 1),
            "gen_tokens": t.get("predicted_n"), "gen_tps": round(t.get("predicted_per_second") or 0, 1),
            "vram_mb": g.get("mem_mb")}


def run_llama(plan, model, outdir, rec) -> dict:
    """Прогон на своём llama-server (start_llama.py): конфигурация - plan["llama_preset"].

    Ollama и прокси не нужны, предпроверка видеопамяти тоже: её занимает сам сервер.
    Перезапуск при зависании - тот же start_llama.py с той же конфигурацией."""
    preset = plan.get("llama_preset", "moe20")
    restart = f'"{sys.executable}" "{START_LLAMA}" {preset}'
    # запас видеопамяти ~0.4 ГБ: любой посторонний процесс на GPU (05.10 - игра, 0.8 ГБ)
    # не даёт загрузиться черновику ("error loading model: vector")
    subprocess.run(f'"{sys.executable}" "{START_LLAMA}" stop', shell=True, capture_output=True, timeout=120)
    deadline = time.time() + plan.get("preflight_wait_s", 900)
    while (gpu_state() or {}).get("mem_mb", 0) > 300 and time.time() < deadline:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().replace("\n", "; ")
        say(f"{model}: видеопамять занята ({gpu_state()}), ждём: {apps[:200]}")
        time.sleep(60)
    out = subprocess.run(restart, shell=True, capture_output=True, text=True, timeout=700).stdout
    rec["preflight"] = [] if "llm ok" in out else ["llama-server start failed"]
    if rec["preflight"]:
        rec["status"] = "probe_failed"
        say(f"{model}: llama-server не поднялся: {out[-300:]}")
        return rec
    try:
        rec["probe"] = probe_llama()
        say(f"{model}: llama-server [{preset}], замер {rec['probe']}")
    except Exception as e:
        rec["probe"] = {"error": str(e)[:300]}
        rec["status"] = "probe_failed"
        say(f"{model}: замер не удался: {e}")
        return rec
    budget = plan.get("model_budget_min", 150) * 60
    env = agent_env(plan, model, outdir, budget, plan.get("ctx", 32768), LLAMA_BASE + "/v1", "llama")
    env["LLM_RESTART_CMD"] = restart
    started = time.time()
    with open(os.path.join(outdir, "console.log"), "w", encoding="utf-8") as con:
        p = subprocess.Popen([AGENT_PY, plan.get("entry", "main.py"), *plan["tasks"]], cwd=AGENT_DIR, env=env,
                             stdout=con, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=budget + 45 * 60)
            rec["status"] = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            p.kill()
            rec["status"] = "killed: hard timeout"
            say(f"{model}: принудительно остановлен по жёсткому пределу")
    rec["minutes"] = round((time.time() - started) / 60, 1)
    subprocess.run(f'"{sys.executable}" "{START_LLAMA}" stop', shell=True, capture_output=True, timeout=120)
    return rec


def run_model(plan, model, outdir) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rec = {"model": model, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    cloud = plan.get("provider")
    if cloud == "llama":
        return run_llama(plan, model, outdir, rec)
    if cloud:
        return run_cloud(plan, model, outdir, rec, *provider(cloud, model))

    problems = preflight(plan.get("preflight_wait_s", 900))
    rec["preflight"] = problems
    if problems:
        say(f"{model}: предпроверка не идеальна, продолжаю: {problems}")

    for attempt in (1, 2):
        try:
            rec["probe"] = probe(model)
            say(f"{model}: замер {rec['probe']}")
            break
        except Exception as e:
            rec["probe"] = {"error": str(e)}
            say(f"{model}: замер не удался (попытка {attempt}): {e}")
            if attempt == 2:
                rec["status"] = "probe_failed"
                return rec
            # вероятно, зависла Ollama: перезапуск и вторая попытка
            subprocess.run(RESTART + " --force", shell=True, capture_output=True, timeout=240)
    gen = rec["probe"].get("gen_tps") or 0
    if gen < plan.get("min_gen_tps", 3):
        rec["status"] = f"skipped: {gen} tok/s < {plan.get('min_gen_tps', 3)}"
        say(f"{model}: пропущена, слишком медленно ({gen} tok/s)")
        unload_all()
        return rec

    budget = plan.get("model_budget_min", 150) * 60
    env = dict(os.environ,
               BITGN_API_KEY=read_key(),
               OPENAI_BASE_URL=PROXY + "/v1", OPENAI_API_KEY="ollama",
               MODEL_ID=model, BENCH_ID=plan.get("bench", "bitgn/ecom1-dev"),
               HARNESS_EDITS=plan.get("edits", ""),
               STEP_MAX_TOKENS=str(plan.get("step_max_tokens", 1024)),
               LLM_TIMEOUT=str(plan.get("llm_timeout_s", 600)),
               TASK_TIME_LIMIT=str(plan.get("task_time_limit_s", 900)),
               STEP_LOG=os.path.join(outdir, "steps.jsonl"),
               RESULTS_PATH=os.path.join(outdir, "results.json"),
               RUN_DEADLINE=str(time.time() + budget),
               RUN_NAME=f"{plan['label']} {model} edits[{plan.get('edits', '') or '1-3'}]",
               LLM_RESTART_CMD=RESTART,
               # реальное окно модели по замеру: по нему оболочка решает, когда скрывать старые выводы
               CTX_WINDOW=str((rec.get("probe") or {}).get("ctx") or plan.get("ctx", 32768)),
               PYTHONIOENCODING="utf-8",
               # без этого вывод буферизуется и при остановке прогона console.log остаётся пустым
               PYTHONUNBUFFERED="1")
    if plan.get("temperature") is not None:
        env["TEMPERATURE"] = str(plan["temperature"])
    started = time.time()
    with open(os.path.join(outdir, "console.log"), "w", encoding="utf-8") as con:
        p = subprocess.Popen([AGENT_PY, plan.get("entry", "main.py"), *plan["tasks"]], cwd=AGENT_DIR, env=env,
                             stdout=con, stderr=subprocess.STDOUT)
        try:
            # дедлайн модели проверяется перед каждой задачей; жёсткий предел — на случай зависания
            p.wait(timeout=budget + 45 * 60)
            rec["status"] = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            p.kill()
            rec["status"] = "killed: hard timeout"
            say(f"{model}: принудительно остановлен по жёсткому пределу")
    rec["minutes"] = round((time.time() - started) / 60, 1)
    unload_all()
    return rec


def agent_env(plan, model, outdir, budget, ctx, base, key, api_model=None) -> dict:
    env = dict(os.environ,
               BITGN_API_KEY=read_key(),
               OPENAI_BASE_URL=base, OPENAI_API_KEY=key,
               MODEL_ID=api_model or model, BENCH_ID=plan.get("bench", "bitgn/ecom1-dev"),
               HARNESS_EDITS=plan.get("edits", ""),
               STEP_MAX_TOKENS=str(plan.get("step_max_tokens", 1024)),
               LLM_TIMEOUT=str(plan.get("llm_timeout_s", 600)),
               LLM_MAX_RETRIES=str(plan.get("llm_max_retries", 0)),
               PARALLEL=str(plan.get("parallel", 1)),
               RULES_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), plan["rules_file"]) if plan.get("rules_file") else "",
               GUIDANCE_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), plan["guidance_file"]) if plan.get("guidance_file") else "",
               DEMOS_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), plan["demos_file"]) if plan.get("demos_file") else "",
               TASK_TIME_LIMIT=str(plan.get("task_time_limit_s", 900)),
               STEP_LOG=os.path.join(outdir, "steps.jsonl"),
               RESULTS_PATH=os.path.join(outdir, "results.json"),
               RUN_DEADLINE=str(time.time() + budget),
               RUN_NAME=f"{plan['label']} {model} edits[{plan.get('edits', '') or '1-3'}]",
               CTX_WINDOW=str(ctx),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if plan.get("temperature") is not None:
        env["TEMPERATURE"] = str(plan["temperature"])
    return env


def run_cloud(plan, model, outdir, rec, base, key_name, api_model) -> dict:
    """Прогон на облачной модели: без Ollama, прокси, видеокарты и охлаждения."""
    key = read_env(key_name)
    rec["preflight"] = []
    try:
        rec["probe"] = probe_cloud(base, key, api_model)
        say(f"{model}: провайдер {plan['provider']}, замер {hide(rec['probe'], base)}")
    except Exception as e:
        rec["probe"] = {"error": hide(e, base)[:300]}
        rec["status"] = "probe_failed"
        say(f"{model}: облачная модель не отвечает: {hide(e, base)}")
        return rec
    budget = plan.get("model_budget_min", 150) * 60
    env = agent_env(plan, model, outdir, budget, plan.get("ctx", 131072), base, key, api_model)
    env.pop("LLM_RESTART_CMD", None)   # перезапускать в облаке нечего
    started = time.time()
    with open(os.path.join(outdir, "console.log"), "w", encoding="utf-8") as con:
        p = subprocess.Popen([AGENT_PY, plan.get("entry", "main.py"), *plan["tasks"]], cwd=AGENT_DIR, env=env,
                             stdout=con, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=budget + 45 * 60)
            rec["status"] = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            p.kill()
            rec["status"] = "killed: hard timeout"
    rec["minutes"] = round((time.time() - started) / 60, 1)
    return rec


def results_from_console(path):
    """Баллы из вывода исходного сэмпла: он не пишет results.json, а печатает
    "FINAL SCORE: x" и строки "- t01: Score: 1.00" по каждой завершённой задаче."""
    if not os.path.exists(path):
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, encoding="utf-8", errors="replace").read())
    if "FINAL SCORE:" not in text:
        return None
    trials = [{"task": t, "score": float(s), "done": True}
              for t, s in re.findall(r"^- (t\d+): Score: ([0-9.]+)", text.split("FINAL SCORE:")[-1], re.M)]
    score = sum(t["score"] for t in trials) / len(trials) if trials else 0.0
    return {"score": score, "trials": trials, "source": "console"}


def summarize(rec, outdir, ctx_limit) -> dict:
    steps, ends, errors, blocked, autocorr, skipped = [], [], [], 0, 0, 0
    path = os.path.join(outdir, "steps.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            t = r.get("type")
            if t == "step": steps.append(r)
            elif t == "task_end": ends.append(r)
            elif t == "error": errors.append(r)
            elif t == "blocked": blocked += 1
            elif t == "autocorrect": autocorr += 1
            elif t == "skipped": skipped += 1
    res_path = os.path.join(outdir, "results.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else None
    if results is None:
        results = results_from_console(os.path.join(outdir, "console.log"))
    done = [t for t in (results or {}).get("trials", []) if t["done"]]
    pt = [s["prompt_tokens"] for s in steps if s.get("prompt_tokens")]
    ctx = (rec.get("probe") or {}).get("ctx") or ctx_limit
    s = {
        "model": rec["model"], "status": rec.get("status"), "minutes": rec.get("minutes"),
        "score": results["score"] if results else None,
        "solved": sum(1 for t in done if t["score"] >= 1), "attempted": len(done),
        "per_task": {t["task"]: t["score"] for t in done},
        "end_answered": sum(1 for e in ends if e["end"] == "answered"),
        "end_loop": sum(1 for e in ends if e["end"] == "loop"),
        "end_budget": sum(1 for e in ends if e["end"] == "budget"),
        "end_time": sum(1 for e in ends if e["end"] == "time"),
        "end_runaway": sum(1 for e in ends if e["end"] == "runaway"),
        "err_model": sum(1 for e in errors if e.get("kind") == "model"),
        "err_infra": sum(1 for e in errors if e.get("kind") == "infra"),
        "skipped_deadline": skipped,
        "steps": len(steps),
        "cost_usd": round(sum((x.get("cost") or 0) for x in steps), 4),
        "avg_step_s": round(statistics.mean(x["ms"] for x in steps) / 1000, 1) if steps else None,
        "max_prompt_tokens": max(pt) if pt else None,
        "near_ctx_steps": sum(1 for x in pt if ctx and x >= 0.9 * ctx),
        "cut_by_length": sum(1 for x in steps if x.get("finish") == "length"),
        "blocked": blocked, "autocorrected": autocorr,
        "gen_tps": (rec.get("probe") or {}).get("gen_tps"),
        "vram_share": (rec.get("probe") or {}).get("vram_share"),
        "ctx": ctx, "preflight": rec.get("preflight"),
    }
    return s


def write_summary(series_dir, plan, rows):
    json.dump({"plan": plan, "rows": rows}, open(os.path.join(series_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    L = [f"# {plan['label']}", "",
         f"Оболочка: `{plan.get('git')}`, правки: 1-3{(' + ' + plan['edits']) if plan.get('edits') else ''}, "
         f"задач: {len(plan['tasks']) or 'все'}, обновлено {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
         "| Модель | Балл | Решено | $ | Ответил сам | Петля | Шаги кончились | Время кончилось | Беск. генерация (задач / шагов) | Ошибки модели / инфраструктуры | Шагов | Сек/шаг | Макс. промпт / окно | У потолка | tok/s | В VRAM | Минут | Статус |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['score'] if r['score'] is not None else '-'} | {r['solved']}/{r['attempted']} | {r.get('cost_usd', 0)} | "
                 f"{r['end_answered']} | {r['end_loop']} | {r['end_budget']} | {r['end_time']} | "
                 f"{r.get('end_runaway', 0)} / {r['cut_by_length']} | "
                 f"{r['err_model']} / {r['err_infra']} | {r['steps']} | {r['avg_step_s']} | "
                 f"{r['max_prompt_tokens']} / {r['ctx']} | {r['near_ctx_steps']} | {r['gen_tps']} | "
                 f"{r['vram_share']} | {r['minutes']} | {r['status']} |")
    L += ["", "## Баллы по задачам", ""]
    tasks = sorted({t for r in rows for t in r["per_task"]})
    if tasks:
        L.append("| Модель | " + " | ".join(tasks) + " |")
        L.append("|---|" + "---|" * len(tasks))
        for r in rows:
            L.append(f"| {r['model']} | " + " | ".join(
                ("1" if r["per_task"].get(t, 0) >= 1 else "0") if t in r["per_task"] else "-" for t in tasks) + " |")
    L += ["", "«У потолка» — шагов, где промпт занял 90%+ окна: там возможно молчаливое обрезание.",
          "Ошибки инфраструктуры — падения Ollama/прокси, модель в них не виновата.",
          "Бесконечная генерация — ответы, оборванные лимитом токенов на шаг; задача считается так, если оборвано 3 подряд."]
    open(os.path.join(series_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    global LOG, AGENT_DIR
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    # папка агента из плана; относительный путь считается от корня проекта
    AGENT_DIR = os.path.join(ROOT, plan.get("agent_dir", AGENT_DIR))
    plan["agent_dir"] = AGENT_DIR
    plan["git"] = git_version()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    series_dir = os.path.join(RUNS_DIR, f"{stamp}_{plan['label']}")
    os.makedirs(series_dir, exist_ok=True)
    LOG = os.path.join(series_dir, "dispatcher.log")
    json.dump(plan, open(os.path.join(series_dir, "plan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    say(f"серия {plan['label']}: {len(plan['models'])} моделей, оболочка {plan['git']}, папка {series_dir}")
    if plan["git"].endswith("DIRTY"):
        say("ВНИМАНИЕ: в оболочке есть незакоммиченные изменения — версия не воспроизводима")
    rows = []
    try:
        for model in plan["models"]:
            outdir = os.path.join(series_dir, model.replace(":", "_").replace("/", "__").replace("@", "_"))
            say(f"=== {model} ===")
            try:
                rec = run_model(plan, model, outdir)
            except Exception as e:
                rec = {"model": model, "status": f"dispatcher error: {e}"}
                say(f"{model}: ошибка диспетчера: {e}")
            row = summarize(rec, outdir, plan.get("ctx", 32768))
            rows.append(row)
            write_summary(series_dir, plan, rows)   # после каждой модели — чтобы итог был даже при обрыве
            say(f"{model}: балл {row['score']}, решено {row['solved']}/{row['attempted']}, {row['minutes']} мин, {row['status']}")
            cool_until = time.time() + (600 if plan.get("provider") in (None, "llama") else 0)
            while time.time() < cool_until:
                g = gpu_state()
                if g and g["temp"] <= 65: break
                time.sleep(20)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        say("серия завершена")


if __name__ == "__main__":
    main()
