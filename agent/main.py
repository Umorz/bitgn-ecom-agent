import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import textwrap
import time
import traceback
import urllib.request

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
    TRIAL_STATE_DONE,
)
from connectrpc.errors import ConnectError

from agent import _trace, run_agent, with_retry


BITGN_URL = os.getenv("BITGN_HOST") or "https://api.bitgn.com"
BITGN_API_KEY = os.getenv("BITGN_API_KEY") or ""
BENCH_ID = os.getenv("BENCH_ID") or os.getenv("BENCHMARK_ID") or "bitgn/ecom1-dev"
MODEL_ID = os.getenv("MODEL_ID") or "gpt-4.1-2025-04-14"

# --- Запуск без присмотра ------------------------------------------------------
RUN_DEADLINE = float(os.getenv("RUN_DEADLINE") or 0)   # unix-время; после него задачи не начинаем
RESULTS_PATH = os.getenv("RESULTS_PATH")                # сюда пишем баллы по задачам (JSON)
LLM_RESTART_CMD = os.getenv("LLM_RESTART_CMD")          # чем поднять модель, если она упала
RUN_NAME = os.getenv("RUN_NAME") or "ECOM Python Sample"


LLM_PROBE_TIMEOUT = float(os.getenv("LLM_PROBE_TIMEOUT") or 180)   # с запасом на загрузку модели
PARALLEL = int(os.getenv("PARALLEL") or 1)   # сколько задач вести одновременно (для облака; Ollama - 1)


class _ThreadRouter:
    """sys.stdout, который в рабочих потоках пишет в буфер своего потока."""

    def __init__(self, base):
        self.base = base
        self.local = threading.local()
        self.lock = threading.Lock()

    def write(self, s):
        buf = getattr(self.local, "buf", None)
        if buf is None:
            with self.lock:
                return self.base.write(s)
        buf.append(s)
        return len(s)

    def flush(self):
        if getattr(self.local, "buf", None) is None:
            self.base.flush()

    def __getattr__(self, name):
        return getattr(self.base, name)


def llm_alive() -> bool:
    """Жив ли сервер модели - проверяем генерацией одного токена.

    Список моделей не годится: 29.09 зависшая Ollama отвечала на /models,
    а генерация стояла, и каждая задача теряла 10 минут на тайм-аут."""
    base = (os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    if not base:
        return True   # облачный провайдер - проверять нечего
    body = json.dumps({"model": MODEL_ID, "max_tokens": 5,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    headers = {"Content-Type": "application/json"}
    if os.getenv("OPENAI_API_KEY"):   # облачному провайдеру нужен ключ, Ollama - нет
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(base + "/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=LLM_PROBE_TIMEOUT) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_llm(known_bad: bool = False) -> bool:
    """Перед задачей: генерирует ли модель. Если нет - принудительный перезапуск."""
    if not known_bad and llm_alive():
        return True
    print(f"{CLI_RED}LLM does not generate, forcing restart{CLI_CLR}")
    _trace({"type": "infra", "event": "llm_hung"})
    if LLM_RESTART_CMD:
        try:
            subprocess.run(LLM_RESTART_CMD + " --force", shell=True, timeout=240)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(3):
            if llm_alive():
                _trace({"type": "infra", "event": "llm_restarted"})
                return True
            time.sleep(10)
    _trace({"type": "infra", "event": "llm_down"})
    return False

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"

def main() -> None:
    task_filter = os.sys.argv[1:]

    try:
        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        run = client.start_run(
            StartRunRequest(
                name=RUN_NAME,
                benchmark_id=BENCH_ID,
                api_key=BITGN_API_KEY,
            )
        )

        # номер прогона сохраняем сразу: если отправка не пройдёт, его можно отправить вручную
        if RESULTS_PATH:
            with open(RESULTS_PATH + ".run_id", "w", encoding="utf-8") as f:
                f.write(run.run_id)

        def process_trial(trial_id):
            try:
                t = with_retry(client.start_trial, StartTrialRequest(trial_id=trial_id), what="start_trial")
            except ConnectError as exc:
                # BitGN недоступен дольше, чем ждут повторы: теряем задачу, но не прогон
                print(f"{CLI_RED}start_trial failed: {exc.code}: {exc.message}{CLI_CLR}")
                _trace({"type": "error", "task": trial_id, "kind": "infra", "error": f"start_trial {exc.code}"})
                return
            if task_filter and t.task_id not in task_filter:
                return

            if RUN_DEADLINE and time.time() > RUN_DEADLINE:
                print(f"{CLI_RED}Run deadline reached, skipping {t.task_id}{CLI_CLR}")
                _trace({"type": "skipped", "task": t.task_id, "reason": "run_deadline"})
                with_retry(client.end_trial, EndTrialRequest(trial_id=t.trial_id), what="end_trial")
                return

            print(f"{'=' * 30} Starting task: {t.task_id} {'=' * 30}")
            print(f"{CLI_BLUE}{t.instruction}{CLI_CLR}\n{'-' * 80}")
            if not ensure_llm():
                print(f"{CLI_RED}LLM unavailable, task {t.task_id} not attempted{CLI_CLR}")
                _trace({"type": "error", "task": t.task_id, "kind": "infra", "error": "llm unavailable"})
                with_retry(client.end_trial, EndTrialRequest(trial_id=t.trial_id), what="end_trial")
                return
            try:
                run_agent(MODEL_ID, t.harness_url, t.instruction, task_id=t.task_id)
            except Exception as exc:
                # ошибку модели (обрыв генерации, невалидный JSON, тайм-аут)
                # отличаем от ошибки инфраструктуры, чтобы не винить модель зря
                name = type(exc).__name__
                print(f"{CLI_RED}{name}: {exc}{CLI_CLR}")
                # тайм-аут бывает и у медленной модели, и у зависшего сервера.
                # Различаем генерацией одного токена: сервер не отвечает -
                # виновата инфраструктура, перезапускаем до следующей задачи.
                infra = False
                if name in ("APITimeoutError", "APIConnectionError", "InternalServerError"):
                    if not llm_alive():
                        infra = True
                        ensure_llm(known_bad=True)
                _trace({"type": "error", "task": t.task_id, "kind": "infra" if infra else "model",
                        "error": f"{name}: {str(exc)[:300]}",
                        "where": traceback.format_exc(limit=3)[-600:]})
            # complete run to get scores
            try:
                with_retry(client.end_trial, EndTrialRequest(trial_id=t.trial_id), what="end_trial")
            except ConnectError as exc:
                print(f"{CLI_RED}end_trial failed: {exc.code}{CLI_CLR}")
                _trace({"type": "error", "task": t.task_id, "kind": "infra", "error": f"end_trial {exc.code}"})

        try:
            if PARALLEL <= 1:
                for trial_id in run.trial_ids:
                    process_trial(trial_id)
            else:
                # Задачи независимы: в облаке их можно вести одновременно.
                # Вывод каждой копится в буфере потока и печатается одним блоком,
                # чтобы логи задач не перемешивались.
                router = _ThreadRouter(sys.stdout)
                sys.stdout = router

                def buffered(trial_id):
                    router.local.buf = []
                    try:
                        process_trial(trial_id)
                    except Exception as exc:   # задача не должна ронять соседей
                        print(f"{CLI_RED}worker error: {type(exc).__name__}: {exc}{CLI_CLR}")
                    finally:
                        text = "".join(router.local.buf)
                        router.local.buf = None
                        with router.lock:
                            router.base.write(text)
                            router.base.flush()

                try:
                    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
                        list(pool.map(buffered, run.trial_ids))
                finally:
                    sys.stdout = router.base
        finally:
            print(f"\n{CLI_GREEN}>>>> Submitting run... <<<<{CLI_CLR}")
            result = with_retry(client.submit_run, SubmitRunRequest(run_id=run.run_id, force=True), what="submit_run")

            if result.score_available:
                print(f"FINAL SCORE: {result.score:0.2f}")
                incomplete = 0
                for t in result.trials:
                    if t.state != TRIAL_STATE_DONE:
                        incomplete+=1
                        continue # skip filtered tasks

                    style = CLI_GREEN if t.score == 1 else CLI_RED
                    explain = "\n" + textwrap.indent("\n".join(t.score_detail), "  ") + "\n"
                    print( f"- {t.task_id}: {style}Score: {t.score:0.2f}{CLI_CLR}{explain}".strip("\n "))

                if incomplete>0:
                    print(f"{CLI_RED}incomplete trials: {incomplete}{CLI_CLR}")

                if RESULTS_PATH:
                    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                        json.dump({
                            "run_id": run.run_id, "score": result.score,
                            "trials": [{"task": t.task_id, "done": t.state == TRIAL_STATE_DONE,
                                        "score": t.score, "detail": list(t.score_detail)}
                                       for t in result.trials],
                        }, f, ensure_ascii=False, indent=1)
            else:
                print(f"\n{CLI_RED}Score is not available. Results are sealed and will be revealed later{CLI_CLR}\n")


    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")


if __name__ == "__main__":
    main()
