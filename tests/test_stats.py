"""Инструменты оценки: балл прогона, сломанные прогоны, статистические тесты.

От них зависят все выводы проекта, поэтому проверяем на данных с заранее известным ответом:
одинаковые варианты - разница 0 и p около 1, явная разница - маленькое p.
"""
import json
import random

import pytest

import stats
import task_types


def trial(task, score, detail=()):
    return {"task": task, "done": True, "score": score, "detail": list(detail)}


def make_run(root, name, trials):
    """Папка прогона как у диспетчера: <root>/<name>/<модель>/results.json."""
    d = root / name / "qwen"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"run_id": name, "score": 0, "trials": trials}), encoding="utf-8")


# --- балл и сломанные прогоны ------------------------------------------------------

def test_run_score_is_mean_over_all_tasks():
    # регрессия: раньше сумма всегда делилась на 55 (размер dev), на prod из 100 задач балл выходил больше 1
    tasks = {f"t{i:03d}": 1.0 if i <= 76 else 0.0 for i in range(1, 101)}
    assert stats.run_score(tasks) == pytest.approx(0.76)


@pytest.mark.parametrize("no_answer, expected", [
    (0, False), (2, False), (5, False), (6, True), (10, True),
], ids=["none", "two", "exactly-half", "majority", "all"])
def test_is_broken_when_most_tasks_have_no_answer(no_answer, expected):
    trials = [trial(f"t{i:02d}", 0.0, ["no answer provided"] if i < no_answer else []) for i in range(10)]
    assert task_types.is_broken(trials) is expected


def test_latest_run_skips_broken_and_unfinished(tmp_path, monkeypatch):
    monkeypatch.setattr(task_types, "RUNS", str(tmp_path))
    make_run(tmp_path, "20261012_2202_qwen_q_dev", [trial("t01", 1.0)])
    make_run(tmp_path, "20261012_2300_qwen_q_dev", [trial("t01", 0.0, ["no answer provided"])])  # сломан
    (tmp_path / "20261012_2359_qwen_q_dev").mkdir()                                              # без результатов
    assert task_types.latest_run("qwen_q_dev").endswith("20261012_2202_qwen_q_dev")


def test_load_group_counts_unfinished_tasks_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "RUNS", str(tmp_path))
    make_run(tmp_path, "r1_x", [trial("t01", 1.0), {"task": "t02", "done": False, "score": 0.0, "detail": []}])
    (name, tasks), = stats.load_group("*_x")
    assert stats.run_score(tasks) == pytest.approx(0.5)   # как у BitGN: 1 из 2, а не 1 из 1


# --- статистические тесты ---------------------------------------------------------

def test_paired_stats_no_difference():
    m, lo, hi, p = task_types.paired_stats([0.0] * 50, random.Random(1), n_boot=2000)
    assert (m, lo, hi) == (0.0, 0.0, 0.0)
    assert p == 1.0


def test_paired_stats_clear_improvement():
    m, lo, hi, p = task_types.paired_stats([1.0] * 30, random.Random(1), n_boot=2000)
    assert m == lo == hi == 1.0
    assert p < 0.001   # все 30 задач лучше: случайно так почти не бывает


def test_paired_stats_p_matches_exact_two_sided_value():
    # 5 задач, все лучше на 1: из 2^5 вариантов знаков столь же крайних два (все + и все -),
    # точное двустороннее p = 2/32 = 0.0625; одностороннее было бы вдвое меньше
    _, _, _, p = task_types.paired_stats([1.0] * 5, random.Random(1), n_boot=20000)
    assert p == pytest.approx(2 / 2 ** 5, abs=0.01)


def test_paired_stats_interval_contains_mean():
    rnd = random.Random(7)
    diffs = [rnd.choice([-1.0, 0.0, 0.0, 1.0, 1.0]) for _ in range(100)]
    m, lo, hi, p = task_types.paired_stats(diffs, random.Random(1), n_boot=2000)
    assert lo < m < hi
    assert 0.0 <= p <= 1.0


def test_paired_stats_reproducible_with_seed():
    diffs = [1.0, -1.0, 0.5, 0.0, 1.0, 0.0, -0.5, 1.0]
    first = task_types.paired_stats(diffs, random.Random(1), n_boot=2000)
    assert task_types.paired_stats(diffs, random.Random(1), n_boot=2000) == first


def test_sign_flip_test_bounds():
    assert stats.sign_flip_test([0.0] * 20, n=2000) == 1.0
    assert stats.sign_flip_test([1.0] * 30, n=2000) < 0.001


@pytest.mark.parametrize("b, c, expected", [
    (0, 0, 1.0),          # расхождений нет
    (3, 3, 1.0),          # поровну - никакой разницы
    (0, 5, 2 / 2 ** 5),   # 5 из 5 в пользу B: 2 * (1/2)^5
    (1, 9, 2 * (1 + 10) / 2 ** 10),
], ids=["no-discordant", "balanced", "five-to-zero", "nine-to-one"])
def test_mcnemar_exact_known_values(b, c, expected):
    assert stats.mcnemar_exact(b, c) == pytest.approx(expected)


def test_bootstrap_ci_of_constant_is_point():
    assert stats.bootstrap_ci([0.7] * 20, n=1000) == (pytest.approx(0.7), pytest.approx(0.7))
