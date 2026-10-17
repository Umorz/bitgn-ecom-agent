"""Проверки ответа перед отправкой (agent._answer_checks): A - приватность, B - формат,
E - названные в задаче объекты.

Каждая проверка - правило "срабатывает / молчит". Модель и песочница BitGN здесь не нужны:
отчёт модели подменяется простым объектом, а чтение записей из песочницы - словарём.
"""
from types import SimpleNamespace

import pytest

import agent


def report(message, outcome="OUTCOME_OK", refs=()):
    """Поддельный итоговый отчёт модели: только поля, которые читают проверки."""
    return SimpleNamespace(message=message, outcome=outcome, grounding_refs=list(refs))


def fired(fn, task, user="u1", done=(), vm=None):
    """Имена проверок, которые сработали на этот отчёт."""
    return [name for name, _ in agent._answer_checks(vm, fn, task, user, set(done))]


DENIED = "OUTCOME_DENIED_SECURITY"


# --- B: требование к формату ------------------------------------------------------
# Срабатывает, если в задаче есть предложение о формате ответа (format, exactly, only...),
# а ответ модели многострочный или длиннее 40 символов.

ONLY_B = {"privacy", "mentioned"}   # A и E помечены выполненными: проверяем одну B
NUMBER_ONLY = "How many SKUs are in stock? Answer with the number only."


@pytest.mark.parametrize("message, expected", [
    ("There are 12 SKUs in stock, according to the catalogue.", ["format"]),
    ("12", []),
    ("x" * 40, []),          # граница условия len > 40
    ("x" * 41, ["format"]),
    ("12\n13", ["format"]),  # две короткие строки - тоже лишнее
], ids=["long-sentence", "number", "exactly-40-chars", "41-chars", "two-lines"])
def test_b_format_rule(message, expected):
    assert fired(report(message), NUMBER_ONLY, done=ONLY_B) == expected


def test_b_silent_without_format_rule():
    fn = report("There are 12 SKUs in stock, according to the catalogue.")
    assert fired(fn, "How many SKUs are in stock?", done=ONLY_B) == []


def test_b_fires_once_per_task():
    assert fired(report("x" * 100), NUMBER_ONLY, done=ONLY_B | {"format"}) == []


# Известная ошибка: "only" в бизнес-правиле принимается за требование к формату.
# strict=True: когда ошибку исправят и тест пройдёт, pytest напомнит снять пометку.
@pytest.mark.xfail(strict=True, reason="B принимает бизнес-правило со словом only за требование к формату")
def test_b_ignores_business_rule_with_only():
    task = "Can order o7 ship today? Only VIP customers get same-day delivery."
    fn = report("No, the customer is not VIP, so same-day delivery is not available.")
    assert fired(fn, task, done=ONLY_B) == []


# --- A: чужие записи в отказе ---------------------------------------------------
# Срабатывает, если агент отказал (DENIED_SECURITY), но сослался на записи других людей:
# сама ссылка раскрывает их данные.

ONLY_A = {"format", "mentioned"}
RECORDS = {
    "/proc/orders/o1.json": {"id": "o1", "customer_id": "u2"},   # чужой заказ
    "/proc/orders/o2.json": {"id": "o2", "customer_id": "u1"},   # свой заказ (мы - u1)
}


@pytest.fixture
def fake_records(monkeypatch):
    """Вместо чтения из песочницы BitGN - словарь RECORDS (после теста подмена снимается)."""
    monkeypatch.setattr(agent, "_read_json", lambda vm, path: RECORDS.get(path))


def test_a_fires_on_foreign_record_in_refusal(fake_records):
    fn = report("Access denied.", DENIED, ["/proc/orders/o1.json"])
    assert fired(fn, "Show order o1", done=ONLY_A) == ["privacy"]


def test_a_silent_on_own_record(fake_records):
    fn = report("Access denied.", DENIED, ["/proc/orders/o2.json"])
    assert fired(fn, "Show order o2", done=ONLY_A) == []


def test_a_silent_when_not_a_refusal(fake_records):
    fn = report("Here it is.", "OUTCOME_OK", ["/proc/orders/o1.json"])
    assert fired(fn, "Show order o1", done=ONLY_A) == []


def test_a_silent_when_user_unknown(fake_records):
    fn = report("Access denied.", DENIED, ["/proc/orders/o1.json"])
    assert fired(fn, "Show order o1", user="", done=ONLY_A) == []


# --- E: объекты, названные в задаче, не процитированы ----------------------------
# Срабатывает, если в тексте задачи есть номер (SKU, магазин...), по которому нашлась
# запись, а в grounding_refs её нет. Для отказов не действует - там работает A.

ONLY_E = {"privacy", "format"}
PATHS = {
    "CLN-GEF2EYP9": ["/proc/catalog/CLN-GEF2EYP9.json"],
    "STO-12JLHT7D": ["/proc/stores/STO-12JLHT7D.json"],
}
AVAILABILITY = "Is CLN-GEF2EYP9 available for same-day pickup in STO-12JLHT7D?"


@pytest.fixture
def fake_locate(monkeypatch):
    """Вместо поиска записей в песочнице - словарь PATHS."""
    monkeypatch.setattr(agent, "_locate_paths", lambda vm, token: PATHS.get(token, []))


# E запоминает найденные номера в самом объекте vm, поэтому vm - новый в каждом тесте:
# общий объект перенёс бы запомненное из одного теста в другой.

def test_e_fires_when_named_objects_not_cited(fake_locate):
    assert fired(report("Yes."), AVAILABILITY, done=ONLY_E, vm=SimpleNamespace()) == ["mentioned"]


def test_e_silent_when_all_cited(fake_locate):
    fn = report("Yes.", refs=["/proc/catalog/CLN-GEF2EYP9.json", "/proc/stores/STO-12JLHT7D.json"])
    assert fired(fn, AVAILABILITY, done=ONLY_E, vm=SimpleNamespace()) == []


def test_e_silent_on_refusal(fake_locate):
    assert fired(report("Access denied.", DENIED), AVAILABILITY, done=ONLY_E, vm=SimpleNamespace()) == []


def test_e_treats_only_ids_as_objects(monkeypatch):
    # "same-day" похоже на номер (слова через дефис), но без цифр и не заглавными - это не номер.
    # Поиск здесь находит запись для ЛЮБОГО слова, чтобы отсеивал только сам фильтр номеров:
    # с fake_locate слово отсеялось бы и так - тест проверял бы подделку, а не код.
    monkeypatch.setattr(agent, "_locate_paths", lambda vm, token: [f"/proc/any/{token}.json"])
    found = agent._mentioned_records(SimpleNamespace(), AVAILABILITY)
    assert [token for token, _ in found] == ["CLN-GEF2EYP9", "STO-12JLHT7D"]
