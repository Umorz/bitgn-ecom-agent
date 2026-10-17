"""Инструмент catalog_find: поиск товаров по словам, цене и свойствам.

Каталог подменяется записями той же формы, что в песочнице BitGN (sku, name, brand, kind_id,
family_id, category_id, price_cents, properties). Через _catalog_from_files проходит и
настоящее превращение записи в строку поиска - тест проверяет его, а не подделку.
"""
import pytest

import agent

RAW = {
    "/proc/catalog/Makita/PT-BLA-MAK-185.json": {
        "id": 1, "sku": "PT-BLA-MAK-185", "name": "Makita Specialized Thin Metal cutting blade set",
        "brand": "Makita", "category_id": "cat-accessories", "kind_id": "kind-saw-blades",
        "family_id": "fam-makita-specialized", "price_cents": 2490, "properties": {"diameter_mm": 185, "pieces": 3},
    },
    "/proc/catalog/Alpen/PT-BIT-ALP-HSS-19.json": {
        "id": 2, "sku": "PT-BIT-ALP-HSS-19", "name": "Alpen HSS Sprint drill bit set 19 pcs",
        "brand": "Alpen", "category_id": "cat-accessories", "kind_id": "kind-drill-bits",
        "family_id": "fam-alpen-hss-sprint", "price_cents": 1990, "properties": {"pieces": 19, "material": "cobalt"},
    },
    "/proc/catalog/PowerTools Plans/PT-DIG-GUIDE-CUT.json": {
        "id": 3, "sku": "PT-DIG-GUIDE-CUT", "name": "PowerTools cutting and drilling selection guide",
        "brand": "PowerTools Plans", "category_id": "cat-digital-ebooks", "kind_id": "kind-guides",
        "family_id": "fam-powertools-guides", "price_cents": 990, "properties": {"delivery_mode": "download"},
    },
}


@pytest.fixture
def catalog(monkeypatch):
    """Каталог из RAW: чтение файлов подменено, разбор записей - настоящий."""
    monkeypatch.setattr(agent, "_read_many", lambda vm, paths: [RAW[p] for p in paths])
    monkeypatch.setattr(agent, "_catalog_records", lambda vm: agent._catalog_from_files(vm, list(RAW)))


def find(text, max_p=0, min_p=0, props=()):
    return agent._tool_catalog_find(None, text, max_p, min_p, list(props))


def count(out):
    """Число найденных товаров из первой строки ответа: "N product(s) match ..."."""
    return int(out.split(" ", 1)[0])


def test_finds_by_words_in_name(catalog):
    out = find("makita thin metal")
    assert count(out) == 1
    assert "PT-BLA-MAK-185" in out


def test_words_are_case_insensitive(catalog):
    # "Thin" есть только в названии и только с заглавной: запрос строчными должен его найти
    assert count(find("thin")) == 1


def test_all_words_must_match(catalog):
    out = find("makita alpen")
    assert count(out) == 0
    assert "Nothing matched every word" in out   # подсказка модели попробовать меньше слов


@pytest.mark.parametrize("max_p, min_p, expected", [
    (0, 0, 1),        # 0 - без ограничения
    (2000, 0, 1),     # цена 1990 в пределах
    (1000, 0, 0),     # дороже верхней границы
    (0, 2000, 0),     # дешевле нижней границы
], ids=["no-limit", "under-max", "over-max", "under-min"])
def test_price_bounds(catalog, max_p, min_p, expected):
    assert count(find("alpen", max_p, min_p)) == expected


def test_property_filter(catalog):
    assert count(find("alpen", props=["pieces=19"])) == 1
    assert count(find("alpen", props=["pieces=25"])) == 0


def test_searches_kind_and_family(catalog):
    # kind_id и family_id входят в строку поиска: слово из них находит товар
    assert count(find("drill-bits")) == 1


# --- Известные слабости (найдены при разборе провалов на prod, analysis-половина) ----------
# Все слова запроса ищутся только в sku, name, brand, kind_id, family_id. На prod из-за этого
# модель получала "0 product(s)" для товаров, которые были в каталоге, и отвечала "нет".

@pytest.mark.xfail(strict=True, reason="размер в запросе ('185 mm') не сопоставляется со свойством diameter_mm")
def test_finds_when_size_given_with_unit(catalog):
    assert count(find("185 mm Makita Specialized thin metal cutting blade set")) == 1


@pytest.mark.xfail(strict=True, reason="значения properties не участвуют в поиске по словам")
def test_finds_by_property_value_word(catalog):
    assert count(find("cobalt alpen")) == 1


@pytest.mark.xfail(strict=True, reason="category_id не участвует в поиске: 'ebook' не находит товар из cat-digital-ebooks")
def test_finds_by_category_word(catalog):
    assert count(find("ebook cutting drilling")) == 1
