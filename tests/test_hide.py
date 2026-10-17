from dispatcher import hide

BASE = "http://192.0.2.7:8123/v1"

ADDRESS_NO_PORT = "https://example.org/v1"


def test_hide_removes_address_without_port():
    text = "ошибка https://example.org/v1 недоступен"
    assert hide(text, ADDRESS_NO_PORT) == "ошибка <адрес> недоступен"
def test_hide_removes_full_address():
    assert hide("облако http://192.0.2.7:8123/v1, замер", BASE) == "облако <адрес>, замер"
def test_hide_removes_host_and_port_from_error():
    text = "HTTPConnectionPool(host='192.0.2.7', port=8123): Max retries exceeded"
    assert hide(text, BASE) == "HTTPConnectionPool(host='<адрес>', port=<порт>): Max retries exceeded"


def test_hide_keeps_text_without_address():
    assert hide("{'latency_s': 1.2}", BASE) == "{'latency_s': 1.2}"
