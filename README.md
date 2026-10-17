# BitGN ECOM agent

LLM-агент для испытания [BitGN ECOM](https://bitgn.com/challenge/ecom) и инструменты для его оценки:
серийные прогоны, сравнение вариантов оболочки и статистическая проверка улучшений.

Агент основан на примере `ecom-py` из [bitgn/sample-agents](https://github.com/bitgn/sample-agents)
(автор - Rinat Abdullin). В исходном репозитории нет файла лицензии; перед публикацией этого
проекта нужно получить разрешение BitGN.

> Черновик: разделы о результатах и методике будут дополнены.

## Структура

| Папка | Что внутри |
|---|---|
| `agent/` | Агент: цикл модели, инструменты предметной области (`locate`, `stock_check`, `catalog_find`), проверки ответа перед отправкой |
| `runner/` | `dispatcher.py` - прогон серии по плану; `run_seq.py` - несколько планов подряд; `task_types.py`, `stats.py`, `failures.py` - анализ результатов |
| `plans/` | Планы прогонов (JSON): провайдер, модель, задачи, бюджеты |
| `hooks/` | `pre-commit` - проверка каждого коммита на секреты (gitleaks) |

## Запуск

Нужны Python, [uv](https://docs.astral.sh/uv/) и ключ BitGN.

1. Окружение агента: `cd agent && uv sync`
2. Настройки: скопировать `.env.example` в `.env` и заполнить
3. Прогон: `python runner/dispatcher.py plans/qwen_e_dev.json`

Результаты пишутся в `runs/` (в git не попадают): лог диспетчера, ответы и шаги агента, сводка.

## Безопасность

- Ключи, адреса серверов и имена моделей на частных серверах - только в `.env`.
- После клонирования включить проверку коммитов:
  `git config core.hooksPath hooks` (нужен [gitleaks](https://github.com/gitleaks/gitleaks)).
