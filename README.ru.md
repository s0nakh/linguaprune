<div align="center">

<img src="assets/banner.png" alt="LinguaPrune" width="720">

**Языково-ориентированный прунинг экспертов в MoE-моделях.**
Оставляем экспертов, которые реально нужны вашим языкам, остальных убираем, затем квантуем.
Первая цель — казахский 🇰🇿 + русский.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.x-FFD21E)](https://github.com/huggingface/transformers)
[![Runs on Kaggle](https://img.shields.io/badge/runs%20on-Kaggle%20T4%20x2-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Status](https://img.shields.io/badge/status-research%20preview-orange)](#дорожная-карта)

[English](README.md) · [Связанные работы](docs/prior-art.ru.md)

</div>

---

## Зачем

Суверенные и многоязычные MoE-модели огромные. Например, MoE на 247 млрд параметров весит ~494 ГБ в BF16, это около 7 серверных видеокарт. Но для казахского и русского модель редко зовёт многих своих экспертов. LinguaPrune измеряет, **какие эксперты реально нужны языку**, и убирает остальных, чтобы модели хватало меньшего числа видеокарт.

> Это **эмпирическое исследование и открытый инструмент**, а не новый метод.
> Прунинг экспертов (например, REAP) и языковой прунинг уже существуют, см. [связанные работы](docs/prior-art.ru.md).
> Наш вклад — казахский + русский: измерено, воспроизводимо и честно о потере качества.

## Результаты

**Ноутбук 01 — профилирование роутера** на [`allenai/OLMoE-1B-7B-0125`](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
(64 эксперта в слое, 8 активных; 100 тыс. токенов Википедии на язык):

| Язык | Экспертов на 90% выборов роутера (из 64) |
|---|---|
| Казахский (kk) | **22.5** |
| Русский (ru) | 35.9 |
| Китайский (zh) | 39.5 |
| Английский (en) | 47.8 |

- **kk + ru вместе нужно ~37.7 эксперта из 64 в слое**, то есть убрать можно ~41% (95% ДИ оставленных: 37.3–38.4).
- На **отложенных статьях** покрытие 96% (kk) и 91% (ru).
- ⚠️ Это частота выбора экспертов, **а не качество** модели. Качество меряет ноутбук 02.

<img src="experiments/01/ru/experts_needed_per_layer.png" alt="Сколько экспертов нужно по слоям" width="720">

- **Глубоким слоям нужно намного меньше экспертов.** В слое 14 казахскому хватает 12 из 64, а в первых слоях нужно ~40. Значит, резать стоит по-разному в разных слоях.
- **Воспроизводимо:** два независимых прогона на Kaggle дали одинаковые цифры и одинаковые наборы экспертов в каждом слое (версия модели `9b0c1aa`).

Сырые цифры: [`experiments/01/summary.json`](experiments/01/summary.json).

## Ноутбуки

| # | Ноутбук | Что делает | Открыть |
|---|---|---|---|
| 01 | [`01_expert_profiling`](notebooks/ru/01_expert_profiling.ipynb) | Каких экспертов использует каждый язык; сколько можно убрать | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/ru/01_expert_profiling.ipynb) |
| 02 | [`02_prune_and_measure`](notebooks/ru/02_prune_and_measure.ipynb) | Убрать экспертов и измерить качество на kk/ru/en в сравнении со случайными и «английскими» экспертами | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/ru/02_prune_and_measure.ipynb) |

Это русские версии ноутбуков; [английские](notebooks/) — основные, код в них одинаковый.

**Запуск на Kaggle:**
1. Нажми *Open in Kaggle* в таблице выше.
2. Выбери **GPU T4 x2**, **Internet On**.
3. Нажми **Save Version → Save & Run All**.

## Дорожная карта

- [x] 01 · Профилирование роутера на OLMoE
- [ ] 02 · Качество после прунинга (Belebele kk/ru/en)
- [ ] 03 · Тот же пайплайн на Qwen3-30B-A3B
- [ ] Критерий в духе REAP и бюджет по слоям
- [ ] Прунинг + 4 бита против равномерных 2–3 бит **при одинаковой памяти**
- [ ] Большие MoE-модели для казахского/русского
- [ ] CLI `pip install linguaprune` для любого языка

**Проверка на опровержение.** При одинаковой памяти «прунинг + 4 бита» должен обойти равномерную квантизацию и меньшую dense-модель по качеству на kk/ru. Если не обойдёт, мы опубликуем и этот результат.

## Лицензия

[Apache-2.0](LICENSE). У моделей и датасетов свои лицензии, например KazMMLU — CC-BY-NC-4.0, Belebele — CC-BY-SA-4.0.
