<div align="center">

<img src="assets/banner.png" alt="LinguaPrune" width="720">

**Языково-ориентированный прунинг экспертов в MoE-моделях.**
Оставляем экспертов, которые реально нужны вашим языкам, остальных убираем, затем квантуем.
Первая цель — казахский 🇰🇿 + русский.

<p>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-A020F0?style=for-the-badge&labelColor=0D1117&logo=apache&logoColor=white"></a>
<a href="https://github.com/s0nakh/linguaprune/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/s0nakh/linguaprune/tests.yml?style=for-the-badge&labelColor=0D1117&logo=githubactions&logoColor=white&label=tests"></a>
<a href="https://github.com/s0nakh/linguaprune/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/s0nakh/linguaprune?style=for-the-badge&labelColor=0D1117&logo=github&color=A020F0"></a>
<a href="https://github.com/s0nakh/linguaprune/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/s0nakh/linguaprune?style=for-the-badge&labelColor=0D1117&logo=git&logoColor=white&color=39FF14"></a>
<a href="#дорожная-карта"><img alt="Status" src="https://img.shields.io/badge/status-research%20preview-FF8C00?style=for-the-badge&labelColor=0D1117&logo=jupyter&logoColor=white"></a>
</p>
<p>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-39FF14?style=for-the-badge&labelColor=0D1117&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&labelColor=0D1117&logo=pytorch&logoColor=white"></a>
<a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5.x-FFD21E?style=for-the-badge&labelColor=0D1117&logo=huggingface&logoColor=white"></a>
<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-GPU%20T4%20%C3%972-20BEFF?style=for-the-badge&labelColor=0D1117&logo=kaggle&logoColor=white"></a>
</p>
<p>
<a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125"><img alt="Model" src="https://img.shields.io/badge/model-OLMoE--1B--7B-F0529C?style=for-the-badge&labelColor=0D1117&logo=huggingface&logoColor=white"></a>
<a href="https://huggingface.co/datasets/facebook/belebele"><img alt="Eval" src="https://img.shields.io/badge/eval-Belebele-00C2FF?style=for-the-badge&labelColor=0D1117&logo=meta&logoColor=white"></a>
<a href="#"><img alt="Languages" src="https://img.shields.io/badge/languages-kk%20%C2%B7%20ru-39FF14?style=for-the-badge&labelColor=0D1117&logo=googletranslate&logoColor=white"></a>
<a href="README.ru.md"><img alt="Docs" src="https://img.shields.io/badge/docs-EN%20%2B%20RU-A020F0?style=for-the-badge&labelColor=0D1117&logo=readthedocs&logoColor=white"></a>
</p>

[English](README.md) · [Связанные работы](docs/prior-art.ru.md)

</div>

---

## Зачем

Суверенные и многоязычные MoE-модели огромные. Например, MoE на 247 млрд параметров весит ~494 ГБ в BF16, это около 7 серверных видеокарт. Но для казахского и русского модель редко зовёт многих своих экспертов. LinguaPrune измеряет, **какие эксперты реально нужны языку**, и убирает остальных, чтобы модели хватало меньшего числа видеокарт.

> **Статус (ноябрь 2026):** мы *измерили*, как увольнение экспертов влияет на качество; увольнение имитируется маской роутера.
> Сама модель **пока не стала меньше**: физическое удаление весов, экспорт и квантизация в 4 бита — следующий шаг.

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

**Ноутбук 02 — качество после прунинга.** Перплексия измерена на [Belebele](https://huggingface.co/datasets/facebook/belebele): 150 отрывков с одинаковым смыслом на всех языках. Значения даны относительно исходной модели (1.00× = без потерь; меньше = лучше):

| Оставлено экспертов (из 64) | kk · наш | kk · случайные (5 раз) | kk · выбранные по англ. | ru · наш | ru · случайные | en · наш |
|---|---|---|---|---|---|---|
| 48 | **1.03×** | 2.0–4.6× | 1.55× | **1.09×** | 2.3–5.6× | 1.45× |
| 38 (−41%) | **1.10×** | 16–1668× | 1.86× | **1.27×** | 11–25983× | 2.08× |
| 32 | **1.16×** | 8.5–285× | 27× | **1.51×** | 18–523× | 3.56× |
| 24 | **1.45×** | 393–7544× | 214× | **2.51×** | 981–17520× | 11.2× |

<img src="experiments/02/ru/perplexity_vs_experts.png" alt="Рост перплексии в зависимости от числа экспертов" width="900">

- **Выбор экспертов по маршрутизации kk/ru работает.** Если убрать 41% экспертов, перплексия казахского растёт на 10%, русского на 27%. Случайный набор того же размера ломает модель (×16 – ×1668 для казахского).
- **Язык калибровки важен.** Эксперты, выбранные по английскому тексту, для kk/ru намного хуже (×1.9 / ×3.3 при 38 экспертах). На этой модели результат подтверждает [arXiv:2611.07654](https://arxiv.org/abs/2611.07654), а не [arXiv:2610.15578](https://arxiv.org/abs/2610.15578).
- **Платит английский, как и задумано** (×2.1 при 38 экспертах). Для английского выбор по частоте не лучше случайного: английский использует экспертов широко.
- **Учёт веса роутера не дал выигрыша** по сравнению с простой частотой.
- **Русский теряет больше казахского.** Во всех слоях оставлено одинаковое число экспертов, а ранним слоям их нужно больше (см. ноутбук 01). Следующий шаг — разный бюджет по слоям.
- **Самопроверки пройдены:** маска совпадает с физическим удалением (совпадение выбора 1.0, разница весов 0).

⚠️ **Ограничения:**
- Точность ответов на Belebele даже у *исходной* модели для kk/ru близка к угадыванию: 31% и 40% при 25% случайных, и модель сильно тяготеет к ответу «C». Поэтому точность ущерб не измеряет, надёжный сигнал здесь — перплексия.
- Перплексия ≠ качество на прикладных задачах: «+10% перплексии» **не** значит «ответы на 10% хуже».
- Исходная OLMoE изначально слабо знает казахский, поэтому мы сохранили *слабый* исходный уровень, а не сильный. Дальше — более сильные многоязычные MoE-модели.
- Это одна небольшая модель, и маска пока не экономит память.

Сырые цифры: [`experiments/02/`](experiments/02/).

## Ноутбуки

| # | Ноутбук | Что делает | Открыть |
|---|---|---|---|
| 01 | [`01_expert_profiling`](notebooks/01_expert_profiling.ipynb) | Каких экспертов использует каждый язык; сколько можно убрать | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/01_expert_profiling.ipynb) |
| 02 | [`02_prune_and_measure`](notebooks/02_prune_and_measure.ipynb) | Убрать экспертов и измерить качество на kk/ru/en в сравнении со случайными и «английскими» экспертами | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/02_prune_and_measure.ipynb) |
| 03 | [`03_remove_and_save`](notebooks/03_remove_and_save.ipynb) | Физически удалить экспертов, проверить, что уменьшенная модель совпадает с имитацией, сохранить и загрузить заново | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/03_remove_and_save.ipynb) |

Ноутбуки двуязычные: в каждом текстовом блоке сначала английский, ниже русский перевод.

**Запуск на Kaggle:**
1. Нажми *Open in Kaggle* в таблице выше.
2. Выбери **GPU T4 x2**, **Internet On**.
3. Нажми **Save Version → Save & Run All**.

## Дорожная карта

- [x] 01 · Профилирование роутера на OLMoE
- [x] 02 · Качество после прунинга (Belebele kk/ru/en)
- [ ] 03 · Физически удалить экспертов, сохранить меньшую модель, проверить совпадение с замером *(ноутбук готов)*
- [ ] 04 · Экспорт в GGUF 4 бита и запуск на ноутбуке с 8 ГБ
- [ ] 05 · Тот же пайплайн на Qwen3-30B-A3B
- [ ] Критерий в духе REAP и бюджет по слоям
- [ ] Прунинг + 4 бита против равномерных 2–3 бит **при одинаковой памяти**
- [ ] Большие MoE-модели для казахского/русского
- [ ] CLI `pip install linguaprune` для любого языка

**Проверка на опровержение.** При одинаковой памяти «прунинг + 4 бита» должен обойти равномерную квантизацию и меньшую dense-модель по качеству на kk/ru. Если не обойдёт, мы опубликуем и этот результат.

## Python-пакет

Код прунинга, которым пользуются ноутбуки, вынесен в небольшой протестированный пакет:

```bash
pip install "git+https://github.com/s0nakh/linguaprune@v0.1.0"
```

```python
from linguaprune import language_frequency_score, top_experts, masked_experts, prune_experts

kept = top_experts(language_frequency_score(router_counts, ["kk", "ru"]), keep=38)
with masked_experts(model, kept):   # имитация: уволенных экспертов нельзя выбрать
    ...
prune_experts(model, kept)           # удалить по-настоящему; затем model.save_pretrained(...)
```

Тесты проверяют, что физическое удаление даёт ровно те же выходы, что маска. Для этого используется копия MoE-классов OLMoE из `transformers` 5.x: `pip install -e ".[test]" && pytest`.

## Лицензия

[Apache-2.0](LICENSE). У моделей и датасетов свои лицензии, например KazMMLU — CC-BY-NC-4.0, Belebele — CC-BY-SA-4.0.
