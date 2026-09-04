# Аудит точности ViKi на `pick_up_u`

Дата среза: 4 сентября 2026 года. Эпизод:
`data/datasets/new-dataset/2026-09-03_16-58-45` (`pick_up_u`), 898 кадров,
30 fps, две Azure Kinect. Ветка: `refactor/kiss-pipeline-structure`, базовый
commit `1947f80`; раздел о checkpoint-артефактах относится также к текущему
рабочему дереву поверх этого commit.

## Короткий вывод

Улучшение настоящее, но его корректно называть улучшением **внутренней
геометрической согласованности и временной стабильности**, а не измеренной
абсолютной 3D-точности. В эпизоде нет внешнего ground truth, поэтому MPJPE и
ошибка кончиков пальцев в миллиметрах относительно эталона пока неизвестны.

На сопоставимых результатах `hand_fit` замена legacy `xyz_mean` на multi-view
triangulation дала:

- валидность позы 85,75% → **98,66%** (+12,91 п.п.);
- пустые depth-блоки `hand_fit` 14,25% → **1,34%**;
- median point-to-capsule residual 9,59 → **8,97 мм** (−6,5%);
- P90 residual 26,11 → **22,47 мм** (−13,9%);
- median шага fitted wrist 13,95 → **6,16 мм/кадр** (−55,8%);
- P95 шага fitted wrist 59,83 → **31,90 мм/кадр** (−46,7%);
- сохранённая метрика wrist jerk after fit 0,09693 → **0,04946** (−49,0%).

Это совпадает с визуальной проверкой: clean triangulated trajectory живёт в
облаке и заметно меньше дёргается. Наблюдавшиеся редкие схлопывания тоже
подтверждаются данными. Их главный источник — не triangulation и не
Savitzky–Golay, а безлимитная покоординатная cubic-интерполяция длинных
пропусков.

## Что именно сейчас является clean-вариантом

Активный `cln.npz` явно помечен так:

```text
active_variant = clean_triangulated_landmarks
pose_source = landmarks
perception_fuse_mode = triangulate
```

В нём нет массивов `hand_fit_*`. Хотя в пользовательской конфигурации задано
`PERCEPTION_HAND_POSE_SOURCE = hand_fit`, общий селектор `cln_pose_keys()`
корректно откатывается на landmark pose, если fitted pose отсутствует. Поэтому
viewer, retarget и export для текущего активного файла фактически используют
чистую triangulated landmark trajectory, а не `cln_triangulate.npz` с capsule
fit.

Это различие важно не терять в последующих сравнениях:

| файл | fusion | фактический pose source | назначение |
|---|---|---|---|
| `cln.npz` | triangulate | landmarks | текущий визуально подтверждённый clean-вариант |
| `cln_triangulate.npz` | triangulate | hand fit | прежний A/B fitted-вариант |
| `cln_xyzmean.npz` | xyz mean | hand fit | legacy fitted-control |

## Проверка входных условий

### Детектор и покрытие камер

В `raw/observations.npz` сохранено 1705 камерных наблюдений:

| показатель | значение |
|---|---:|
| наблюдения `kinect_0` | 896 |
| наблюдения `kinect_1` | 809 |
| кадры с обеими камерами | 807 / 898 = **89,87%** |
| кадры только с одной камерой | 91 / 898 = **10,13%** |
| кадры без детекции обеих камер | 0 |

Текущий stereo triangulator не выдумывает точку по одному виду. Поэтому все 91
single-view кадра дают ноль triangulated joints и должны восстанавливаться
последующими временными/анатомическими стадиями либо оставаться невалидными.

### Синхронизация: хорошая медиана, но тяжёлый хвост

По `raw/timestamps.json` межкамерная разница рассчитана как
`|offset_kinect_0 - offset_kinect_1|`:

| percentile / gate | разница |
|---|---:|
| median | **0,503 мс** |
| P90 | 4,518 мс |
| P95 | **5,409 мс** |
| P97.5 | 31,388 мс |
| P99 | 32,374 мс |
| max | 35,809 мс |
| больше 10 мс | 38 / 898 = **4,23%** |
| больше половины периода 30 fps (16,67 мс) | 38 / 898 = **4,23%** |

Следовательно, формулировка «hardware sync полностью решил синхронизацию» была
слишком сильной. Большинство кадров синхронизировано хорошо, но 38 групп похожи
на срыв на один период кадра. Среди этих 38 кадров 34,2% попадают в
анатомические outlier frames активной траектории, против 18,1% среди остальных
кадров. Это корреляция, не доказательство причинности, но tail нужно явно
гейтить и диагностировать.

### Геометрия triangulation

Актуальный `raw/joints3d.npz` содержит 898 × 21 сустав:

| показатель | значение |
|---|---:|
| валидно по двум видам | 13 556 / 18 858 = **71,88%** |
| недоступно | 5 302 / 18 858 = **28,12%** |
| one-view результата | 0 — намеренно |
| median суставов на кадр | 17 из 21 |
| кадров со всеми 21 суставами | 115 / 898 = 12,81% |
| кадров хотя бы с 4 суставами | 807 / 898 = 89,87% |
| кадров с 0 суставов | 91 / 898 = 10,13% |
| reprojection error median / P95 / max | **1,30 / 3,06 / 3,89 px** |
| ray angle для валидных точек, P5 / median / P95 | 58,21° / **68,01°** / 78,29° |
| quality валидных точек, P5 / median / P95 | 0,618 / **0,838** / 0,968 |

Углы лучей широкие, а reprojection residual укладывается примерно в 4 px:
stereo geometry для принятых точек хорошо обусловлена. Главная проблема уже не
качество принятых triangulated joints, а покрытие пропусков без разрушения
анатомии.

Старый `docs/triangulation_results.md` указывает распределение views
`0 / 1911 / 16947`. Оно **не соответствует текущему** `joints3d.npz`, где
распределение `5302 / 0 / 13556` для `0 / 1 / 2` views. Для новых выводов нужно
использовать цифры из этого отчёта и текущего артефакта.

## A/B: triangulation против legacy `xyz_mean`

Это сопоставление двух root-артефактов с одинаковой конфигурацией capsule
`hand_fit`; меняется прежде всего источник landmark warm start.

| метрика | `xyz_mean + hand_fit` | `triangulate + hand_fit` | изменение |
|---|---:|---:|---:|
| pose valid | 85,75% | **98,66%** | +12,91 п.п. |
| empty data frames | 14,25% | **1,34%** | −12,91 п.п. |
| point→capsule residual median | 9,59 мм | **8,97 мм** | −6,5% |
| point→capsule residual P90 | 26,11 мм | **22,47 мм** | −13,9% |
| fitted wrist step median | 13,95 мм/fr | **6,16 мм/fr** | −55,8% |
| fitted wrist step P95 | 59,83 мм/fr | **31,90 мм/fr** | −46,7% |
| fitted wrist third-difference P95 | 202,75 | **108,97** | −46,3% |
| `jerk_after_m` из hand-fit metrics | 0,09693 | **0,04946** | −49,0% |
| доля data/cloud term в objective | 7,05% | **19,89%** | +12,84 п.п. |
| total objective, diagnostic only | 1295,18 | **387,40** | −70,1% |

`total objective` нельзя трактовать как физическую ошибку: это сумма
разнородных взвешенных residual blocks. Но падение одновременно с residual,
step, jerk и empty-frame fraction показывает, что optimiser получает гораздо
более согласованную начальную геометрию и меньше борется с ней.

## Ablation по стадиям: где появляется мусор

Для каждого варианта сохранены одни и те же границы:

```text
00 per-camera observed
05 per-camera filled
10 fused observed
20 fused filled
30 smoothed
40 hand fit (только когда эта стадия действительно запускалась)
```

Ниже приведены метрики из
`intermediates/prepare/comparison.json`. `finite` — доля конечных 3D joints,
`pose valid` — кадры, где можно построить устойчивый palm frame, `anatomy
outliers` — кадры минимум с двумя слишком короткими/длинными костями,
`palm collapse` — кадры с длиной одной из palm edges меньше 55% reference.

| fusion / stage | max gap | finite | pose valid | anatomy outliers | palm collapse | wrist step med / P95, мм | wrist 3rd diff P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xyz mean / observed | — | 96,33% | 81,18% | **52,12%** | 4,68% | 7,58 / 43,37 | 122,33 |
| xyz mean / smoothed | all | 100% | 85,75% | **43,10%** | 5,23% | 5,29 / 18,64 | 29,63 |
| triangulate / observed | — | 71,88% | 38,53% | **1,22%** | **0%** | 3,27 / 13,70 | 11,94 |
| triangulate / smoothed | 3 fr | 77,61% | 44,54% | 2,78% | **0%** | 2,66 / 13,51 | 2,91 |
| triangulate / smoothed | 6 fr | 81,31% | 50,00% | 3,34% | **0%** | 2,68 / 13,08 | 2,65 |
| triangulate / smoothed | 12 fr | 86,75% | 60,91% | 3,90% | **0%** | 2,64 / 13,05 | 2,63 |
| triangulate / smoothed | 24 fr | 90,08% | 70,04% | 7,02% | **0%** | 2,35 / 12,45 | 2,53 |
| triangulate / smoothed | 30 fr | 90,79% | 72,94% | 9,91% | **0%** | 2,09 / 12,21 | 2,52 |
| triangulate / smoothed | 45 fr | 92,35% | 82,29% | 10,69% | **0%** | 1,60 / 12,04 | 2,45 |
| triangulate / smoothed | all | 100% | 98,66% | **19,38%** | **3,12%** | 1,88 / 12,05 | 2,40 |

### Интерпретация

1. **`xyz_mean` действительно является источником мусорной геометрии.** Оно
   даёт больше заполненных точек, но уже до интерполяции 52,1% кадров имеют
   анатомические выбросы. Smoothing делает движение спокойнее, но не может
   восстановить физически неверные кости.
2. **Принятые triangulated observations чистые.** На границе `10` только 1,22%
   anatomy outliers и ни одного palm collapse. Цена — 71,88% joint coverage.
3. **Savitzky–Golay полезен.** Например при gap 12 P95 третьей разности падает с
   10,51 после fill до 2,63 после smoothing, а anatomy outliers немного
   уменьшаются 4,12% → 3,90%. Он не создаёт схлопывания.
4. **Мусор добавляет unbounded gap fill.** При `max_gap=all` переход observed →
   filled повышает anatomy outliers с 1,22% до 19,82% и создаёт 28 palm-collapse
   frames; smoothing оставляет 19,38% и те же 28 frames.
5. **Видимая плавность и физическая корректность здесь конфликтуют.** Fill-all
   даёт почти непрерывную траекторию и лучший temporal score, но часть этой
   непрерывности сфабрикована независимыми cubic splines для XYZ каждого
   сустава.

Схлопывания активного clean-варианта локализованы в 28 кадрах:
`451–454`, `470–479`, `482–495`. Это согласуется с визуальным наблюдением
нескольких коротких эпизодов деградации, а не постоянной ошибки triangulation.

## Что было реализовано для проверяемости

Текущий рабочий код больше не позволяет «потерять» место появления ошибки:

- `cln.npz` дополнен optional provenance-полями: `observed_points`,
  `filled_points`, `observed_mask`, `interpolated_mask`,
  `perception_fuse_mode`, `checkpoint_stage`, `checkpoint_params_json`;
- prepare сохраняет per-camera observed/filled и fused observed/filled/smoothed
  boundaries;
- каждый run лежит в параметризованной директории вида
  `intermediates/prepare/triangulate__gap-12__sg-7-2/`;
- `manifest.json` фиксирует реально запрошенный и реально использованный fusion
  mode, gap, окно и polyorder;
- `comparison.json` сохраняет availability, temporal и anatomy diagnostics;
- `viki checkpoints <episode> --fusion triangulate xyz_mean
  --interp-max-gap N` строит варианты неразрушающе и не заменяет активный
  `cln.npz`;
- viewer-compatible checkpoint’ы выбираются как отдельные stage variants.

Для `pick_up_u` уже сохранены triangulation runs с gap
`3, 6, 12, 24, 30, 45, all` и legacy control `xyz_mean + gap-all`.

## Что пока нельзя утверждать

По этому эксперименту нельзя честно написать, что абсолютная ошибка кисти
стала, например, «меньше сантиметра». Сейчас измерены:

- reprojection consistency между двумя камерами;
- согласованность с depth cloud через capsule residual;
- стабильность длины костей и palm geometry;
- temporal step/acceleration/third difference;
- availability и доля прямых наблюдений.

Не измерены:

- MPJPE относительно motion-capture или размеченного 3D ground truth;
- fingertip error отдельно от wrist/palm;
- угловая ошибка каждого сустава относительно reference;
- абсолютная ошибка world frame и калибровки;
- правильность контакта пальца с объектом;
- end-to-end ошибка после robot retargeting.

Поэтому итог этого аудита — «геометрия стала существенно согласованнее и
стабильнее», но не абсолютный сертификат миллиметровой точности.

## Рекомендуемая следующая комбинация стадий

Не следует делать активным ни legacy `xyz_mean`, ни текущий fill-all как
финальную стратегию. Наиболее обоснованный следующий эксперимент:

```text
triangulate observed
  → заполнять только короткие gaps (начать с 12 и 24 кадров)
  → Savitzky–Golay
  → batch hand_fit, где оставшиеся длинные gaps являются empty data blocks
  → сравнить 40_hand_fit между gap-12, gap-24, gap-45 и gap-all
```

Причина: capsule hand model сохраняет длины костей и может протянуть временную
траекторию через пропуск анатомически, тогда как независимые cubic splines этого
не гарантируют. При этом `hand_fit` имеет известный correspondence basin и не
может надёжно открыть сгибание, которого нет в warm start, поэтому gap-12 и
gap-24 нужно сравнить визуально и по joint angles, а не выбирать только по
минимальному jerk.

Отдельно нужно:

1. помечать или исключать 38 frame-slip групп с межкамерной разницей >10 мс;
2. не превращать single-view кадры в равноправный stereo результат — хранить их
   как отдельный fallback с пониженной confidence;
3. добавить manual/ground-truth validation хотя бы для 50–100 ключевых кадров,
   особенно fingertip/contact frames;
4. после выбора gap policy изменить `PERCEPTION_INTERP_MAX_GAP`: сейчас и
   default, и user config равны `0`, то есть production prepare по-прежнему
   заполняет пропуски любой длины.

Практический кандидат для первого `hand_fit` ablation — gap 12: он сохраняет
86,75% суставов, не имеет palm collapse и держит anatomy outliers на 3,90%.
Gap 24 — более непрерывный вариант (90,08% finite, 70,04% pose valid), но уже с
7,02% anatomy outliers. Выбор между ними должен делать fitted-stage результат,
а не один показатель плавности.

## Верификация кода

4 сентября 2026 года в Docker запущены целевые тесты:

```text
tests/unit_tests/perception/test_triangulate.py
tests/unit_tests/prepare/test_fuse_triangulate.py
tests/unit_tests/prepare/test_interpolate.py
tests/unit_tests/cameras/test_sync_stats.py
tests/unit_tests/cameras/test_wired_sync.py

23 passed in 0.89 s
```

Они проверяют triangulation core, подключение triangulated artifact к prepare,
сохранение stage boundaries, ограничение длины gap, sync statistics и wired-sync
configuration. Это защита от программной регрессии, но не замена реальному 3D
ground truth.

## Источники данных и кода

- `data/datasets/new-dataset/2026-09-03_16-58-45/raw/observations.npz`
- `data/datasets/new-dataset/2026-09-03_16-58-45/raw/joints3d.npz`
- `data/datasets/new-dataset/2026-09-03_16-58-45/raw/timestamps.json`
- `data/datasets/new-dataset/2026-09-03_16-58-45/cln.npz`
- `data/datasets/new-dataset/2026-09-03_16-58-45/cln_triangulate.npz`
- `data/datasets/new-dataset/2026-09-03_16-58-45/cln_xyzmean.npz`
- `data/datasets/new-dataset/2026-09-03_16-58-45/intermediates/prepare/comparison.json`
- `viki/perception/triangulate.py`
- `viki/prepare/run.py`
- `viki/prepare/interpolate.py`
- `viki/prepare/checkpoints.py`
- `viki/perception/hand_fit.py`
- `viki/contracts.py`
- `docs/triangulation_results.md`
- `docs/hand_fit_batch_design.md`
