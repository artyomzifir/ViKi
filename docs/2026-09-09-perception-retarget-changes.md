# Отчёт по изменениям perception → retarget за 2026-09-09

## 1. Какой фидбек поступил

Работа началась с внешнего ревью `viki/retarget/`, тестов и контракта с
`prepare`. Ревью сначала подтвердило, что главный заявленный механизм не
является заглушкой: `solve_trajectory` действительно оптимизирует всю
траекторию `q[0:T]` одной разреженной QP на каждой итерации
Gauss–Newton/SQP. Также были положительно оценены Huber IRLS в data term,
линеаризация floor barrier и quintic approach.

После этого ревью перечислило 13 проблем и пробелов:

1. отчётный `objective` смешивал pose residual от старого `q` и регуляризаторы
   от нового `q` из-за aliasing `q.reshape(-1)`;
2. единый Huber применяется к совместной норме метров и масштабированных
   радиан, поэтому position/orientation не робастифицируются независимо;
3. основной pose Jacobian вычислялся конечными разностями и требовал `nq + 1`
   FK на кадр;
4. continuous joints UR не получали конечных position limits;
5. `q_approach` не проверялся на floor, collision и joint limits;
6. глобальный backtracking по геометрии был дорогим и не сохранял все
   нелинейные инварианты;
7. фиксированный Tikhonov-член назывался Levenberg–Marquardt damping;
8. `interpolated_mask` сохранялся, но не влиял на solver evidence;
9. `omega` нормировался на максимум собственного эпизода и потому был
   несопоставим между сценами;
10. невалидные targets интерполировались, а `confidence_floor=0.05` снова
    превращал их в data term;
11. отсутствовал последовательный per-frame IK baseline для честной абляции;
12. objective не раскладывался по data/velocity/acceleration/posture;
13. не хватало тестов на velocity limits, монотонность objective, физические
    лимиты UR, реальный robot path и batch-vs-sequential comparison.

Затем от пользователя поступили дополнительные требования:

- отдельно разобраться, что означает MediaPipe confidence `0.5`, и не
  отбрасывает ли он отдельные точки скелета;
- не менять стабильный путь вслепую, а сравнивать варианты на одном новом
  эпизоде, где движение выполняется только большим и указательным пальцами;
- хранить историю решений и отрицательных результатов в Markdown;
- показывать метрику по кадрам двумя линиями на pyplot-графиках;
- после сравнения закрепить linear interpolation как V2;
- отдельно проверить запрет экстраполяции за границы наблюдений;
- не сливать no-extrap с V2 до проверки на нескольких сценах, а оформить его
  отдельным воспроизводимым пресетом.

Контрольным стал эпизод
`data/datasets/new-dataset/2026-09-09_11-54-10`: 897 кадров, 30 FPS, правая
рука, две аппаратно синхронизированные Kinect-камеры.

## 2. Что сделано по пунктам ревью

| Пункт | Итог | Что произошло |
|---|---|---|
| 1. Aliasing objective | исправлено | Старый view больше не используется после in-place update. Objective заново вычисляется на одной целой кандидатной траектории. |
| 2. Общий Huber для m/rad | не исправлено | Поведение сохранено для сравнимости. Ограничение явно записано в `docs/math.md`; independent position/orientation Huber остаётся отдельным экспериментом. |
| 3. Numerical pose Jacobian | исправлено для Pinocchio | Добавлен аналитический Jacobian позиции task point и ориентации SO(3); generic test adapters сохраняют finite-difference fallback. |
| 4. UR position limits | исправлено | Для UR заданы явные физические диапазоны, передаваемые solver и replay screen даже для continuous Pinocchio joints. |
| 5. Непроверенный approach | исправлено | Approach начинается из известной floor-safe home pose, растягивается по velocity limit и затем целиком проверяется вместе с демонстрацией. |
| 6. Backtracking | исправлена корректность, не стоимость | Line search теперь не принимает рост exact objective и не теряет уже достигнутую floor/collision feasibility. Проверка всё ещё глобальная и дорогая. |
| 7. Терминология LM | исправлено | В UI и документации это называется fixed Tikhonov damping демпфированного Gauss–Newton/SQP, не adaptive LM. |
| 8. Fabricated evidence | частично исправлено | V3 обнуляет landmark/pose confidence по `observed_mask`; retarget учитывает нулевой `omega`. Сам `interpolated_mask` остаётся audit field, а стабильный V2 сохраняет старую episode-max схему. |
| 9. Episode-max `omega` | добавлен отдельный путь | Создан V3 с absolute confidence. После контрольных прогонов V3 не стал default; V2 оставлен воспроизводимым control. |
| 10. `confidence_floor` оживлял пропуски | исправлено | Floor применяется только при `omega > 0`; нулевой evidence остаётся нулевым. Интерполированные pose values являются только конечными placeholders для FK. |
| 11. Нет sequential baseline | исправлено | Добавлен опциональный causal per-frame IK → Savitzky–Golay baseline с отдельной траекторией, метриками и constraint margins. Он измеряется, но не исполняется. |
| 12. Нет разбивки objective | исправлено | Сохраняются `data`, `velocity`, `acceleration`, `posture`, `total` и история по итерациям. |
| 13. Покрытие | расширено | Добавлены тесты на objective consistency/monotonicity, zero-confidence, velocity, nonlinear collision, UR limits, analytic Jacobian, sequential baseline, approach replay и реальные HDF5-поля. |

Дополнительно обнаружена ещё одна математическая оговорка: QP использует
стандартный множитель `1/2` квадратичной формы, а отчётные temporal/posture
terms записаны без него. Это документировано, но в этот набор правок не
включено, чтобы не менять веса без отдельной A/B проверки.

## 3. Confidence `0.5` и экспериментальный V3

Проверка показала, что MediaPipe confidence `0.5` — не покоординатный фильтр
21 landmark. Эти значения подаются в graph как detection, presence и tracking
threshold. Если рука принята, MediaPipe возвращает весь набор точек; если нет —
теряется кадр руки целиком. Возвращаемый текущим backend общий score фактически
является handedness score, а не оценкой точности каждого сустава.

Для контролируемых экспериментов были добавлены:

- безопасный внутренний epsilon вместо буквального нуля, который аварийно
  завершает MediaPipe graph;
- раздельный `tracking_confidence`;
- optional relaxed handedness, чтобы колебание Left/Right label не вырезало
  единственную руку;
- поддержка настоящего `lm_score[21]` для backend, который его предоставляет;
- возможность включить detector score в triangulation quality;
- V3 с absolute confidence и нулевым evidence для заполненных landmark.

Это инфраструктура для эксперимента, а не новая стабильная истина. Дефолт был
возвращён на V2 после того, как более плотный admission не дал устойчивого
улучшения геометрически пригодных 3D-наблюдений.

## 4. Последовательные эксперименты с интерполяцией

### 4.1 Cubic → linear

На одном и том же `10_fused_observed.npz` сравнивались только способ заполнения
пропусков и одинаковый Savitzky–Golay `7/2`. Timestamps, landmark IDs,
`observed_points`, `observed_mask` и исходная confidence совпали.

| Метрика | Cubic | Linear edge hold | Изменение |
|---|---:|---:|---:|
| Максимум thumb–index | 25 211.4 мм | 256.7 мм | −99.0% |
| Thumb–index p95 | 434.5 мм | 156.7 мм | −63.9% |
| Pinch-centre step p95 | 12.27 мм/кадр | 8.36 мм/кадр | −31.8% |
| RMS второй разности pinch center | 5.43 мм | 0.98 мм | −82.0% |
| Valid pose frames | 828 | 858 | +30 |
| Positive IK evidence frames | 469 | 477 | +8 |

Вывод: natural cubic создавал катастрофическую экстраполяцию; linear стал
стабильным V2. При этом осталась физически плохая внутренняя область: 9 кадров
с thumb–index больше 250 мм внутри 93-кадрового асинхронного пропуска.

### 4.2 Linear edge hold → no extrapolation

В следующем изменении linear оставлен только между двумя наблюдениями. До
первого и после последнего наблюдения fused landmarks остаются `NaN`.

| Метрика | Linear edge hold | Linear no-extrap | Изменение |
|---|---:|---:|---:|
| Finite thumb/index frames | 897 | 853 | −44 неподтверждённых края |
| Первый/последний finite frame | 0 / 896 | 27 / 879 | соответствует support |
| Clean valid pose frames | 858 | 821 | −37 |
| Retarget-valid pinch frames | 858 | 805 | −53 |
| Positive IK evidence frames | 477 | 471 | −6 |
| Максимум thumb–index | 256.7 мм | 256.7 мм | без изменения |
| RMS второй разности pinch center | 0.978 мм | 0.983 мм | +0.005 мм |

Изменение удалило выдуманные края, но закономерно не исправило внутренний
93-кадровый разрыв. Рост p95 thumb–index с 156.7 до 162.4 мм вызван удалением из
распределения 44 спокойных edge-hold значений; совпадающая внутренняя часть
траектории практически не изменилась.

Первый downstream-прогон no-extrap нашёл интеграционную ошибку: fused wrist
стал честным `NaN`, но root state articulated hand получил скачок. Temporal gate
отклонил overlay: fitted jerk вырос с 1.13 до 2.39 мм. После исправления model
root отдельно удерживает ближайшую наблюдаемую wrist pose при нулевой sensor
confidence. Финальный overlay прошёл все gates: median anchor residual 9.43 мм,
p95 37.18 мм, fitted jerk 1.25 мм, wrist preservation 0 мм.

### 4.3 Полный IK A/B

Оба варианта прогнаны через одинаковую production-конфигурацию UR10 + Robotiq:

| Метрика итогового IK | V2 edge hold | No-extrap |
|---|---:|---:|
| Converged / iterations | да / 17 | да / 16 |
| Objective | 0.6053 | 0.6010 |
| Position RMSE | 42.53 мм | 44.32 мм |
| Position p95 | 99.29 мм | 106.23 мм |
| Orientation RMSE | 24.11° | 19.14° |
| Max joint velocity | 1.390 рад/с | 1.502 рад/с |
| Max joint acceleration | 2.129 рад/с² | 2.301 рад/с² |
| Positive-weight frames | 477 | 471 |

Оба решения сошлись и сохранили положительные collision, floor, joint-limit и
velocity margins. No-extrap улучшил происхождение данных, orientation RMSE и
немного objective, но ухудшил position error и пики joint motion. Поэтому это
правильный с точки зрения evidence кандидат, но ещё не доказанный новый default.

## 5. Итоговые профили

`stable-fused-hand-v2` остаётся профилем по умолчанию и контрольной линией.
`fused-hand-no-extrap-v1` доступен как отдельный locked preset во вкладке
Extract, API и CLI.

| Параметр | `stable-fused-hand-v2` | `fused-hand-no-extrap-v1` |
|---|---|---|
| Detector/fusion/SG | одинаковые | одинаковые |
| Fused interpolation | linear | linear |
| Leading/trailing gaps | nearest-value hold | `NaN`, zero evidence |
| Internal gaps | linear, unlimited | linear, unlimited |
| Gripper | continuous | continuous |

Команды для будущего сравнения сцен:

```bash
viki perceive EPISODE --profile stable-fused-hand-v2
viki perceive EPISODE --profile fused-hand-no-extrap-v1
```

## 6. Какие файлы изменены

Ниже перечислены все текущие изменённые tracked-файлы и роль изменений. Это
полезнее простого `git diff --stat`: отдельно отмечено, где находится алгоритм,
где только контракт/UI, а где тест.

### Конфигурация и контракты

- `data/default_configuration.json` — добавлен выключенный по умолчанию
  `RETARGET_SEQUENTIAL_BASELINE`.
- `data/user_configuration.json` — добавлен тот же переключатель; файл также
  содержит локальные операторские настройки текущей машины (`base x=-0.7`,
  active calibration `skrip`) и форматирование массивов. Это не результат
  математического эксперимента.
- `viki/config.py` — тип и fallback нового configuration key.
- `viki/contracts.py` — metadata метода/краевой экстраполяции в CLN, поля
  sequential baseline в PLAN.
- `pyproject.toml` — pytest ограничен `tests/unit_tests`, чтобы не собирать
  тесты из скачанных сторонних URDF-репозиториев под `models/`.

### Detection, lift и triangulation

- `viki/perception/backends/mediapipe.py` — безопасный zero threshold,
  отдельный tracking threshold и relaxed-handedness режим.
- `viki/perception/extract.py` — протяжка новых detector options и запись их в
  observation provenance без изменения исторических V1/V2 manifests.
- `viki/perception/geometry.py` — per-landmark detector score имеет приоритет
  над одним whole-hand score, если backend действительно его предоставляет.
- `viki/perception/triangulate.py` — optional detector-score factor в
  triangulation quality.
- `viki/perception/run.py` — профиль передаёт новые detector knobs в extraction.
- `viki/perception/profiles.py` — зафиксированы V1/V2 manifests, experimental
  V3 absolute-confidence и отдельный `fused-hand-no-extrap-v1`.

### Prepare и articulated hand

- `viki/dsp.py` — `interpolate_nans(..., extrapolate_edges=...)`.
- `viki/prepare/interpolate.py` — единый dispatcher `fill_fused_gaps` для
  cubic/linear и запрет краевой экстраполяции для обоих методов.
- `viki/prepare/checkpoints.py` — method/no-extrap входят в имя checkpoint,
  поэтому A/B-прогоны не перезаписывают друг друга.
- `viki/prepare/run.py` — выбор интерполятора из профиля, запись metadata,
  absolute/episode-max confidence paths, zero evidence для ненаблюдавшихся
  landmarks в V3 и запрет установки articulated overlay, не прошедшего gate.
- `viki/perception/articulated.py` — конечный model state мостит отсутствующий
  wrist, не подменяя fused measurement; motion metric игнорирует отсутствующие
  source differences вместо превращения всей метрики в `NaN`.
- `viki/prepare/README.md` — описаны checkpoint naming и два линейных пресета.

### Retarget solver и robot model

- `viki/retarget/solver.py` — исправленный objective, per-term breakdown и
  history; monotone nonlinear line search; exact floor/collision recheck;
  zero-confidence semantics; аналитический Pinocchio pose Jacobian; UR position
  limits; общий evaluator constraint margins; sequential baseline.
- `viki/retarget/robots.py` — floor-safe home configurations и явные limits для
  семейства UR.
- `viki/retarget/run.py` — конфигурация sequential baseline, передача limits,
  velocity-limited approach, полная проверка approach+demo, новые метрики и
  HDF5 plan schema v8.

### Replay, API и frontend

- `viki/replay/run.py` — `q_approach` теперь действительно исполняется и
  screen-ится; replay artifact по-прежнему содержит только demo-aligned frames
  для совместимости с export/annotations.
- `viki/replay/screen.py` — проверяет физические UR limits в actuated joint
  order, а не бесконечные limits continuous joints из Pinocchio.
- `viki/server/routes/pipeline.py` — отдаёт sequential trajectory и её
  покадровые ошибки в retarget scene API.
- `viki/server/static/js/core.js` — frontend default для sequential toggle.
- `viki/server/static/js/retarget.js` — переключатель baseline, метрики и точное
  название GN Tikhonov damping.
- `viki/server/static/js/scene3d.js` — отдельный фиолетовый слой sequential
  baseline для сравнения с target и batch-achieved trajectory.
- `viki/server/static/js/perception.js` — V2/V3/no-extrap в selector, locked
  параметры и пояснение edge-hold против bounded support.

### Тесты

- `tests/unit_tests/retarget/test_batch_solver.py` — objective consistency и
  monotonicity, velocity constraint, zero-confidence, nonlinear collision,
  sequential hold и post-smoothing violations.
- `tests/unit_tests/retarget/test_physical_grippers.py` — реальные UR limits и
  аналитический Jacobian task point между панелями Robotiq против finite diff.
- `tests/unit_tests/retarget/test_retarget_episode.py` — plan v8, margins,
  objective terms и sequential arrays на реальном robot description; skip теперь
  скрывает только отсутствие модели, а не ошибку solver.
- `tests/unit_tests/replay/test_replay_stub.py` — физический UR limit и реальное
  исполнение approach без нарушения demo-aligned replay artifact.
- `tests/unit_tests/perception/test_backends.py` — zero-threshold safety и
  strict/relaxed handedness.
- `tests/unit_tests/perception/test_lift_weights.py` — per-landmark score.
- `tests/unit_tests/perception/test_triangulate.py` — low-score admission с
  уменьшением quality вместо ложной высокой уверенности.
- `tests/unit_tests/perception/test_perceive.py` — неизменность V1, различия
  V2/V3/no-extrap и доказательство, что no-extrap отличается от V2 ровно
  соответствующим флагом плюс identity/description.
- `tests/unit_tests/prepare/test_interpolate.py` — bracketed-only linear,
  explicit cubic/linear routing и неизвестный method.
- `tests/unit_tests/prepare/test_fuse_triangulate.py` — absolute confidence,
  zero confidence заполненной точки и уникальные checkpoint paths.
- `tests/unit_tests/perception/test_articulated.py` — finite model pose при NaN
  edges без фабрикации fused measurements.

### Документация

- `docs/math.md` — фактическая математика confidence, interpolation, analytic
  Jacobian, objective, line search, constraints и известные ограничения.
- `docs/desc.md` — архитектурное описание обновлённого perception/retarget path.
- `docs/robust_retarget_experiments.md` — неизменяемый журнал E0–E3, метрики,
  отрицательный articulated run и решения о promotion/rollback.
- `docs/2026-09-09-perception-retarget-changes.md` — данный дневной отчёт.

## 7. Нетрекаемые артефакты, которые были созданы или обновлены

- `data/analysis/plots/cubic-vs-linear-thumb-index.png` — cubic/linear ошибка
  thumb–index по кадрам.
- `data/analysis/plots/linear-edge-extrapolation-ab.png` — edge-hold/no-extrap
  и final IK data weight по кадрам.
- `data/analysis/no-extrap-retarget-ab.json` — полный UR10 + Robotiq A/B.
- `intermediates/prepare/triangulate__interp-linear__gap-all__sg-7-2/` —
  linear edge-hold checkpoints.
- `intermediates/prepare/triangulate__interp-linear-no-extrap__gap-all__sg-7-2/`
  — no-extrap checkpoints.
- `intermediates/archive/stable-fused-hand-v2-cubic-pre-linear/` — исходный
  cubic control.
- `intermediates/archive/stable-fused-hand-v2-linear-edge-hold/` — снимок
  linear control перед no-extrap.
- `intermediates/archive/no-extrap-failed-before-model-bridge/` — специально
  сохранённый отрицательный результат до исправления articulated bridge.
- `intermediates/baselines/stable-fused-hand-v2/` — актуальный защищённый
  linear edge-hold baseline.
- `intermediates/baselines/fused-hand-no-extrap-v1/` — отдельный защищённый
  no-extrap baseline.
- Активный `cln.npz` контрольного эпизода перегенерирован с metadata
  `profile=fused-hand-no-extrap-v1`, `fused_interpolation=linear`,
  `fused_extrapolate_edges=false`.

SHA-256 защищённых CLN:

- cubic: `1d820cf7be39d654c2d0ffa04810b1b724e026ce7dadd9ee2c85598803ee2ed1`;
- linear edge hold: `5582be962b2158e14cdba139eedebc9e400670ebdfeed7426d1dd31f2fc10295`;
- linear no-extrap: `8037027140a4dea9bac41be4f27b68e9e771f23ba98f4a9b2b0c30736b57818b`.

## 8. Проверки

- Полный Docker test suite: **278 passed, 3 skipped**, 36 предупреждений OSQP.
- Финальная профильная выборка после разделения пресетов: **16 passed**.
- Реальный prepare нового пресета на 897 кадрах: успешно; 13 002 из 18 816
  joints триангулированы, median reprojection error 1.55 px.
- Полный UR10 + Robotiq IK A/B: оба варианта converged.
- Frontend `node --check`: успешно.
- `git diff --check`: успешно.

## 9. Что намеренно осталось

1. Общий Huber всё ещё смешивает position и orientation residual после их
   весового масштабирования.
2. Масштаб отчётных regularizer terms и локальной QP-модели требует отдельной
   унификации.
3. Глобальный nonlinear line search корректнее прежнего, но остаётся дорогим.
4. V3 absolute confidence ещё не доказан на нескольких эпизодах и не default.
5. `interpolated_mask` напрямую не читается retarget; evidence передаётся через
   `omega` и landmark confidence.
6. `max_gap=0` означает unlimited internal fill. Главный следующий эксперимент
   — pair-aware evidence: нельзя считать pinch target измеренным, если thumb и
   index наблюдались в разные моменты длинного разрыва.
7. No-extrap нужно сравнить на нескольких сценах прежде, чем заменять им V2.

Итог дня: solver стал существенно лучше проверяемым и честнее репортит
ограничения/evidence, cubic устранён из стабильного V2, а no-extrap сохранён как
отдельный кандидат. Это ещё не основание объявлять pipeline готовым к слепому
физическому replay: оставшиеся пункты выше должны быть закрыты отдельными
контролируемыми экспериментами.
