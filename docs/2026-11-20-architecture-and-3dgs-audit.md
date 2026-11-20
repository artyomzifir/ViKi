# Архитектура ViKi и подключение 3DGS

Дата: 20 ноября 2026. База: `dev-eval`, `135e6fb` от 2 ноября.

ViKi уже предоставляет файловый offline-конвейер: RGB-D → рука и объекты →
кинематический план → trajectory bundle. Расширение для 3DGS должно добавлять
реконструкцию визуальных assets и синтез кадров поверх этих артефактов.
Реконструкция, robot compositing и генерация синтетических видеодатасетов пока
предложены, но не реализованы. План направления находится в
[proposal от 1 ноября](2026-11-01-thesis-pivot-3dgs-dataset-generation.md).

## Карта репозитория

| Каталог или модуль | Ответственность и использование | Состояние |
|---|---|---|
| `viki/contracts.py` | Общие DTO, Episode, ключи NPZ/HDF5 | Основная граница данных между стадиями |
| `viki/episode.py`, `datasets.py` | Пути, meta/status, организация эпизодов | Файловая единица работы; общего workflow engine нет |
| `viki/cameras/` | SDK, синхронизация, `record` → `raw/` | Реализовано; двухкамерный стенд требует независимых USB-контроллеров |
| `viki/calibration/` | Intrinsics/extrinsics, world anchor, ChArUco, validation | Реализовано, есть специфика SDK Kinect |
| `viki/perception/` | `extract`, `perceive`, `cloud`, SAM, object models, hand fit | Реализованы разные ветви обработки; качество оценивается отдельно |
| `viki/prepare/` | `rec.npz` → `cln.npz`, checkpoints, fusion, smoothing | Рабочая стадия; default profile в коде `stable-fused-hand-v4` |
| `viki/retarget/` | Whole-trajectory IK, модели роботов и захватов, `plan.h5` | Рабочий solver; object grasp ограничен поддержанными задачами и моделями |
| `viki/benchmark/` | HO-Cap import/evaluation и reference-данные | Внешний источник проверки; восемь видов полезны для GS |
| `viki/object_centric.py` | Post-IK object-relative sidecar | Диагностика относительно object tracks, не самостоятельная policy |
| `viki/replay/` | Явный CLI dry-run, копия плана и joint-limit check | Сохранён для совместимости; не часть `viki run` |
| `viki/export/` | NPZ trajectory bundle и optional LeRobot | Bundle работает; LeRobot частичный и зависит от dry-run/labels |
| `viki/render/` | Лёгкие pixel helpers | numpy/cv2/matplotlib; тяжёлое GS-обучение сюда не относится |
| `viki/server/` | FastAPI, jobs, статические JS-панели | Исследовательский интерфейс; выполнение заданий в памяти процесса |
| `experiments/` | MuJoCo, альтернативные захваты, BC/SAC, поиски команд | 76 Python-файлов на базовой ревизии; часто конкретные scene fixtures |
| `scripts/`, `tools/` | Batch, аудиты, отчёты, загрузка HO-Cap | Служебные entrypoints поверх основного кода |
| `docs/` | Протоколы, решения, результаты, планы | Даты и статус документа существенны; старые планы исторические |
| `paper/` | Диплом и сохранённые численные evidence artifacts | JSON/NPZ/хеши не заменяют исходные RGB-D и внешние модели |
| `tests/unit_tests/` | Тесты основного кода и экспериментов | E2E smoke использует синтетический план; реальный IK проверяется отдельно |

`viki/` содержит десять пакетов. CLI находится в `viki/cli.py`; frontend —
plain HTML/CSS/ES modules без build step. `scene3d.js` используется Viewer,
Extract и Retarget для облаков, скелетов, meshes и траекторий.

## Артефакты и существующие пути

```text
raw/ → rec.npz → cln.npz → plan.h5 → trajectory bundle
  ├→ cloud/                              Viewer geometry
  └→ SAM masks → semantic cloud → object_models.npz
                                      + cln.npz + plan.h5
                                      → object_relative.npz
```

В `raw/` есть видео по камерам, depth NPY, intrinsics/extrinsics, timestamps,
world anchor и SDK-калибровка для offline registration. Запись RGB использует
MP4: декодирование PNG для обучения не восстанавливает утраченную компрессией
деталь. Пропущенная depth-карта должна оставаться отсутствующим измерением.

`cloud/` — производный Viewer artifact. По умолчанию он вычитает фон,
ограничивает workspace, прореживает и voxel-фильтрует точки. Его можно применять
для диагностики и выбранной инициализации, но он не заменяет исходные изображения,
камеры и полную сцену для GS.

SAM masks доступны покадрово через `MaskArchive`. Object model содержит
канонические поверхности, pose tracks, confidence и rotation information.
Это геометрия и оценки движения, не готовый textured/Gaussian asset.

`plan.h5` уже содержит `robot_visuals_json` и
`robot_joint_placements_calibration`: существующий Viewer загружает meshes и
применяет per-frame transforms. Для первого robot render не нужно повторять IK.
Поле `achieved_*` означает FK результата IK, а не выполненную манипуляцию.

HO-Cap импортируется в обычный episode layout. Oracle masks и GT-assisted
совмещение объектов необходимо отмечать в provenance; их результаты нельзя
выдавать за автономную end-to-end сегментацию/оценку позы.

## Границы и ограничения расширения

1. **Один активный план.** `retarget_episode` пишет `ep.plan_h5`. Для нескольких
   embodiments нужны именованные планы или explicit output path, иначе новый
   робот заменяет предыдущего. Переписывать solver для этого не требуется.
2. **Статус стадии не описывает run полностью.** `mark_stage` пишет `done` и
   дополнительные поля в JSON. Нет общего сравнения input/config fingerprints
   или DAG инвалидирования. SAM/object-model имеют собственные проверки и
   инвалидирование; их наличие не делает такую проверку универсальной.
3. **Очередь заданий не сохраняется и не управляет долгим процессом.** Подробности
   ниже. Это ограничение управления работой, а не запрет на новые алгоритмы.
4. **Публичные границы неполные.** В коде встречаются импорты внутренних модулей
   соседей; `server/routes/pipeline.py` на базе занимает 1322 строки и готовит
   геометрию. Новый router должен вызывать публичный stage API.
5. **Описания местами устарели.** Например, Prepare README описывает старый
   default profile, а proposal помечает будущие GS contributions как
   `Implemented`. Статус определяется кодом и сохранёнными результатами.

Общая реорганизация репозитория не является предварительным условием прототипа.
Нужно выделять только интерфейсы, которыми действительно пользуются новые стадии.

## Что означает ограничение очереди

`viki/server/jobs.py` хранит jobs, progress и logs в Python-словарях. У каждой
линии один FIFO worker thread: `main` и `cloud` могут выполняться одновременно.
Отмена работает только до начала задания. При рестарте веба реестр теряется;
работа daemon thread не отделена от жизненного цикла сервера.

Пример: пользователь обучает GS, затем меняет конфиг и перезапускает веб.
Checkpoint может остаться на диске, но UI теряет job, progress и связь с run.
Текущая очередь сама не умеет найти этот checkpoint и продолжить обучение.
Другой пример: новая отдельная линия `gs` позволит ей работать одновременно с
SAM/RTMPose, поскольку lanes не имеют общего бюджета GPU/RAM.

Для первых скриптов это не препятствие: запуск из терминала и явный checkpoint
достаточны. Для веб-запуска нужен минимальный сохраняемый run manifest:
`run_id`, input hashes, config, state (`queued/running/interrupted/done/error`),
checkpoint и путь к логам. После рестарта следует сверять живой worker с
manifest и отличать прерванный run от завершённого. Resume должен восстанавливать
состояние обучения, а не только ставить старый `done`.

Первый веб-этап может только просматривать готовые CLI runs. Затем отдельный
worker в GS-образе выполняет обучение и рендер, а FastAPI запрашивает его статус.
Worker client создаётся в lifespan и доступен через deps. Docker socket и
загрузка torch внутри HTTP handlers для этого не нужны. GPU stages должны
делить один ресурсный бюджет; CPU-only стадии можно совмещать с GPU работой
после измерения памяти и пропускной способности.

## Предлагаемое подключение 3DGS

```text
raw RGB-D + calibration + masks
  → reconstruction → static GS background + rigid object assets
object tracks + named robot plan + virtual cameras + appearance settings
  → synthesis → RGB/depth/masks + synchronised robot states/actions
```

После успешных экспериментов выделить `viki/reconstruction/` для построения
assets и `viki/synthesis/` для композиции и генерации кадров. Общие DTO находятся
в `contracts.py`, stage API экспортируется через `__init__.py`. Не добавлять
обучение в лёгкий `render/`; команда `scene` уже обозначает SAM/object pipeline.

Первый переиспользуемый интерфейс — reader записанного RGB-D с per-view
intrinsics, pose, timestamp, depth registration и validity. Сейчас эти операции
распределены по extract/cloud/segmentation. Экспорт GS не должен зависеть от
приватного `_camera_samples` или live `CameraManager`.

Системы координат фиксируются в manifest: локальные записи используют `rig`,
план — `calibration_base`, HO-Cap — `tag_1`. `T_world_display` связывает
локальный rig с calibration base и применяется один раз. Любое центрирование
или масштабирование backend необходимо сохранить и обратить при композиции.
Camera conventions OpenCV/OpenGL переводятся в адаптере экспорта.

RGB и depth необходимо зарегистрировать; resize между разными оптическими
камерами Kinect не является регистрацией. Хранить depth units, z/ray-distance
convention, intrinsics после rectification/crop и маски неизвестных областей.

Для статической GS исключить человека и движущиеся объекты; последние строить
отдельными rigid assets. Background plate сейчас содержит depth, но не полный
RGB-набор пустой сцены. Может потребоваться дополнительная запись/scan.
Две фиксированные камеры не превращаются в дополнительные ракурсы от длины
видео. Возможность novel-view rendering на двух видах — проверяемая гипотеза.

Robot compositing требует общих камер/масштаба/часов и правильных перекрытий.
Начальная RGB/depth/alpha-композиция полезна для диагностики; ожидаемая глубина
GS не равна точному mesh z-buffer на полупрозрачных границах. Отдельно проверить
робота перед/за столом и объектом. Тени и согласованное освещение — отдельный
уровень качества, не результат самого совмещения координат.

## Путь от эксперимента к CLI и вебу

Предлагаемые экспериментальные entrypoints, ещё не реализованные:

| Скрипт | Результат и критерий |
|---|---|
| `experiments/gs_export_inputs.py` | Выбранные RGB-D/masks/cameras, manifest, reprojection overlays |
| `experiments/gs_fit_static.py` | Checkpoint, splat asset, held-out renders и метрики |
| `experiments/gs_render_robot.py` | Короткая последовательность с одним существующим plan; проверка frame/time/occlusion |

Сначала одна HO-Cap сцена. Зафиксировать тестовые камеры; сравнить, например,
обучение на шести и на двух камерах с одинаковыми двумя тестовыми. Измерять
PSNR/SSIM, depth error на валидных пикселях, время, память и render fps. Затем
добавить движущийся object asset и проверить temporal consistency. Разделять
train/test по исходным demonstrations, а не по синтетическим ракурсам одного run.

Nerfstudio/Splatfacto подходит как готовый reconstruction baseline; gsplat —
как управляемый rasterizer с RGB/depth/alpha. Входной `depth_file_path` сам по
себе не доказывает наличие depth loss в выбранном trainer. Версии PyTorch/CUDA,
gsplat и образа нужно фиксировать после smoke на доступной GPU.
Первичные источники: [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html),
[camera/depth conventions](https://docs.nerf.studio/quickstart/data_conventions.html),
[gsplat API](https://docs.gsplat.studio/main/apis/rasterization.html).

После стабилизации функций предлагаются команды `viki reconstruct` и
`viki synthesize`, вызывающие тот же Python API, что и отдельные HTTP routers.
Первый веб-просмотр использует готовые кадры/видео, input/config/metrics.
Интерактивный splat renderer в `scene3d.js` — последующий шаг; текущий Three.js
код поддерживает points/meshes, не Gaussian rendering.

```text
episode/intermediates/reconstruction/<run_id>/
  manifest.json  config.json  status.json  checkpoints/  assets/  evaluation/
data/synthetic/<dataset>/samples/<sample_id>/
  manifest.json  rgb/  depth/  masks/  trajectory.npz
```

Каждый sample ссылается на source episode, asset/run, конкретный robot plan,
camera path и appearance seed. Множитель M×K×P считает производные варианты,
а не независимые человеческие демонстрации или доказанную learning utility.

## Изменение области проекта

20 ноября удалены незавершённый аппаратный driver, HTTP route и Replay-tab,
их параметры CLI и обещание аппаратного этапа в активной документации.
`viki run` заканчивается после IK и optional object-relative. Отдельный
`viki replay` оставлен как offline check для совместимости с существующими
архивами и partial LeRobot. Его legacy поля `*_attained` содержат копии плана,
residual остаётся NaN, и успешная проверка имеет verdict `dry-run`.
MuJoCo-эксперименты и исторические отчёты сохранены.

Карта CPU/GPU, существующего параллелизма, исторические замеры и протокол
следующей проверки находятся в [вычислительном аудите](2026-11-20-compute-and-parallelism-audit.md).

## Проверка изменений

Полный `pytest tests/ -q`: **563 passed, 17 skipped**, 68.16 с, 76 warnings.
Три изменённых JS-модуля прошли `node --check`; OpenAPI не содержит
`/api/replay`, CLI `run` не предлагает `--driver`; `git diff --check` чистый.

Штатный `docker compose build test` остановился на TLS-доступе к внешнему
apt-репозиторию librealsense. Набор выполнен в отдельном CPU Docker-образе
Python 3.12 с зависимостями из `pyproject.toml` (в том числе Pinocchio 4.1.0,
OSQP 1.1.3), headless OpenCV, read-only SDK и отдельным временным `data/`.
Два теста калибровочного I/O изолированы от настоящего detector inference,
чтобы не требовать RTMPose и загрузку весов для проверки intrinsics/projector.

Это проверка Python-кода и схем, не live cameras, CUDA, SAM, MuJoCo или GS.
Для Kinect depth engine в этом тестовом окружении недоступен GL runtime.
Текст главы 3 обновлён; готовый thesis PDF в этой работе не пересобирался.
