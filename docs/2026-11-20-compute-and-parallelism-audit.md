# Вычисления и параллелизм ViKi

Дата: 20 ноября 2026. База: `dev-eval`, `135e6fb`.

ViKi использует GPU для отдельных нейросетей, CPU для геометрии/оптимизации
и браузерный WebGL для Viewer. Доступная CUDA не переносит NumPy, SciPy,
Pinocchio или OSQP на GPU. Загрузка одного ядра сама по себе не доказывает
проблему GIL: она может приходиться на последовательный алгоритм или native solver.

## Устройство вычислений по коду

| Участок | Где выполняется | Что уже распараллелено и что последовательно |
|---|---|---|
| Capture (`cameras/manager.py`, `record.py`) | CPU, SDK, USB, encode/I/O | Поток на камеру; запись камер также использует потоки. Это не GPU video encode |
| MediaPipe (`backends/mediapipe.py`) | Backend не запрашивает GPU delegate | Отдельный HandLandmarker на камеру; `extract.py` обходит камеры и кадры последовательно. Native library может иметь внутренние потоки |
| RTMPose (`backends/rtmpose.py`) | ONNX Runtime CUDA при успешном probe, иначе CPU | Backend логирует выбранное устройство. Вызовы по одному кадру; batching/prefetch общего extract отсутствуют |
| MMPose heatmap | ONNX Runtime CUDA/CPU provider | Provider выбирается по наличию; отсутствие опубликованных ONNX остаётся отдельным ограничением |
| Decode, lift, triangulation | CPU: OpenCV/NumPy/SciPy | `extract` и per-landmark robust triangulation не имеют собственного task pool |
| Prepare | CPU: NumPy/SciPy | Векторные операции и SG; native threading зависит от операции, не от числа lanes |
| Capsule hand fit (`hand_fit.py`) | CPU: SciPy least_squares/LSMR и Pinocchio | Окна через ThreadPoolExecutor, auto до 4 workers; BLAS ограничивается 1 на время пула. Короткая сцена/одно окно идёт последовательно |
| SAM2 masks (`segmentation.py`) | CUDA, bfloat16 autocast | Камеры и chunks обрабатываются последовательно; состояние видеотрекера требует порядка кадров. CPU decode, masks/overlay и запись остаются вокруг GPU |
| Semantic lift и cloud | CPU: NumPy, SDK geometry, I/O | Векторные операции на кадре; проходы по камерам/кадрам последовательные. Первая K4A ray-map строится через SDK-вызовы по пикселям, затем кэшируется |
| Object model (`object_model.py`) | CPU: cKDTree, SVD, registration | Список объектов строится последовательно; cKDTree.query явно `workers=1`. Временной tracker зависит от предыдущих оценок |
| Основной IK (`retarget/solver.py`) | CPU: Pinocchio/Pink/hpp-fcl, SciPy sparse, QP через qpsolvers | GN-итерации и per-frame linearization последовательные. Точный acceptance также CPU; стандартные вызовы не передают `workers>1` |
| Exact constraint utility | CPU processes, Unix fork | `trajectory_constraint_margins(..., workers=N)` делит кадры между процессами. Вызывается с пулом в Tesollo experiment, не во всём solver |
| Retarget batch (`scripts/run_hocap_retarget_batch.py`) | Независимые Python subprocesses | ThreadPoolExecutor управляет процессами эпизодов; каждый получает BLAS/OpenMP=1. Пул потоков здесь не означает вычисления под одним GIL |
| Tesollo/Shadow finger fit | CPU: покадровые least_squares/FK | В частности Tesollo использует предыдущую позу как initial guess и regularizer; наивное деление кадров меняет алгоритм |
| MuJoCo experiments | В существующих runners физика CPU; изображение через GL/EGL | Один rollout последовательный. GPU-render и GPU-обучение policy — разные участки; SAC pilot явно `device=cpu`, `n_envs=1` |
| Viewer (`scene3d.js`) | WebGL на клиенте | Нагрузка браузера не является CUDA-загрузкой серверного IK |
| Export | CPU + файловый I/O | Чтение/копирование NPZ/HDF5/video; отсутствие GPU здесь ожидаемо |

Источник истины для активного hand path — frozen profile в `profiles.py`.
HO-Cap batch явно выбирает MediaPipe CPU. Нельзя менять detector на RTMPose
и называть разницу чистым ускорением: это смена метода и его accuracy/coverage.

## Три независимых уровня параллелизма

1. **Jobs:** сервер сериализует `main`, а `cloud` идёт отдельной линией.
   CLI, внешние batch-скрипты и другие контейнеры эту очередь не используют.
2. **Task pools:** окна hand fit, процессы эпизодов и exact checks.
3. **Native threads:** OpenBLAS/OpenMP, OpenCV, ONNX Runtime и библиотеки моделей.

Compose задаёт BLAS/OpenMP=6 и лимит 12 CPU. Это квота CPU time, а не привязка
к 12 выделенным ядрам. `os.cpu_count()` видит логические CPU хоста и может
переоценить бюджет контейнера. Четыре процесса с шестью BLAS-потоками каждый
могут создавать oversubscription; batch-скрипт уже снижает эти переменные до 1.
Лимит 11 GiB — host RAM контейнера, не VRAM. Независимые GPU jobs могут вместе
исчерпать VRAM даже при соблюдении этого лимита.

`threadpool_limits` имеет область действия процесса. Параллельный сторонний
job в том же процессе также может попасть под изменение BLAS-лимита.
Fork utility рассчитан на изолированный CPU-процесс; включать его из
многопоточного CUDA/web процесса без проверки жизненного цикла небезопасно.

## Сохранённые замеры

Это исторические результаты на прежнем настроенном стенде, не новый benchmark
на текущем checkout. Полные входы в текущем `data/` отсутствуют.

| Измерение | Результат | Область вывода |
|---|---|---|
| Удаление повторного collision sweep и reuse данных | 53.423 → 37.270 с, −30.2% | 150 кадров, 6 iterations, pyramid; [release notes](releases/v0.0.2.md) |
| Tesollo exact check, 1 → 12 процессов | 58.84 → 6.98 с | Проверка всей 190-frame trajectory; цели/команды/margins совпали побитово |
| Тот же Tesollo run целиком | 81.95 → 29.91 с, 2.74× | Включает чтение, finger fit, IK initialization и запись; не включает perception/SAM/MuJoCo |
| Последовательный остаток в Tesollo | Около 23 с | Finger fit ~17 с, IK initialization ~5 с; не общее время основного ViKi solver |

Подробности: [Tesollo profiling](2026-10-20-tesollo-dg5f.md#время-и-вычислительные-ресурсы).
Один и тот же utility с 8 процессами дал полный run 32.62 с. Это показывает
убывающий выигрыш и накладные расходы, а не универсальный выбор 12 workers.

## Что измерять следующим

В Docker можно обернуть одну стадию в `scripts/profile_viki_stage.py`. Он
сохраняет wall time, target CPU time с завершёнными потомками, среднее число
занятых CPU, sampled process/thread count и сумму RSS процессов. Optional GPU
sampling сохраняет загрузку/VRAM всего видимого устройства и явно помечает
её как неатрибутированную процессу. Это важно при других workloads и различии
PID namespaces хоста/контейнера. Запускать каждую стадию отдельно.

```bash
docker compose run --rm --entrypoint python3 terminal \
  scripts/profile_viki_stage.py --out data/profiles/extract.json --gpu -- \
  python3 -m viki.cli extract data/datasets/<dataset>/<episode>
```

Неизменные входы и outputs, профиль, backend, hash/version модели, число кадров
и качество результата обязательны для сравнения. Отдельно считать cold start
(import, CUDA init, загрузка модели, K4A maps) и warm processing. Минимум три
повтора после прогрева; сравнивать медиану и разброс. CPU samples требуют
достаточного интервала работы, короткие процессы могут остаться между отсчётами.

Для точного распределения времени дополнительно нужны stage timers:
decode/read → preprocess → inference → registration/lift → optimisation →
encode/write. CUDA timing должен синхронизировать границы измерений либо
использовать CUDA events; время постановки kernel в очередь не равно времени
вычисления. Не синхронизировать каждый кадр в штатном throughput benchmark.

## Приоритеты оптимизации

1. Инструментировать end-to-end stages и холодный/тёплый путь. Сначала установить,
   что доминирует: inference, decode, registration, QP, mesh distance или I/O.
2. Проверить per-camera CPU detection/decode workers с отдельными экземплярами
   backend и сохранением временного порядка. Для CUDA проверить batching и
   overlap CPU preparation с inference. Размеры очередей должны быть ограничены.
3. Для object model сравнить параллельные независимые объекты с более широкими
   cKDTree queries. При малых облаках dispatch может стоить дороже поиска;
   temporal tracker нельзя произвольно разбить на независимые кадры.
4. Для hand fit сравнить 1/2/4 окна с BLAS=1 и одинаковым quality gate. Более
   короткие окна меняют результат, поэтому это не только performance knob.
5. Для IK отдельно профилировать FK/Jacobian, sparse assembly, QP и exact checks.
   Сохранить reuse collision data и полный acceptance. Исследовать persistent
   CPU pool с отдельным mutable Pinocchio Data; текущий utility создаёт pool
   на каждый вызов. Не переносить только NumPy-обвязку на GPU, оставив FCL на CPU.
6. Для batch ограничить суммарный бюджет workers × native threads и RAM,
   не просто увеличить workers. Для будущего GS резервировать GPU на весь run,
   а CPU export/preview совмещать после проверки общей памяти и I/O.

Долговечный worker — условие надёжного веб-запуска, не само по себе ускорение.
Общий distributed scheduler пока не требуется. Сначала достаточно одного GPU
worker, ограниченного CPU пула и воспроизводимого run manifest.

## Условия текущей машины

На 20 ноября: Intel i7-12650H, 16 logical CPUs, около 14 GiB RAM.
`nvidia-smi` не смог связаться с NVIDIA driver. В checkout есть только default
configuration, без исходных сцен и моделей. Поэтому таблица устройства получена
из кода, а новые scene-level CPU/GPU speedup здесь не измерены.
