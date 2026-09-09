Да. Ниже — фактическая математика текущего пути по коду, включая V3, многокамерную геометрию, позу ладони, гриппер, системы координат и конечную batch-IK.

Коротко весь тракт выглядит так:

```text
RGB + depth каждой камеры
        ↓
2D-ландмарки (u, v), score
        ↓
многовидовая триангуляция каждого сустава
        ↓
3D-скелет в системе камерного рига
        ↓
заполнение пропусков + Savitzky–Golay
        ↓
позиция pinch center + ориентация ладони/hand-fit
        ↓
rig → calibration → robot base
        ↓
фиксированный hand→EE поворот
        ↓
FK робота + адаптер + Robotiq
        ↓
совместная оптимизация q₀…qₜ
        ↓
plan.h5
```

## 0. Обозначения

Буду использовать:

- \(c\) — камера;
- \(t\) — синхронизированный кадр;
- \(l\in[0,20]\) — сустав MediaPipe;
- \(\mathbf u_{ctl}=[u,v]^T\) — 2D-координата;
- \(\mathbf X_{tl}\in\mathbb R^3\) — 3D-координата;
- \(\mathbf q_t\in\mathbb R^n\) — углы манипулятора;
- \(\mathbf R\in SO(3)\) — матрица поворота;
- \({}^{A}\mathbf T_B\) — преобразование из системы \(B\) в \(A\).

Основные системы координат:

```text
camera_c → rig → calibration_base → robot_base
                                      ↓
                            flange → adapter → gripper
```

Все 3D-позиции после чтения depth хранятся в метрах, углы IK — в радианах.

---

## 1. Что выдаёт модель для каждой камеры

Текущий профиль по умолчанию — `stable-fused-hand-v2`. Он принудительно выбирает MediaPipe, даже если в `user_configuration.json` сейчас записан RTMPose.

Для каждого кадра каждой камеры MediaPipe выдаёт 21 точку:

\[
\hat u_l,\hat v_l\in[0,1],\qquad \hat z_l\in\mathbb R.
\]

Они переводятся в пиксели:

\[
u_l=\hat u_lW,\qquad v_l=\hat v_lH.
\]

\(\hat z_l\) — относительная глубина MediaPipe, не метры.

Порог `0.5` у MediaPipe используется внутри графа как:

- минимальная уверенность детекции руки;
- минимальная уверенность присутствия;
- минимальная уверенность трекинга.

Это не отбрасывание отдельных суставов. Либо MediaPipe принимает руку и отдаёт все 21 точки, либо весь кадр руки отсутствует.

Ещё одна важная деталь: `HandDetection.confidence` у MediaPipe сейчас является score классификации handedness, то есть насколько модель уверена, что рука правая/левая. Это не измеренная точность каждой отдельной точки. Поскольку MediaPipe не даёт нам per-joint score, этот один score размножается на все 21 точки.

Если используется RTMPose, он, напротив, возвращает отдельные \(s_l\) для каждого сустава, но текущий V3 его не использует.

---

## 2. Сырые многокамерные наблюдения

Для каждой принятой детекции сохраняется:

\[
(\mathbf u_{ctl},s_{ctl},d_{ctl},\sigma^{depth}_{ctl}).
\]

Глубина берётся не обязательно из одного пикселя. Цветовая точка сначала проецируется в depth image, затем вокруг неё берётся окно радиуса 15 px:

\[
d_{ctl}=\operatorname{median}\{D(u,v)\mid (u,v)\in ROI,\ D(u,v)>0\}.
\]

Разброс:

\[
\sigma^{depth}_{ctl}=\operatorname{std}\{D(u,v)\mid(u,v)\in ROI\}.
\]

Depth считается валидным, если:

- валиден центральный пиксель; или
- валидно более 25% окна.

Для Kinect переход color→depth выполняет SDK-проектор. Для RealSense с уже выровненной глубиной это тождественное отображение.

Камеры объединяются по `frame_index`, а не по их индивидуальным timestamp. То есть кадр `i` каждой камеры считается одной синхронной группой. Timestamp камеры используется для `rec.npz`, но сама стереопара собирается по номеру кадра.

---

# Основной путь V3: многовидовая триангуляция

## 3. Геометрия камеры

Из калибровки имеется:

\[
\mathbf p_c=\mathbf R_{cw}\mathbf p_{rig}+\mathbf t_{cw}.
\]

В файле хранится обратная матрица:

\[
{}^{rig}\mathbf T_c=
\begin{bmatrix}
\mathbf R_{cw}^T&-\mathbf R_{cw}^T\mathbf t_{cw}\\
0&1
\end{bmatrix}.
\]

Для триангуляции она снова инвертируется, после чего строится projection matrix:

\[
\mathbf P_c=\mathbf K_c
\begin{bmatrix}
\mathbf R_{cw}&\mathbf t_{cw}
\end{bmatrix}.
\]

Сырые пиксели один раз исправляются от дисторсии:

\[
\tilde{\mathbf u}_{ctl}
=
\operatorname{undistort}
(\mathbf u_{ctl},\mathbf K_c,\mathbf d_c).
\]

После этого везде используется обычная pinhole-проекция без повторного применения дисторсии.

---

## 4. Отбор камер для сустава

Для каждого \(t,l\) берутся камеры, где:

\[
s_{ctl}\ge 0.3
\]

и координата \(\mathbf u_{ctl}\) конечна.

Нужно минимум две камеры. Иначе сустав остаётся пропуском:

\[
\mathbf X_{tl}=\mathrm{NaN},\qquad quality_{tl}=0.
\]

Для MediaPipe \(s_{ctl}\) фактически является размноженным score handedness.

---

## 5. Начальная DLT-триангуляция

Перебираются все пары доступных камер \(a,b\). Для каждой строится система:

\[
\mathbf A=
\begin{bmatrix}
u_a\mathbf P_{a,3}-\mathbf P_{a,1}\\
v_a\mathbf P_{a,3}-\mathbf P_{a,2}\\
u_b\mathbf P_{b,3}-\mathbf P_{b,1}\\
v_b\mathbf P_{b,3}-\mathbf P_{b,2}
\end{bmatrix}.
\]

Ищется однородное решение:

\[
\mathbf A\tilde{\mathbf X}=0.
\]

Оно берётся как последний правый сингулярный вектор SVD:

\[
\tilde{\mathbf X}=\mathbf V_{last},\qquad
\mathbf X=\frac{\tilde{\mathbf X}_{1:3}}{\tilde X_4}.
\]

После этого проверяется cheirality:

\[
z_a(\mathbf X)>0,\qquad z_b(\mathbf X)>0.
\]

---

## 6. Угол между лучами

Для центров камер \(\mathbf C_a,\mathbf C_b\):

\[
\theta_{ab}
=
\arccos
\frac{
(\mathbf X-\mathbf C_a)^T(\mathbf X-\mathbf C_b)
}{
\|\mathbf X-\mathbf C_a\|
\|\mathbf X-\mathbf C_b\|
}.
\]

Пара отбрасывается, если:

\[
\theta_{ab}<5^\circ.
\]

Это защищает от случая, когда лучи почти параллельны и небольшая пиксельная ошибка превращается в огромную ошибку по глубине.

---

## 7. Выбор лучшей гипотезы

Для каждой гипотезы вычисляется reprojection error во всех доступных камерах:

\[
e_c=
\left\|
\pi_c(\mathbf X)-\tilde{\mathbf u}_c
\right\|_2.
\]

Камера считается inlier, если:

\[
e_c\le4\text{ px},\qquad z_c(\mathbf X)>0.
\]

Гипотезы сравниваются лексикографически по:

1. числу inlier-камер;
2. сумме их scores;
3. отрицательной медиане reprojection error.

То есть сначала выбираем максимум согласующихся камер, затем наиболее уверенные наблюдения, затем наименьшую ошибку.

---

## 8. Нелинейное уточнение 3D-сустава

Выбранная точка уточняется `least_squares`.

### Пиксельная часть

Для камеры \(c\):

\[
\mathbf r^{uv}_c
=
\sqrt{s_c}
\left(
\pi_c(\mathbf X)-\tilde{\mathbf u}_c
\right).
\]

### Depth-часть

Depth видит поверхность кожи, а нам нужен приблизительный центр сустава. Поэтому используется поправка \(\delta_l\):

\[
r^d_c
=
\sqrt{\lambda_d w^d_c}
\frac{f_c}{z_c}
\left(
z_c(\mathbf X)-d_c-\delta_l
\right).
\]

Где:

\[
\lambda_d=0.1,
\]

\[
w^d_c
=
s_c
\exp\left(
-\frac{\sigma^{depth}_c}{0.02}
\right),
\]

\[
f_c=\frac{f_{x,c}+f_{y,c}}{2}.
\]

Множитель \(f/z\) переводит ошибку глубины в приблизительный пиксельный масштаб, чтобы её можно было складывать с reprojection residual.

Базовая поправка:

\[
\delta=0.01\text{ м}.
\]

Она масштабируется по типу landmark:

- fingertip: \(1.0\delta\);
- суставы пальцев: \(1.6\delta\);
- wrist: \(2.0\delta\);
- остальные: \(1.3\delta\).

Функционал:

\[
\min_{\mathbf X}
\sum_c
\rho\left((r^u_c)^2\right)
+
\rho\left((r^v_c)^2\right)
+
\rho\left((r^d_c)^2\right).
\]

Используется `soft_l1` с характерным масштабом 4 px.

---

## 9. Геометрическое качество 3D-точки

После оптимизации:

\[
q_{tl}
=
\frac{N_{inlier}}{N_{usable}}
\cdot
\operatorname{clip}
\left(
1-\frac{\bar e}{2\cdot4},
0,1
\right)
\cdot
\operatorname{clip}
\left(
\frac{\theta_{max}}{20^\circ},
0,1
\right).
\]

То есть quality падает, если:

- часть камер не согласилась;
- reprojection error близка к 8 px;
- лучи камер имеют слабый baseline.

В текущем V3 detector score дополнительно в `quality` не умножается. Он участвует в отборе камер и нелинейном уточнении, но сама итоговая quality — в основном геометрическая эвристика. Это не дисперсия и не статистически откалиброванная вероятность.

Результат:

\[
\mathbf X^{rig}_{tl},\quad q_{tl}
\]

записывается в `raw/joints3d.npz`.

---

# Параллельный монокулярный путь

## 10. Зачем всё ещё существует `rec.npz`

Одновременно каждая камера строит свою приближённую 3D-руку. Для V3 эта траектория не является основной финальной геометрией, но сохраняется для:

- A/B-сравнения;
- старого `xyz_mean`;
- диагностики;
- обратной совместимости.

Если у landmark есть измеренная глубина \(z_l\), обычная deprojection:

\[
X_l=\frac{(u_l-c_x)z_l}{f_x},
\qquad
Y_l=\frac{(v_l-c_y)z_l}{f_y},
\qquad
Z_l=z_l.
\]

Но окончательная форма руки строится иначе: берётся медианная глубина ладонных узлов \(z_d\), после чего относительный MediaPipe \(z\) масштабируется:

\[
s=z_d\frac{W}{f_x},
\]

\[
X_l^{mp}=\frac{(u_l-c_x)z_d}{f_x},
\]

\[
Y_l^{mp}=\frac{(v_l-c_y)z_d}{f_y},
\]

\[
Z_l^{mp}=z_d+\hat z_l s.
\]

Затем вся модель переносится так, чтобы медиана ладонных landmark совпала с медианой реально измеренных depth-точек:

\[
\Delta\mathbf p
=
\mathbf p_{hand}^{depth}
-
\operatorname{median}_{l\in palm}
\mathbf p_l^{mp},
\]

\[
\mathbf p_l=\mathbf p_l^{mp}+\Delta\mathbf p.
\]

Вес камеры для каждой точки:

\[
w_{ctl}
=
visibility_{ctl}
\cdot sensor_{ctl}
\cdot\frac{1}{d_{ctl}^2}
\cdot|\cos\theta_{ctl}|.
\]

Здесь:

- `sensor = 1`, если есть реальная глубина;
- `sensor = 0.1`, если точка поставлена только через относительную форму;
- \(d=\|\mathbf p_c\|\) — расстояние до камеры;
- \(|\cos\theta|\) — угол между лучом и нормалью depth-поверхности.

Нормаль оценивается конечными разностями:

\[
\mathbf t_x=
\begin{bmatrix}
z/f_x\\
0\\
(z_{u+1}-z_{u-1})/2
\end{bmatrix},
\qquad
\mathbf t_y=
\begin{bmatrix}
0\\
z/f_y\\
(z_{v+1}-z_{v-1})/2
\end{bmatrix},
\]

\[
\mathbf n=\mathbf t_x\times\mathbf t_y.
\]

Затем камера→rig:

\[
\mathbf p^{rig}
=
\mathbf R_{cw}^{T}
(\mathbf p^c-\mathbf t_{cw}).
\]

---

# Prepare: единая временная траектория

## 11. Сетка времени

Основная сетка берётся из `raw/timestamps.json`, то есть включает все синхронные кадры, даже если в каком-то кадре руку не увидела ни одна камера.

Триангулированные точки ставятся на ближайший timestamp в пределах половины периода кадров. Незаполненные места остаются NaN.

В `xyz_mean`-режиме вместо триангуляции было бы:

\[
\mathbf X_{tl}^{fused}
=
\frac{
\sum_c w_{ctl}\mathbf X_{ctl}
}{
\sum_c w_{ctl}
}.
\]

Но текущий стабильный V2 использует `joints3d.npz`, а не это среднее.

---

## 12. Заполнение пропусков

Стабильный V2 и кандидат `fused-hand-no-extrap-v1` для каждого landmark и каждой
координаты \(x,y,z\) используют линейную интерполяцию между ближайшими
наблюдениями:

\[
\mathbf X(t)=(1-\alpha)\mathbf X(t_a)+\alpha\mathbf X(t_b),
\qquad
\alpha=\frac{t-t_a}{t_b-t_a}.
\]

На краях стабильный V2 удерживает ближайшее наблюдение — это контрольный путь.
`fused-hand-no-extrap-v1` оставляет там `NaN`: без двух границ интерполяция
математически не определена. Конечный articulated/robot state может удерживать
ближайшую позу как model prediction с нулевым весом данных, но она не записывается
как измеренный landmark. Исторический V1 и custom-путь сохраняют natural cubic
spline:

\[
S_i(\tau)=a_i+b_i\tau+c_i\tau^2+d_i\tau^3,
\]

с условиями:

- совпадение с известными точками;
- непрерывность \(S,S',S''\);
- \(S''\) на концах равна нулю.

В cubic-пути, если валидных точек меньше четырёх:

- 2–3 точки → линейная интерполяция;
- одна точка → постоянное значение.

В текущих профилях `max_gap=0`, поэтому заполняются пропуски любой длины.

Оба метода покоординатные и не являются интерполяцией конфигурации руки или
SE(3). Linear исключает cubic overshoot, обнаруженный на двухпальцевом эпизоде,
но сам по себе не сохраняет геометрию кисти при асинхронной видимости пальцев.

---

## 13. Сглаживание

После заполнения применяется Savitzky–Golay с окном 7 и полиномом степени 2.

Локально для каждой координаты решается:

\[
\min_{a_0,a_1,a_2}
\sum_{k=-3}^{3}
\left(
x_{t+k}-(a_0+a_1k+a_2k^2)
\right)^2,
\]

а сглаженным значением становится:

\[
\hat x_t=a_0.
\]

Это выполняется отдельно для каждого landmark и координаты.

---

# Поза руки и уверенность

## 14. Базовая ориентация ладони

По четырём точкам:

- wrist;
- index MCP;
- middle MCP;
- pinky MCP.

Строятся:

\[
\mathbf f
=
\mathbf p_{middle}-\mathbf p_{wrist},
\]

\[
\mathbf s
=
\mathbf p_{pinky}-\mathbf p_{index}.
\]

Оси ладони:

\[
\mathbf x_p=\frac{\mathbf f}{\|\mathbf f\|},
\]

\[
\mathbf z_p=
\frac{\mathbf f\times\mathbf s}
{\|\mathbf f\times\mathbf s\|},
\]

\[
\mathbf y_p=
\frac{\mathbf z_p\times\mathbf x_p}
{\|\mathbf z_p\times\mathbf x_p\|}.
\]

Матрица:

\[
\mathbf R^{rig}_{palm}
=
\begin{bmatrix}
\mathbf x_p&\mathbf y_p&\mathbf z_p
\end{bmatrix}.
\]

Базовая позиция pose — wrist:

\[
\mathbf p^{rig}_{hand}=\mathbf p^{rig}_{wrist}.
\]

Если полный palm frame не построился, используется центр доступных ладонных точек и единичная ориентация. Но такой кадр обычно получает нулевой либо низкий вес дальше.

---

## 15. Проверка стабильности ориентации

Дополнительно проверяется геометрия:

\[
L_{thumb}=\|\mathbf p_{thumbCMC}-\mathbf p_{wrist}\|,
\]

\[
L_{middle}=\|\mathbf p_{middleMCP}-\mathbf p_{wrist}\|,
\]

\[
\sin\alpha=
\frac{
\|\mathbf v_{middle}\times\mathbf v_{thumb}\|
}{
L_{middle}L_{thumb}
}.
\]

Допустимо:

\[
0.5\tilde L_{thumb}
\le L_{thumb}\le
2.0\tilde L_{thumb},
\]

\[
0.5\tilde L_{middle}
\le L_{middle}\le
1.75\tilde L_{middle},
\]

\[
\sin\alpha\ge0.25.
\]

Межкадровый поворот:

\[
\Delta\mathbf R_t
=
\mathbf R_t^T\mathbf R_{t+1},
\]

\[
\theta_t=
\arccos
\frac{\operatorname{tr}(\Delta\mathbf R_t)-1}{2}.
\]

Если \(\theta_t>60^\circ\), оба соседних кадра помечаются невалидными.

---

## 16. Confidence V3

Для триангуляции per-landmark confidence:

\[
c_{tl}=\operatorname{clip}(q_{tl},0,1).
\]

Если координата была интерполирована, а не непосредственно получена из камер:

\[
c_{tl}=0.
\]

Покадровая уверенность pose считается по:

- wrist;
- index MCP;
- middle MCP;
- pinky MCP.

\[
\omega_t=
\frac{
c_{t,wrist}
+c_{t,index}
+c_{t,middle}
+c_{t,pinky}
}{4}.
\]

Но V3 требует, чтобы все четыре точки были непосредственно наблюдаемыми. Если хотя бы одна была заполнена сплайном:

\[
\omega_t=0.
\]

Затем:

\[
\omega_t
=
\operatorname{clip}(\omega_t,0,1)^\alpha
\cdot valid_t,
\qquad \alpha=1.
\]

---

# Антропоморфная модель руки

## 17. Что V3 добавляет поверх чистого скелета

V3 запускает `articulated-landmarks-v1`. Это отдельная кинематическая модель человеческой руки:

- free-flyer wrist: 6 DoF, но 7 configuration coordinates из-за quaternion;
- по 4 вращательных сустава на палец;
- всего \(n_v=26\), \(n_q=27\).

Геометрия пальцев калибруется по эпизоду. Длины фаланг:

\[
L_{finger,k}
=
\operatorname{median}_{t\in calibration}
\|\mathbf p_{t,k+1}-\mathbf p_{t,k}\|.
\]

Радиус capsule:

\[
r_k=
\operatorname{clip}(0.35L_k,\,0.006,\,0.014).
\]

Для palm используется одна широкая capsule.

Из landmark строится начальная конфигурация:

- abduction:

\[
q_{abd}=\operatorname{atan2}(d_y,d_x);
\]

- flexion MCP — signed angle между rest-направлением и первой фалангой;
- PIP и DIP — абсолютный угол между соседними фалангами.

Signed angle:

\[
\theta
=
\operatorname{sign}
\left(
\mathbf a^T(\mathbf u\times\mathbf v)
\right)
\arccos
\frac{\mathbf u^T\mathbf v}{\|\mathbf u\|\|\mathbf v\|}.
\]

Затем вся траектория руки оптимизируется примерно как:

\[
E_{hand}
=
\sum_{t,l}
750\,c_{tl}
\|
FK_l(\mathbf q_t)-\mathbf X_{tl}
\|^2
+
E_{vel}
+
E_{acc}
+
0.0005E_{posture}.
\]

Временные веса разделены на translation, rotation и finger joints:

\[
w_v=(40,\ 30,\ 3),
\qquad
w_a=(120,\ 150,\ 15).
\]

Оптимизация идёт окнами по 120 кадров с перекрытием 30. Перекрытия смешиваются на manifold через `Pinocchio.interpolate`.

После оптимизации wrist translation принудительно возвращается ровно к чистому wrist:

\[
\mathbf p^{fit}_{wrist,t}
=
\mathbf p^{clean}_{wrist,t}.
\]

Результат не заменяет исходный скелет, а добавляется как `hand_fit_*` overlay.

---

# Что именно сейчас передаётся в retarget

## 18. Текущая комбинация фактически гибридная

Текущие настройки:

```text
pose_source = hand_fit
position_anchor = pinch_center
```

Поэтому:

- ориентация берётся из `hand_fit_rotations`;
- позиция wrist/hand-fit загружается, но затем заменяется pinch center из чистого сглаженного скелета.

Позиция цели:

\[
\mathbf p^{rig}_{target,t}
=
\frac{
\mathbf p^{rig}_{thumbTip,t}
+
\mathbf p^{rig}_{indexTip,t}
}{2}.
\]

То есть робот ведёт середину между большим и указательным пальцем человека к середине контактных панелей Robotiq.

Confidence этой позиции:

\[
c^{pinch}_t
=
\min(c_{thumbTip,t},c_{indexTip,t}),
\]

\[
\omega^{IK}_t
=
valid_t
\cdot
\min(\omega_t,c^{pinch}_t).
\]

Невалидные позиции временно линейно интерполируются, ориентации — через quaternion SLERP. Но это только конечные числовые placeholders:

\[
\omega^{IK}_t=0
\]

для невалидного кадра, поэтому он не притягивает робота к выдуманной цели. Такие кадры определяются временными регуляризаторами.

---

# Системы координат

## 19. Rig → calibration base

Из `world_anchor.json` берётся:

\[
{}^{cal}\mathbf T_{rig}
=
\begin{bmatrix}
\mathbf R_{cr}&\mathbf t_{cr}\\
0&1
\end{bmatrix}.
\]

Тогда:

\[
\mathbf p^{cal}_t
=
\mathbf R_{cr}\mathbf p^{rig}_t+\mathbf t_{cr},
\]

\[
\mathbf R^{cal}_t
=
\mathbf R_{cr}\mathbf R^{rig}_t.
\]

Если anchor отсутствует, используется identity.

Объектноцентричной математики сейчас нет:

\[
{}^{obj}\mathbf T_{hand}
=
({}^{world}\mathbf T_{obj})^{-1}
{}^{world}\mathbf T_{hand}
\]

в коде предусмотрена, но pose объекта передаётся как `None`, поэтому результат не создаётся.

---

## 20. Фиксированное hand → robot EE

Пользователь задаёт локальное преобразование:

\[
{}^{hand}\mathbf T_{EE}
=
\begin{bmatrix}
\mathbf R_{HE}&\mathbf t_{HE}\\
0&1
\end{bmatrix}.
\]

Тогда:

\[
\mathbf p^{cal}_{EE,t}
=
\mathbf p^{cal}_{hand,t}
+
\mathbf R^{cal}_{hand,t}\mathbf t_{HE},
\]

\[
\mathbf R^{cal}_{EE,t}
=
\mathbf R^{cal}_{hand,t}\mathbf R_{HE}.
\]

Сейчас:

\[
\mathbf t_{HE}=[0,0,0]^T,
\]

а extrinsic XYZ RPY:

\[
[-175.675^\circ,\ 15.45^\circ,\ -47.922^\circ].
\]

Это постоянная коррекция между ориентацией человеческой ладони и желаемой ориентацией физического гриппера.

---

## 21. Calibration → robot base

Поза базы робота в calibration frame:

\[
{}^{cal}\mathbf T_{robot}
=
\begin{bmatrix}
\mathbf R_b&\mathbf b\\
0&1
\end{bmatrix}.
\]

Тогда цели IK:

\[
\mathbf p^{robot}_{target,t}
=
\mathbf R_b^T
(\mathbf p^{cal}_{target,t}-\mathbf b),
\]

\[
\mathbf R^{robot}_{target,t}
=
\mathbf R_b^T\mathbf R^{cal}_{target,t}.
\]

Сейчас в конфигурации:

\[
\mathbf b=[0,0,0],\qquad RPY_b=[0,0,0].
\]

Настройки фронта могут передать другие значения на конкретный запуск.

---

# Гриппер, адаптер и FK

## 22. Семантическое раскрытие руки

Сначала вычисляется нормированный pinch ratio:

\[
d_t
=
\frac{
\|\mathbf p_{thumbTip,t}-\mathbf p_{indexTip,t}\|
}{
\|\mathbf p_{wrist,t}-\mathbf p_{middleMCP,t}\|
}.
\]

Он отображается в opening:

\[
o_t^{raw}
=
\operatorname{clip}
\left(
\frac{d_t-0.35}{1.20-0.35},
0,1
\right).
\]

Затем EMA:

\[
o_t
=
o_{t-1}
+
0.35(o_t^{raw}-o_{t-1}).
\]

Где:

- \(o=0\) — закрыт;
- \(o=1\) — полностью открыт.

Если нужные landmark отсутствуют, удерживается предыдущее значение.

---

## 23. Ограничение скорости Robotiq

Для Robotiq 2F-85:

\[
W_{max}=0.085\text{ м},
\qquad
V_{max}=0.150\text{ м/с}.
\]

Максимальное изменение нормированной команды за кадр:

\[
\Delta o_{max}
=
\frac{V_{max}\Delta t}{W_{max}}.
\]

Дальше:

\[
o_t^{profiled}
=
o_{t-1}^{profiled}
+
\operatorname{clip}
(o_t-o_{t-1}^{profiled},
-\Delta o_{max},
\Delta o_{max}).
\]

Физическая ширина:

\[
W_t=0.085o_t.
\]

Master joint Robotiq:

\[
q^g_t
=
q_{closed}
+
o_t(q_{open}-q_{closed}),
\]

\[
q^g_t=0.8+o_t(0-0.8)=0.8(1-o_t).
\]

Этот сустав не оптимизируется IK. Он передаётся как известный passive joint для каждого кадра, а mimic-механизм URDF двигает остальные пальцы.

---

## 24. Адаптер

Гриппер присоединяется:

\[
{}^{flange}\mathbf T_{gripper}
=
{}^{flange}\mathbf T_{adapter}
{}^{adapter}\mathbf T_{URDF}.
\]

Сейчас adapter:

\[
\mathbf t_A=[0,0,0.011]\text{ м},
\qquad
RPY_A=[0,0,0].
\]

Цилиндрическая шайба — не только визуализация. Та же трансформация участвует в Pinocchio FK, а цилиндр участвует в floor/collision geometry.

---

## 25. Какой именно point гриппера отслеживает IK

Ориентация берётся из URDF-фрейма:

```text
robotiq_arg2f_tcp
```

Но позиция — не конец TCP. Для левой и правой distal phalanx вычисляются центры контактных панелей:

\[
\mathbf p_i(\mathbf q,q^g_t)
=
\mathbf t_i+\mathbf R_i\mathbf o_{pad},
\]

где:

\[
\mathbf o_{pad}
=
[0,\ -0.0220203,\ 0.03242]^T.
\]

Позиция task point:

\[
\mathbf p_{grasp}
=
\frac{\mathbf p_{left}+\mathbf p_{right}}{2}.
\]

Именно поэтому цель человека `pinch_center` сопоставляется с серединой панелей Robotiq.

Ориентация и позиция здесь намеренно происходят из разных геометрических объектов:

- position — середина движущихся панелей;
- rotation — ориентация TCP.

---

# Конечная кинематика манипулятора

## 26. Forward kinematics

Для заданных углов манипулятора и известного состояния гриппера:

\[
{}^0\mathbf T_{EE}
=
\prod_i
{}^{i-1}\mathbf T_i(q_i)
\cdot
{}^{flange}\mathbf T_{adapter}
\cdot
{}^{adapter}\mathbf T_{gripper}(q^g_t).
\]

Pinocchio вычисляет:

\[
\mathbf p(\mathbf q_t,q^g_t),
\qquad
\mathbf R(\mathbf q_t,q^g_t).
\]

Для UR10 оптимизируются только шесть arm joint:

\[
\mathbf q_t\in\mathbb R^6.
\]

Гриппер известен и подставляется отдельно.

Reference pose внутри основной оптимизации:

\[
\mathbf q_{ref}=\mathbf 0.
\]

Это не approach home. Home используется позже и равен:

\[
[0,-\pi/2,0,-\pi/2,0,0].
\]

---

## 27. Ошибка позы

Позиционная ошибка:

\[
\mathbf e^p_t
=
\sqrt{w_p}
\left(
\mathbf p(\mathbf q_t)-\mathbf p^*_t
\right).
\]

Ориентационная:

\[
\boldsymbol\phi_t
=
\operatorname{Log}
\left(
(\mathbf R^*_t)^T
\mathbf R(\mathbf q_t)
\right),
\]

\[
\mathbf e^R_t
=
\sqrt{w_R}\boldsymbol\phi_t.
\]

Общий residual:

\[
\mathbf e_t=
\begin{bmatrix}
\mathbf e^p_t\\
\mathbf e^R_t
\end{bmatrix}.
\]

Текущие веса:

\[
w_p=1,
\qquad
w_R=0.01.
\]

Следовательно один радиан orientation error входит в общую норму как \(0.1\), то есть ориентация сейчас существенно слабее позиции.

---

## 28. Полный функционал batch IK

Все кадры оптимизируются одновременно:

\[
\mathbf Q=
\begin{bmatrix}
\mathbf q_0\\
\mathbf q_1\\
\vdots\\
\mathbf q_{T-1}
\end{bmatrix}.
\]

Основной функционал:

\[
J(\mathbf Q)
=
\sum_t
\omega_t
\rho_\delta(\|\mathbf e_t\|)
+
w_v\|D_1\mathbf Q\|^2
+
w_a\|D_2\mathbf Q\|^2
+
w_{post}\|\mathbf Q-\mathbf Q_{ref}\|^2.
\]

Где:

\[
D_1\mathbf q_t
=
\frac{\mathbf q_{t+1}-\mathbf q_t}{\Delta t},
\]

\[
D_2\mathbf q_t
=
\frac{
\mathbf q_{t+2}-2\mathbf q_{t+1}+\mathbf q_t
}{\Delta t^2}.
\]

Текущие коэффициенты:

\[
w_v=0.002,
\]

\[
w_a=0.00002,
\]

\[
w_{post}=0.0001.
\]

Huber:

\[
\rho_\delta(x)=
\begin{cases}
\frac12x^2,&x\le\delta,\\
\delta(x-\frac12\delta),&x>\delta,
\end{cases}
\]

\[
\delta=0.05.
\]

Поскольку Huber применяется к совместной норме \([\text{метры},\sqrt{0.01}\text{ радианы}]\), breakpoint приблизительно соответствует:

- 50 mm чистой позиционной ошибки;
- \(0.5\) rad \(=28.6^\circ\) чистой ориентационной ошибки.

Это единый SE(3)-outlier, а не два независимых Huber.

---

## 29. Confidence floor

Для реального ненулевого наблюдения:

\[
\tilde\omega_t
=
\max(\omega_t,0.05).
\]

Для отсутствующего наблюдения:

\[
\omega_t=0
\Rightarrow
\tilde\omega_t=0.
\]

То есть confidence floor не оживляет интерполированные кадры. Он лишь не даёт слабому, но реальному наблюдению стать полностью незначимым.

---

## 30. Линеаризация

На каждой итерации:

\[
\mathbf e_t(\mathbf q_t+\Delta\mathbf q_t)
\approx
\mathbf e_t+\mathbf J_t\Delta\mathbf q_t.
\]

Позиционный Jacobian:

\[
\mathbf J^p_t
=
\sqrt{w_p}\mathbf J_{position}.
\]

Для контактной точки со смещением \(\mathbf r\):

\[
\mathbf J_{point}
=
\mathbf J_v-[\mathbf r]_\times\mathbf J_\omega.
\]

Для двух панелей Robotiq:

\[
\mathbf J_{grasp}
=
\frac{\mathbf J_{left}+\mathbf J_{right}}{2}.
\]

Ориентационный Jacobian:

\[
\mathbf J^R_t
=
\sqrt{w_R}
\mathbf J_l^{-1}(\boldsymbol\phi_t)
(\mathbf R_t^*)^T
\mathbf J_{\omega,t}.
\]

Обратный левый Jacobian SO(3):

\[
\mathbf J_l^{-1}(\boldsymbol\phi)
=
\mathbf I
-\frac12[\boldsymbol\phi]_\times
+A[\boldsymbol\phi]_\times^2,
\]

\[
A=
\frac{
1-\frac12\theta\cot(\theta/2)
}{\theta^2},
\qquad
\theta=\|\boldsymbol\phi\|.
\]

При малом \(\theta\):

\[
A\approx\frac1{12}+\frac{\theta^2}{720}.
\]

---

## 31. IRLS для Huber

Для \(n_t=\|\mathbf e_t\|\):

\[
\gamma_t=
\begin{cases}
1,&n_t\le\delta,\\
\sqrt{\delta/n_t},&n_t>\delta.
\end{cases}
\]

Строки pose residual и Jacobian умножаются на:

\[
\sqrt{\omega_t}\gamma_t.
\]

Так большой выброс продолжает влиять, но уже линейно, а не квадратично.

---

## 32. QP одной итерации

Собирается sparse-матрица из:

- block-diagonal pose Jacobian;
- \(D_1\);
- \(D_2\);
- posture identity;
- Tikhonov damping.

Локальная задача:

\[
\min_{\Delta\mathbf Q}
\frac12\|
\mathbf A\Delta\mathbf Q+\mathbf r
\|^2.
\]

Дополнительный damping:

\[
\lambda_{damp}\|\Delta\mathbf Q\|^2,
\qquad
\lambda_{damp}=10^{-5}.
\]

Решает OSQP через `qpsolvers`.

Максимальный шаг одного joint за итерацию:

\[
|\Delta q_{tj}|\le0.2\text{ rad}.
\]

Сходимость:

\[
\max_{t,j}|\Delta q_{tj}|<0.001\text{ rad},
\]

либо максимум 40 итераций.

Это фиксированно демпфированный Gauss–Newton/SQP, не адаптивный Levenberg–Marquardt.

---

# Жёсткие ограничения

## 33. Joint limits

\[
q_j^{min}
\le
q_{tj}+\Delta q_{tj}
\le
q_j^{max}.
\]

Для UR используются явные лимиты:

- большинство осей: \([ -2\pi,2\pi]\);
- elbow: \([-\pi,\pi]\).

---

## 34. Ограничение скорости

\[
-v_j^{max}
\le
\frac{
q_{t+1}+\Delta q_{t+1}
-q_t-\Delta q_t
}{\Delta t}
\le
v_j^{max}.
\]

Это жёсткое QP-ограничение, а не только штраф.

Acceleration отдельного жёсткого ограничения сейчас не имеет — только мягкий член функционала.

---

## 35. Пол для всей геометрии

Пол задаётся в calibration frame:

\[
z_{cal}\ge0.
\]

В robot frame нормаль:

\[
\mathbf n_r=\mathbf R_b^T
\begin{bmatrix}
0\\0\\1
\end{bmatrix},
\]

offset:

\[
d_r=-b_z.
\]

Для каждой подвижной collision geometry находится самая нижняя support point:

\[
\mathbf p_{low}
=
\arg\min_{\mathbf p\in geometry}
\mathbf n_r^T\mathbf p.
\]

Margin:

\[
m_{floor}
=
\mathbf n_r^T\mathbf p_{low}-d_r.
\]

Требование:

\[
m_{floor}\ge0.
\]

В ограничение входят:

- все подвижные collision meshes робота;
- цилиндр адаптера;
- геометрия гриппера;
- обе контактные точки.

Не входит только неподвижный pedestal с `parentJoint == 0`.

Линеаризация:

\[
m(\mathbf q+\Delta\mathbf q)
\approx
m(\mathbf q)
+
\mathbf J_m\Delta\mathbf q
\ge0.
\]

Где:

\[
\mathbf J_m
=
\mathbf n_r^T
\left(
\mathbf J_v-[\mathbf r]_\times\mathbf J_\omega
\right).
\]

---

## 36. Self-collision

После исключения:

- геометрий одного joint;
- соседних звеньев;
- пар, находящихся в штатном контакте в neutral pose,

строится барьер Pink:

\[
m_{col,k}(\mathbf q)\ge0.
\]

Минимальная разрешённая дистанция:

\[
d_{min}=0.02\text{ м}.
\]

В текущей настройке используется до восьми collision pairs.

Линеаризация:

\[
m_{col,k}
+
\mathbf J_{col,k}\Delta\mathbf q
\ge0.
\]

---

## 37. Проверка нелинейного шага

QP использует линейные приближения, но FK, пол и collision нелинейны. Поэтому после решения пробуется:

\[
\mathbf Q_{candidate}
=
\mathbf Q+\alpha\Delta\mathbf Q,
\]

где:

\[
\alpha=1,\frac12,\frac14,\ldots
\]

Шаг принимается, только если:

- уже достигнутая floor-feasibility не потеряна;
- уже достигнутая collision-feasibility не потеряна;
- точный robust objective не вырос.

Если подходящий шаг не найден до \(10^{-6}\), шаг обнуляется.

---

# Финальный approach и артефакт

## 38. Подъезд из home

Перед демонстрационной траекторией строится quintic approach:

\[
q(s)
=
q_{home}
+c_3s^3+c_4s^4+c_5s^5,
\qquad s\in[0,1).
\]

Коэффициенты выбираются так, чтобы:

- в начале были \(q_{home}\), нулевая скорость и ускорение;
- в конце совпадали позиция, скорость и ускорение начала оптимизированной траектории.

\[
c_3=10\Delta q-4Tv_1+\frac12T^2a_1,
\]

\[
c_4=-15\Delta q+7Tv_1-T^2a_1,
\]

\[
c_5=6\Delta q-3Tv_1+\frac12T^2a_1.
\]

Если approach нарушает velocity limits, его длительность увеличивается. После этого целиком проверяется:

```text
approach + demonstration
```

на:

- joint limits;
- velocity limits;
- пол;
- collision.

Только после прохождения проверки пишется `plan.h5`.

---

## Что в итоге физически означает один кадр

Для каждого кадра \(t\) система решает примерно такую задачу:

> Найти шесть углов UR10, чтобы середина движущихся контактных панелей Robotiq оказалась в середине между большим и указательным пальцем человека, а TCP Robotiq имел ориентацию антропоморфно восстановленной ладони с постоянной hand→EE поправкой. При этом вся геометрия робота, адаптера и гриппера должна оставаться над полом, робот не должен сталкиваться сам с собой, превышать joint/velocity limits и должен двигаться достаточно плавно относительно соседних кадров.

## Математически важные оговорки

- V3 confidence не является вероятностью или covariance. Это геометрический score.
- MediaPipe confidence сейчас — handedness score, а не точность landmark.
- Сплайн landmark — координатный, не SE(3) и не костно-связный.
- Объектноцентричной репрезентации пока нет.
- Position и orientation цели сейчас гибридные: clean pinch center + hand-fit rotation.
- Гриппер — passive input для IK: его положение влияет на FK, пол и collision, но IK не меняет opening ради улучшения позы.
- Ускорение только штрафуется, жёсткого acceleration limit нет.
- Общий Huber смешивает метры и радианы после весового масштабирования.
- Есть техническая тонкость: локальный QP минимизирует квадратичные блоки с обычным множителем \(1/2\), а функция, используемая для отчёта и line search, записывает temporal/posture terms без \(1/2\), тогда как Huber внутри квадратичной области уже содержит \(1/2\). Поэтому локальная модель и отчётный функционал сейчас не буквально одинаково масштабируют регуляризаторы. Это уже отдельный кандидат на аккуратную правку.

Основные места реализации: [triangulate.py](/home/zifka/Personal/inno_study/ViKi/viki/perception/triangulate.py), [geometry.py](/home/zifka/Personal/inno_study/ViKi/viki/perception/geometry.py), [prepare/run.py](/home/zifka/Personal/inno_study/ViKi/viki/prepare/run.py), [hand_angles.py](/home/zifka/Personal/inno_study/ViKi/viki/perception/hand_angles.py), [retarget/run.py](/home/zifka/Personal/inno_study/ViKi/viki/retarget/run.py), [solver.py](/home/zifka/Personal/inno_study/ViKi/viki/retarget/solver.py).
