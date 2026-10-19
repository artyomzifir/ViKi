# HO-Cap retarget: численный дневник (2026-10-14)

Эпизодов: 28, завершено: 28, не завершено: 0. Источник: `paper/results/hocap/retarget/retarget_batch_status.json`.

UR10 + Robotiq 2F-85; hand-only IK, без cube-object grasp. База `[0, -1.0, 0]` и бесконечная плоскость стола `z=0` заданы в `hocap_tag_1`. Искусственный подход из robot-home отключён (`approach_sec=0`): оценивается только записанная демонстрация, безопасный переход из home этим тестом не доказан. Для этого HO-Cap батча IK инициализирован нулевым суставным вектором; для левой и правой руки используются разные фиксированные palm-to-TCP повороты. Значение `complete` означает, что численный план построен, а не что демонстрация исполнена роботом или прошла физическую проверку. Жёсткая столешница ограничивает движущиеся collision bodies и contact points захвата, но её реальная высота не измерена независимо. Стол не является конечной mesh-геометрией; человек и предметы не участвуют в collision avoidance. Это не гарантия физической безопасности или успешного захвата.

Сначала 676-кадровый smoke. Батч допускается при RMSE позиции ≤30 мм, p95 ≤50 мм, RMSE ориентации ≤45°, минимальном зазоре плоскости ≥−0.1 мм, self-collision margin ≥−0.1 мм и конечных суставах. Это операционный gate, не научный порог качества.

| Эпизод | Статус | Кадры | RMSE позиции, мм | p95, мм | RMSE ориентации, ° | Поправка скорости, мрад | Мин. зазор до плоскости, мм | Мин. self-collision margin, мм |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `subject_1_20231025_165502` | complete | 676 | 16.21 | 31.88 | 16.86 | 0.000 | 0.00 | 41.21 |
| `subject_1_20231025_165807` | complete | 798 | 17.29 | 36.20 | 26.58 | 0.000 | 0.00 | 41.21 |
| `subject_1_20231025_170105` | complete | 831 | 24.12 | 50.30 | 28.24 | 0.000 | 0.00 | 39.27 |
| `subject_1_20231025_170650` | complete | 741 | 23.83 | 44.05 | 22.47 | 0.000 | 0.00 | 39.27 |
| `subject_1_20231025_170959` | complete | 874 | 24.87 | 48.43 | 20.14 | 0.000 | 0.00 | 39.27 |
| `subject_2_20231022_201316` | complete | 828 | 11.65 | 20.03 | 16.29 | 0.000 | 0.00 | 39.27 |
| `subject_2_20231022_201942` | complete | 1561 | 16.57 | 33.15 | 57.03 | 0.000 | 0.01 | 35.37 |
| `subject_2_20231022_202617` | complete | 1271 | 11.95 | 19.28 | 11.05 | 0.000 | 0.00 | 39.29 |
| `subject_2_20231023_163929` | complete | 1129 | 11.07 | 20.96 | 11.84 | 0.000 | 0.00 | 39.34 |
| `subject_3_20231024_154531` | complete | 1686 | 12.26 | 20.84 | 10.75 | 0.000 | 0.00 | 39.81 |
| `subject_3_20231024_155008` | complete | 1046 | 16.11 | 28.51 | 13.34 | 0.000 | 0.00 | 40.25 |
| `subject_3_20231024_161209` | complete | 1071 | 14.86 | 24.15 | 12.95 | 0.000 | 0.00 | 39.97 |
| `subject_3_20231024_161937` | complete | 1269 | 12.22 | 21.23 | 10.97 | 0.000 | 0.00 | 40.20 |
| `subject_3_20231024_162327` | complete | 878 | 15.39 | 27.36 | 14.27 | 0.000 | 0.00 | 40.41 |
| `subject_3_20231024_162756` | complete | 946 | 14.10 | 23.98 | 11.31 | 0.000 | 0.00 | 39.72 |
| `subject_4_20231026_163223` | complete | 939 | 11.44 | 24.16 | 15.69 | 0.000 | 0.00 | 40.05 |
| `subject_4_20231026_164812` | complete | 974 | 7.87 | 16.25 | 10.20 | 0.000 | 0.00 | 39.64 |
| `subject_5_20231027_112303` | complete | 702 | 15.75 | 26.20 | 23.08 | 0.000 | 0.00 | 39.53 |
| `subject_6_20231025_110646` | complete | 1292 | 12.23 | 22.82 | 14.64 | 0.000 | -0.00 | 39.36 |
| `subject_6_20231025_111118` | complete | 1272 | 12.74 | 20.99 | 20.86 | 0.000 | 0.00 | 39.70 |
| `subject_6_20231025_112546` | complete | 1311 | 12.68 | 21.56 | 14.95 | 0.000 | 0.00 | 39.27 |
| `subject_7_20231022_193506` | complete | 1136 | 12.93 | 26.54 | 16.46 | 0.000 | 0.01 | 40.03 |
| `subject_7_20231023_162803` | complete | 1092 | 19.54 | 37.13 | 56.75 | 0.000 | 0.00 | 39.33 |
| `subject_8_20231024_180651` | complete | 985 | 11.01 | 24.28 | 19.50 | 0.000 | 0.00 | 39.61 |
| `subject_9_20231027_123403` | complete | 992 | 18.40 | 28.46 | 18.14 | 0.000 | 0.00 | 39.30 |
| `subject_9_20231027_123725` | complete | 1134 | 12.79 | 20.99 | 11.11 | 0.000 | 0.00 | 39.33 |
| `subject_9_20231027_124057` | complete | 1136 | 15.39 | 25.53 | 15.56 | 0.000 | 0.00 | 39.41 |
| `subject_9_20231027_125315` | complete | 1136 | 12.02 | 21.02 | 10.11 | 0.000 | 0.00 | 39.61 |

Ниже исходный JSON статуса целиком, без округления и без удаления полей. Полный `plan.h5` каждого успешного эпизода остаётся рядом с входным `cln.npz`; его `metrics_json` и `config_json` скопированы в статус без изменения значений.

SHA-256 статуса: `54bc5d4d8a26adebb48310bf8267343d0e004fed1999c67f9e992757446c6979`

```json
{
  "episodes": {
    "subject_1_20231025_165502_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          -175.675,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 676,
      "metrics": {
        "achieved_tool_axis_down_fraction": 1.0,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 24,
        "max_joint_acceleration_rad_s2": 2.508989097087639,
        "max_joint_velocity_rad_s": 0.7888532743111969,
        "min_collision_margin": 0.04121384269911854,
        "min_floor_margin_mm": 0.001019845628509597,
        "min_joint_limit_margin_rad": 1.810808884540286,
        "min_urdf_velocity_limit_margin_rad_s": 1.5520917583756022,
        "min_velocity_limit_margin_rad_s": 1.5520917583756022,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.4029317441944709,
        "objective_history": [
          {
            "acceleration": 0.0004318610186266663,
            "data": 18.77383526650427,
            "posture": 0.015098198490825101,
            "total": 18.79096502554104,
            "velocity": 0.001599699527319083
          },
          {
            "acceleration": 0.0007362474013356217,
            "data": 15.038888050657107,
            "posture": 0.05771686468362551,
            "total": 15.107823028127163,
            "velocity": 0.010481865385095537
          },
          {
            "acceleration": 0.001033608437516356,
            "data": 11.878363161247883,
            "posture": 0.12578562409038074,
            "total": 12.021463981602654,
            "velocity": 0.016281587826875287
          },
          {
            "acceleration": 0.0014335692615979783,
            "data": 8.766800210672846,
            "posture": 0.16210663995146293,
            "total": 8.956151665013829,
            "velocity": 0.02581124512792183
          },
          {
            "acceleration": 0.00183173332857667,
            "data": 5.864538622963299,
            "posture": 0.2356771244104828,
            "total": 6.13445505783467,
            "velocity": 0.032407577132311446
          },
          {
            "acceleration": 0.0023663286818159505,
            "data": 4.658225283233094,
            "posture": 0.2695337138630512,
            "total": 4.968054343790588,
            "velocity": 0.03792901801262682
          },
          {
            "acceleration": 0.0033667020109265982,
            "data": 3.774114725627198,
            "posture": 0.3070807759662168,
            "total": 4.138009382898949,
            "velocity": 0.053447179294607856
          },
          {
            "acceleration": 0.0070398293504392055,
            "data": 2.772367423942888,
            "posture": 0.38437848171405475,
            "total": 3.2764249850582723,
            "velocity": 0.11263925005089039
          },
          {
            "acceleration": 0.009061102364369543,
            "data": 2.3279780749904884,
            "posture": 0.4418830064452988,
            "total": 2.940930548750805,
            "velocity": 0.16200836495064844
          },
          {
            "acceleration": 0.009880170484355622,
            "data": 1.9889554674760879,
            "posture": 0.48785032863524613,
            "total": 2.6469660560631905,
            "velocity": 0.16028008946750108
          },
          {
            "acceleration": 0.010281842370580187,
            "data": 1.8195323758421793,
            "posture": 0.5153653688483872,
            "total": 2.5008105201412136,
            "velocity": 0.15563093308006695
          },
          {
            "acceleration": 0.011316466207380695,
            "data": 1.475215032879384,
            "posture": 0.5831025669761881,
            "total": 2.2211018504124405,
            "velocity": 0.15146778434948752
          },
          {
            "acceleration": 0.012820285001163653,
            "data": 1.112884606526121,
            "posture": 0.6682969438158987,
            "total": 1.9480810606926195,
            "velocity": 0.15407922534943613
          },
          {
            "acceleration": 0.014901479325031285,
            "data": 0.7592016179264327,
            "posture": 0.7650684600012904,
            "total": 1.7009883877915737,
            "velocity": 0.1618168305388194
          },
          {
            "acceleration": 0.017570503396806093,
            "data": 0.47715987013868444,
            "posture": 0.8539853799726307,
            "total": 1.5261566504695545,
            "velocity": 0.17744089696143328
          },
          {
            "acceleration": 0.01980692291973869,
            "data": 0.3094537202606467,
            "posture": 0.9256124920904463,
            "total": 1.447722462524957,
            "velocity": 0.19284932725412526
          },
          {
            "acceleration": 0.02043741108363562,
            "data": 0.26379125598530573,
            "posture": 0.9442556008948394,
            "total": 1.4290750091079325,
            "velocity": 0.20059074114415182
          },
          {
            "acceleration": 0.021045395569818864,
            "data": 0.22931662068131675,
            "posture": 0.9586837699613425,
            "total": 1.4182663995471259,
            "velocity": 0.20922061333464764
          },
          {
            "acceleration": 0.02150863871868322,
            "data": 0.2013450450712263,
            "posture": 0.9706192435651508,
            "total": 1.411335835926052,
            "velocity": 0.21786290857099153
          },
          {
            "acceleration": 0.02186244512708842,
            "data": 0.1795281394790644,
            "posture": 0.9797733615093931,
            "total": 1.406566837759709,
            "velocity": 0.225402891644163
          },
          {
            "acceleration": 0.02211754900673145,
            "data": 0.16254823312541666,
            "posture": 0.9869833868100814,
            "total": 1.403600418228301,
            "velocity": 0.2319512492860716
          },
          {
            "acceleration": 0.02228980576918956,
            "data": 0.14982191912783635,
            "posture": 0.9929327615076041,
            "total": 1.4029324894920445,
            "velocity": 0.23788800308741442
          },
          {
            "acceleration": 0.02230727431863617,
            "data": 0.14966261854533908,
            "posture": 0.992957535928508,
            "total": 1.4029317440994988,
            "velocity": 0.2380043153070156
          },
          {
            "acceleration": 0.022307275227228012,
            "data": 0.14966260951700908,
            "posture": 0.9929575376045369,
            "total": 1.4029317441944709,
            "velocity": 0.23800432184569698
          }
        ],
        "objective_terms": {
          "acceleration": 0.022307275227228012,
          "data": 0.14966260951700908,
          "posture": 0.9929575376045369,
          "total": 1.4029317441944709,
          "velocity": 0.23800432184569698
        },
        "orientation_rmse_deg": 16.858252763289553,
        "position_p95_mm": 31.881947907638235,
        "position_rmse_mm": 16.21135266451845,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 1.0,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 1.0,
        "wrist_target_height_median_mm": 134.79598998705515
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_1_20231025_165807_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          -175.675,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 798,
      "metrics": {
        "achieved_tool_axis_down_fraction": 1.0,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 20,
        "max_joint_acceleration_rad_s2": 2.7727295060460526,
        "max_joint_velocity_rad_s": 0.9346833093217545,
        "min_collision_margin": 0.041212735289025054,
        "min_floor_margin_mm": 2.2236032112821214e-05,
        "min_joint_limit_margin_rad": 1.8730765314985058,
        "min_urdf_velocity_limit_margin_rad_s": 1.510217298433799,
        "min_velocity_limit_margin_rad_s": 1.510217298433799,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.8267485477452703,
        "objective_history": [
          {
            "acceleration": 0.00035178525517538665,
            "data": 22.26615839276128,
            "posture": 0.017972564342257472,
            "total": 22.28614133463878,
            "velocity": 0.0016585922800658485
          },
          {
            "acceleration": 0.000854585669629638,
            "data": 17.54318715547074,
            "posture": 0.06884690884476052,
            "total": 17.623300315336518,
            "velocity": 0.010411665351386124
          },
          {
            "acceleration": 0.0011036447274349827,
            "data": 13.6321203679591,
            "posture": 0.14285887112041057,
            "total": 13.795246750812758,
            "velocity": 0.019163867005814274
          },
          {
            "acceleration": 0.0015751260771014852,
            "data": 9.837328395900203,
            "posture": 0.18261196997106152,
            "total": 10.055009392819427,
            "velocity": 0.033493900871062056
          },
          {
            "acceleration": 0.002181612861219197,
            "data": 6.5880090383366205,
            "posture": 0.259673178705182,
            "total": 6.89370723805178,
            "velocity": 0.043843408148759345
          },
          {
            "acceleration": 0.0030497976252466955,
            "data": 5.4074224230142605,
            "posture": 0.2950810974885869,
            "total": 5.760400603100524,
            "velocity": 0.05484728497243021
          },
          {
            "acceleration": 0.0071544601706809855,
            "data": 4.076377743742129,
            "posture": 0.375801480261624,
            "total": 4.575734924105343,
            "velocity": 0.11640123993090899
          },
          {
            "acceleration": 0.009063710410440724,
            "data": 3.4633897589856026,
            "posture": 0.4427812245100102,
            "total": 4.099696181353584,
            "velocity": 0.18446148744753163
          },
          {
            "acceleration": 0.01026903741734112,
            "data": 3.0094850370933757,
            "posture": 0.49833059262894697,
            "total": 3.7054559984205278,
            "velocity": 0.18737133128086397
          },
          {
            "acceleration": 0.01031256733128357,
            "data": 2.9007485616253845,
            "posture": 0.5123132403726594,
            "total": 3.6066596973718448,
            "velocity": 0.1832853280425172
          },
          {
            "acceleration": 0.011019397906021964,
            "data": 2.472283182556456,
            "posture": 0.5791001903490157,
            "total": 3.236762271506445,
            "velocity": 0.1743595006949511
          },
          {
            "acceleration": 0.01212590048398282,
            "data": 2.0281408697950587,
            "posture": 0.664581642286967,
            "total": 2.8751639967144866,
            "velocity": 0.17031558414847864
          },
          {
            "acceleration": 0.013702776093080423,
            "data": 1.5701555715913407,
            "posture": 0.7661095762806927,
            "total": 2.5222105835047097,
            "velocity": 0.17224265953959586
          },
          {
            "acceleration": 0.015722297797814325,
            "data": 1.1332234310810319,
            "posture": 0.8804759448671774,
            "total": 2.207720897014513,
            "velocity": 0.17829922326848932
          },
          {
            "acceleration": 0.01816757331781201,
            "data": 0.8006515716702014,
            "posture": 0.9949866880159851,
            "total": 2.003198067200969,
            "velocity": 0.18939223419697074
          },
          {
            "acceleration": 0.02114456168291029,
            "data": 0.5723819409132771,
            "posture": 1.0773024033103291,
            "total": 1.8893152463028837,
            "velocity": 0.21848634039636716
          },
          {
            "acceleration": 0.022255025468687375,
            "data": 0.4870156577795664,
            "posture": 1.1095893433324353,
            "total": 1.8556592674910215,
            "velocity": 0.23679924091033228
          },
          {
            "acceleration": 0.02324554054283193,
            "data": 0.4175268389661273,
            "posture": 1.1376988048067056,
            "total": 1.8345625112590378,
            "velocity": 0.256091326943373
          },
          {
            "acceleration": 0.023613299058419662,
            "data": 0.38829232077833425,
            "posture": 1.1496803604607415,
            "total": 1.8268572822095321,
            "velocity": 0.2652713019120368
          },
          {
            "acceleration": 0.02361820766059616,
            "data": 0.3878686485303273,
            "posture": 1.1498525471485126,
            "total": 1.8267485477452703,
            "velocity": 0.26540914440583424
          }
        ],
        "objective_terms": {
          "acceleration": 0.02361820766059616,
          "data": 0.3878686485303273,
          "posture": 1.1498525471485126,
          "total": 1.8267485477452703,
          "velocity": 0.26540914440583424
        },
        "orientation_rmse_deg": 26.584551211474178,
        "position_p95_mm": 36.20450233892133,
        "position_rmse_mm": 17.285824863914826,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 1.0,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 1.0,
        "wrist_target_height_median_mm": 129.39559085093288
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_1_20231025_170105_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 831,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.5944645006016848,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 31,
        "max_joint_acceleration_rad_s2": 2.8267122495921417,
        "max_joint_velocity_rad_s": 0.9673571452391312,
        "min_collision_margin": 0.039269509060080074,
        "min_floor_margin_mm": 0.00013793033075068517,
        "min_joint_limit_margin_rad": 1.7447878477923442,
        "min_urdf_velocity_limit_margin_rad_s": 1.3826293368673452,
        "min_velocity_limit_margin_rad_s": 1.3826293368673452,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.567002743382626,
        "objective_history": [
          {
            "acceleration": 0.00041751518192467246,
            "data": 20.127792141345385,
            "posture": 0.016166380704074352,
            "total": 20.147536156545875,
            "velocity": 0.0031601193144915104
          },
          {
            "acceleration": 0.0009457464915429241,
            "data": 15.356925711614902,
            "posture": 0.06545125337871377,
            "total": 15.438436639421537,
            "velocity": 0.01511392793637938
          },
          {
            "acceleration": 0.0012752242137794663,
            "data": 11.35558953908681,
            "posture": 0.1469804810712597,
            "total": 11.531076382263551,
            "velocity": 0.027231137891702492
          },
          {
            "acceleration": 0.001700690434851302,
            "data": 7.331947552498009,
            "posture": 0.18811544668167832,
            "total": 7.558151300259059,
            "velocity": 0.03638761064452098
          },
          {
            "acceleration": 0.002612629663083698,
            "data": 3.898632336278581,
            "posture": 0.2647681841351037,
            "total": 4.21221716558845,
            "velocity": 0.046204015511682656
          },
          {
            "acceleration": 0.004362490799829413,
            "data": 2.9094569256718152,
            "posture": 0.2930225525925143,
            "total": 3.2696798124629085,
            "velocity": 0.06283784339875009
          },
          {
            "acceleration": 0.007305683119185713,
            "data": 2.1768114260441274,
            "posture": 0.3180545545091855,
            "total": 2.604619353500534,
            "velocity": 0.10244768982803537
          },
          {
            "acceleration": 0.00877267213176186,
            "data": 1.939705870506177,
            "posture": 0.3282568880113387,
            "total": 2.405938457662384,
            "velocity": 0.12920302701310615
          },
          {
            "acceleration": 0.010275317922738439,
            "data": 1.7464504808334795,
            "posture": 0.3375259118688199,
            "total": 2.25390230836971,
            "velocity": 0.159650597744672
          },
          {
            "acceleration": 0.010979935679183027,
            "data": 1.6672408595362618,
            "posture": 0.3414039064450008,
            "total": 2.195172241232282,
            "velocity": 0.17554753957183616
          },
          {
            "acceleration": 0.014033059043927562,
            "data": 1.3955544003633973,
            "posture": 0.3569218172034102,
            "total": 2.017907450238836,
            "velocity": 0.25139817362810096
          },
          {
            "acceleration": 0.01640161218899801,
            "data": 1.2329029900821273,
            "posture": 0.36835025117822584,
            "total": 1.94686533366569,
            "velocity": 0.32921048021633886
          },
          {
            "acceleration": 0.018261023258464723,
            "data": 1.142060259987213,
            "posture": 0.3692081426657145,
            "total": 1.922817997417736,
            "velocity": 0.3932885715063436
          },
          {
            "acceleration": 0.01890190372940624,
            "data": 1.1116895993907447,
            "posture": 0.36811020853060966,
            "total": 1.9152619198942267,
            "velocity": 0.4165602082434662
          },
          {
            "acceleration": 0.01943623499910666,
            "data": 1.0835086683688724,
            "posture": 0.3668952210318024,
            "total": 1.9070269172581935,
            "velocity": 0.4371867928584121
          },
          {
            "acceleration": 0.0198920447220263,
            "data": 1.0559073747511825,
            "posture": 0.3655729099167967,
            "total": 1.8970421696750321,
            "velocity": 0.4556698402850269
          },
          {
            "acceleration": 0.02029366921632139,
            "data": 1.0281273535503743,
            "posture": 0.3644057278024807,
            "total": 1.8851401140224535,
            "velocity": 0.47231336345327707
          },
          {
            "acceleration": 0.020623601822756906,
            "data": 0.9996784689604074,
            "posture": 0.3634321427386625,
            "total": 1.8712992225990885,
            "velocity": 0.48756500907726186
          },
          {
            "acceleration": 0.020912690301800596,
            "data": 0.9701233637189367,
            "posture": 0.3626635045084333,
            "total": 1.8554901201838603,
            "velocity": 0.5017905616546897
          },
          {
            "acceleration": 0.021439820935499895,
            "data": 0.9056231353495803,
            "posture": 0.3622356898676637,
            "total": 1.8195477983026542,
            "velocity": 0.5302491521499103
          },
          {
            "acceleration": 0.021859333773428167,
            "data": 0.8370166514479014,
            "posture": 0.36311941576061185,
            "total": 1.777239265444511,
            "velocity": 0.5552438644625698
          },
          {
            "acceleration": 0.021945737426353737,
            "data": 0.8195344059299329,
            "posture": 0.36346311557839106,
            "total": 1.765431875977784,
            "velocity": 0.5604886170431067
          },
          {
            "acceleration": 0.02213747687658661,
            "data": 0.7838128474703909,
            "posture": 0.3644849409663381,
            "total": 1.7416362922399289,
            "velocity": 0.5712010269266133
          },
          {
            "acceleration": 0.023095235658958452,
            "data": 0.6413174484243472,
            "posture": 0.37372688005842225,
            "total": 1.6598359581427613,
            "velocity": 0.6216963940010334
          },
          {
            "acceleration": 0.024372221629116758,
            "data": 0.535818720259295,
            "posture": 0.38520919185029096,
            "total": 1.6025954963824562,
            "velocity": 0.6571953626437538
          },
          {
            "acceleration": 0.02455075607257193,
            "data": 0.5214198771453511,
            "posture": 0.3874146046620536,
            "total": 1.591978443020356,
            "velocity": 0.6585932051403794
          },
          {
            "acceleration": 0.02473676436543718,
            "data": 0.5090853018836528,
            "posture": 0.3895898015639222,
            "total": 1.5836264331876468,
            "velocity": 0.6602145653746346
          },
          {
            "acceleration": 0.024917443025568684,
            "data": 0.49847355239371965,
            "posture": 0.391735924092434,
            "total": 1.5769387995887527,
            "velocity": 0.6618118800770302
          },
          {
            "acceleration": 0.025263126544019313,
            "data": 0.48044451327954296,
            "posture": 0.39612155817723066,
            "total": 1.5672190296789947,
            "velocity": 0.665389831678202
          },
          {
            "acceleration": 0.02531553615033261,
            "data": 0.4799415536318962,
            "posture": 0.39617093230854883,
            "total": 1.5670242299407247,
            "velocity": 0.665596207849947
          },
          {
            "acceleration": 0.02532150818403972,
            "data": 0.47988394314363336,
            "posture": 0.3961770367534883,
            "total": 1.567002743382626,
            "velocity": 0.6656202553014647
          }
        ],
        "objective_terms": {
          "acceleration": 0.02532150818403972,
          "data": 0.47988394314363336,
          "posture": 0.3961770367534883,
          "total": 1.567002743382626,
          "velocity": 0.6656202553014647
        },
        "orientation_rmse_deg": 28.241488719326128,
        "position_p95_mm": 50.303020617401735,
        "position_rmse_mm": 24.116758184912193,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.690734055354994,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.6425992779783394,
        "wrist_target_height_median_mm": 39.49038898006785
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_1_20231025_170650_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 741,
      "metrics": {
        "achieved_tool_axis_down_fraction": 1.0,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 27,
        "max_joint_acceleration_rad_s2": 3.013154598249749,
        "max_joint_velocity_rad_s": 1.0733912936983532,
        "min_collision_margin": 0.03927263533224719,
        "min_floor_margin_mm": 3.337211222642633e-06,
        "min_joint_limit_margin_rad": 1.882969193848075,
        "min_urdf_velocity_limit_margin_rad_s": 1.3349593895432803,
        "min_velocity_limit_margin_rad_s": 1.3349593895432803,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.2161166020176735,
        "objective_history": [
          {
            "acceleration": 0.0002979331445958369,
            "data": 20.15861191111963,
            "posture": 0.013990202124442479,
            "total": 20.174954803149937,
            "velocity": 0.0020547567612708407
          },
          {
            "acceleration": 0.0009986223745493837,
            "data": 15.489895354458202,
            "posture": 0.05646320421236243,
            "total": 15.559812548438355,
            "velocity": 0.012455367393242616
          },
          {
            "acceleration": 0.001303255669869194,
            "data": 11.599865662037445,
            "posture": 0.12279473447146107,
            "total": 11.748016525695423,
            "velocity": 0.02405287351664914
          },
          {
            "acceleration": 0.0019567211379896256,
            "data": 7.6687155879198645,
            "posture": 0.15397985921763652,
            "total": 7.858494444548118,
            "velocity": 0.033842276272627314
          },
          {
            "acceleration": 0.0029766293539110473,
            "data": 4.343400764492575,
            "posture": 0.21719159056146714,
            "total": 4.605272814853357,
            "velocity": 0.04170383044540389
          },
          {
            "acceleration": 0.00976984168274584,
            "data": 2.3824437129081835,
            "posture": 0.27155717605583185,
            "total": 2.7631779486243033,
            "velocity": 0.09940721797754255
          },
          {
            "acceleration": 0.012354647158112204,
            "data": 1.8646127266064094,
            "posture": 0.2856581886783559,
            "total": 2.3068502134896853,
            "velocity": 0.14422465104680804
          },
          {
            "acceleration": 0.015302997732565979,
            "data": 1.5187454624141212,
            "posture": 0.2945190248713041,
            "total": 2.033304484661901,
            "velocity": 0.20473699964390998
          },
          {
            "acceleration": 0.017765342931069437,
            "data": 1.317403978371652,
            "posture": 0.297217071141302,
            "total": 1.8983651684207086,
            "velocity": 0.2659787759766851
          },
          {
            "acceleration": 0.018753216990485516,
            "data": 1.2534100759808067,
            "posture": 0.2959609638888999,
            "total": 1.857912034734166,
            "velocity": 0.28978777787397386
          },
          {
            "acceleration": 0.020579338616271148,
            "data": 1.13817717626694,
            "posture": 0.2948467490833336,
            "total": 1.7874969817898771,
            "velocity": 0.3338937178233325
          },
          {
            "acceleration": 0.02193430037093783,
            "data": 1.0345698739606872,
            "posture": 0.29525569502507043,
            "total": 1.7169461619760988,
            "velocity": 0.36518629261940344
          },
          {
            "acceleration": 0.024337244040538296,
            "data": 0.8244400864226372,
            "posture": 0.3029330634852031,
            "total": 1.5643221254913655,
            "velocity": 0.41261173154298675
          },
          {
            "acceleration": 0.026356518403551323,
            "data": 0.6245237746940282,
            "posture": 0.31765554946116287,
            "total": 1.3998056109082724,
            "velocity": 0.4312697683495301
          },
          {
            "acceleration": 0.02876881596357769,
            "data": 0.4673601313572035,
            "posture": 0.3373412566766984,
            "total": 1.282536019739246,
            "velocity": 0.44906581574176635
          },
          {
            "acceleration": 0.030036624018216863,
            "data": 0.37857799786688107,
            "posture": 0.3537568154838411,
            "total": 1.230369720826494,
            "velocity": 0.4679982834575552
          },
          {
            "acceleration": 0.03078280381313926,
            "data": 0.336090943342874,
            "posture": 0.36455094167561686,
            "total": 1.2176853867977662,
            "velocity": 0.4862606979661361
          },
          {
            "acceleration": 0.03077641517169964,
            "data": 0.3355680297210597,
            "posture": 0.36467089058265,
            "total": 1.2171441078169316,
            "velocity": 0.4861287723415222
          },
          {
            "acceleration": 0.0307791749183843,
            "data": 0.3351131784931556,
            "posture": 0.3648044803395901,
            "total": 1.21678132559112,
            "velocity": 0.4860844918399898
          },
          {
            "acceleration": 0.030786023506504688,
            "data": 0.3348166077868652,
            "posture": 0.36490886468856165,
            "total": 1.2166012544352836,
            "velocity": 0.48608975845335206
          },
          {
            "acceleration": 0.030792054356570897,
            "data": 0.3345556655294852,
            "posture": 0.36500723703006926,
            "total": 1.2164750129003983,
            "velocity": 0.48612005598427294
          },
          {
            "acceleration": 0.03079702385080528,
            "data": 0.3343171008044881,
            "posture": 0.36510045268040725,
            "total": 1.2163807427456026,
            "velocity": 0.48616616540990193
          },
          {
            "acceleration": 0.03080108266242524,
            "data": 0.3340942193924107,
            "posture": 0.36518929014419127,
            "total": 1.2163071905389224,
            "velocity": 0.48622259833989534
          },
          {
            "acceleration": 0.03080439354686893,
            "data": 0.3338826368565523,
            "posture": 0.3652743302159897,
            "total": 1.216247853737017,
            "velocity": 0.486286493117606
          },
          {
            "acceleration": 0.030807155765766318,
            "data": 0.3336804896340439,
            "posture": 0.3653561961013449,
            "total": 1.216199394098669,
            "velocity": 0.486355552597514
          },
          {
            "acceleration": 0.03081201085215224,
            "data": 0.33329442017022104,
            "posture": 0.3655146917840383,
            "total": 1.2161254386071267,
            "velocity": 0.4865043158007152
          },
          {
            "acceleration": 0.030811627749811204,
            "data": 0.3332701564549273,
            "posture": 0.36552225167612074,
            "total": 1.2161166020176735,
            "velocity": 0.4865125661368144
          }
        ],
        "objective_terms": {
          "acceleration": 0.030811627749811204,
          "data": 0.3332701564549273,
          "posture": 0.36552225167612074,
          "total": 1.2161166020176735,
          "velocity": 0.4865125661368144
        },
        "orientation_rmse_deg": 22.468745830316617,
        "position_p95_mm": 44.05298910893427,
        "position_rmse_mm": 23.82640726740428,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9595141700404858,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9919028340080972,
        "wrist_target_height_median_mm": 88.88726183876456
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_1_20231025_170959_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 874,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9416475972540046,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 23,
        "max_joint_acceleration_rad_s2": 3.7439131610522542,
        "max_joint_velocity_rad_s": 1.1812926939236088,
        "min_collision_margin": 0.03927033709074207,
        "min_floor_margin_mm": 0.0011529516918892813,
        "min_joint_limit_margin_rad": 1.8227073967403562,
        "min_urdf_velocity_limit_margin_rad_s": 1.3634856111539522,
        "min_velocity_limit_margin_rad_s": 1.3634856111539522,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.392310862831228,
        "objective_history": [
          {
            "acceleration": 0.0006966858675639365,
            "data": 23.650375707431586,
            "posture": 0.016712293242180303,
            "total": 23.67081453461303,
            "velocity": 0.003029848071702303
          },
          {
            "acceleration": 0.0011810987041027832,
            "data": 18.09828983216568,
            "posture": 0.06681254030004215,
            "total": 18.184867528039845,
            "velocity": 0.01858405687002281
          },
          {
            "acceleration": 0.0015434692755579768,
            "data": 13.54248152370483,
            "posture": 0.14343577328330517,
            "total": 13.713271964168399,
            "velocity": 0.02581119790470436
          },
          {
            "acceleration": 0.0019582081793860237,
            "data": 8.817524080108356,
            "posture": 0.17765456134856725,
            "total": 9.033973196979494,
            "velocity": 0.036836347343183
          },
          {
            "acceleration": 0.0032890818467045373,
            "data": 4.831712476046284,
            "posture": 0.2475738598880569,
            "total": 5.134082657956504,
            "velocity": 0.051507240175459025
          },
          {
            "acceleration": 0.005417098300856915,
            "data": 3.534213337583622,
            "posture": 0.2775219329106533,
            "total": 3.8915055657826514,
            "velocity": 0.0743531969875189
          },
          {
            "acceleration": 0.00900994027784018,
            "data": 2.6438538636929465,
            "posture": 0.3044307568052603,
            "total": 3.0754920072882395,
            "velocity": 0.11819744651219256
          },
          {
            "acceleration": 0.013039926970976671,
            "data": 2.114524909457237,
            "posture": 0.3226713601196716,
            "total": 2.6284555154246263,
            "velocity": 0.1782193188767408
          },
          {
            "acceleration": 0.016673155288779853,
            "data": 1.7929817803075814,
            "posture": 0.3304804870625005,
            "total": 2.3885685854022807,
            "velocity": 0.2484331627434185
          },
          {
            "acceleration": 0.019614325280488402,
            "data": 1.5829494689402221,
            "posture": 0.3378613350978129,
            "total": 2.2527325432835306,
            "velocity": 0.312307413965007
          },
          {
            "acceleration": 0.022113485757882537,
            "data": 1.4263700715502519,
            "posture": 0.3423978035671915,
            "total": 2.154254720020564,
            "velocity": 0.36337335914523766
          },
          {
            "acceleration": 0.0240508219185234,
            "data": 1.2879058480502543,
            "posture": 0.347541502576706,
            "total": 2.060865447105837,
            "velocity": 0.4013672745603533
          },
          {
            "acceleration": 0.027398851481605528,
            "data": 1.0126359641628653,
            "posture": 0.3649871929990103,
            "total": 1.8671807761854622,
            "velocity": 0.4621587675419811
          },
          {
            "acceleration": 0.030126361466470363,
            "data": 0.7546678353087781,
            "posture": 0.3858881207196008,
            "total": 1.6617572990541456,
            "velocity": 0.49107498155929624
          },
          {
            "acceleration": 0.0327528974872983,
            "data": 0.5498937222004888,
            "posture": 0.4124475261727105,
            "total": 1.5096231157714664,
            "velocity": 0.5145289699109686
          },
          {
            "acceleration": 0.034523800437698954,
            "data": 0.4323366738247554,
            "posture": 0.4332797482763047,
            "total": 1.4339718746222538,
            "velocity": 0.5338316520834948
          },
          {
            "acceleration": 0.034684658826927804,
            "data": 0.40660322466567766,
            "posture": 0.43767220961070036,
            "total": 1.4128789599938383,
            "velocity": 0.5339188668905324
          },
          {
            "acceleration": 0.03480380175475003,
            "data": 0.3882726424269519,
            "posture": 0.4415552900977275,
            "total": 1.4006928632317452,
            "velocity": 0.5360611289523158
          },
          {
            "acceleration": 0.03487747183504809,
            "data": 0.3752482088286443,
            "posture": 0.44491176477836253,
            "total": 1.3940074626414507,
            "velocity": 0.5389700171993959
          },
          {
            "acceleration": 0.0348864810107324,
            "data": 0.3743299691471593,
            "posture": 0.4450637792417452,
            "total": 1.3931460779664975,
            "velocity": 0.5388658485668606
          },
          {
            "acceleration": 0.034892973409493454,
            "data": 0.37381646838887084,
            "posture": 0.44517770641543514,
            "total": 1.3926887532654686,
            "velocity": 0.5388016050516691
          },
          {
            "acceleration": 0.034895475205539324,
            "data": 0.37344283118182847,
            "posture": 0.44528144077964493,
            "total": 1.3923741207216158,
            "velocity": 0.538754373554603
          },
          {
            "acceleration": 0.034894618085662514,
            "data": 0.3733668963815342,
            "posture": 0.4453011032993009,
            "total": 1.392310862831228,
            "velocity": 0.5387482450647303
          }
        ],
        "objective_terms": {
          "acceleration": 0.034894618085662514,
          "data": 0.3733668963815342,
          "posture": 0.4453011032993009,
          "total": 1.392310862831228,
          "velocity": 0.5387482450647303
        },
        "orientation_rmse_deg": 20.140141580617243,
        "position_p95_mm": 48.43461740755887,
        "position_rmse_mm": 24.87373304668864,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9244851258581236,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.937070938215103,
        "wrist_target_height_median_mm": 118.14755064183258
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_2_20231022_201316_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 828,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9480676328502415,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 25,
        "max_joint_acceleration_rad_s2": 1.6019332757656388,
        "max_joint_velocity_rad_s": 0.72635669247492,
        "min_collision_margin": 0.03927045935629214,
        "min_floor_margin_mm": 0.0002257140086225906,
        "min_joint_limit_margin_rad": 1.7929318639388003,
        "min_urdf_velocity_limit_margin_rad_s": 1.7110938050390583,
        "min_velocity_limit_margin_rad_s": 1.7110938050390583,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 0.9251228547086556,
        "objective_history": [
          {
            "acceleration": 0.0003273813889762661,
            "data": 24.23830992369844,
            "posture": 0.0154171858910385,
            "total": 24.255158096310467,
            "velocity": 0.0011036053320075737
          },
          {
            "acceleration": 6.789683532038746e-05,
            "data": 18.533207387797397,
            "posture": 0.05740812103380528,
            "total": 18.59101342362303,
            "velocity": 0.0003300179565044599
          },
          {
            "acceleration": 4.671406567026972e-05,
            "data": 13.699228287265642,
            "posture": 0.13142413340579512,
            "total": 13.831500719082277,
            "velocity": 0.0008015843451687345
          },
          {
            "acceleration": 0.00027951984147324627,
            "data": 8.620091931600431,
            "posture": 0.16311814801174043,
            "total": 8.787033570690472,
            "velocity": 0.0035439712368270284
          },
          {
            "acceleration": 0.001142415018435063,
            "data": 4.205459432923026,
            "posture": 0.23017734131613612,
            "total": 4.467339903509902,
            "velocity": 0.030560714252304992
          },
          {
            "acceleration": 0.003008192829676897,
            "data": 1.7602137991250064,
            "posture": 0.2892967142201146,
            "total": 2.1285606873396024,
            "velocity": 0.0760419811648048
          },
          {
            "acceleration": 0.0035694908125902215,
            "data": 1.3413521793014191,
            "posture": 0.3039490067887049,
            "total": 1.7520641472824823,
            "velocity": 0.10319347037976796
          },
          {
            "acceleration": 0.004451811710269797,
            "data": 1.0887297857010005,
            "posture": 0.31148889254057627,
            "total": 1.541972420343909,
            "velocity": 0.13730193039206223
          },
          {
            "acceleration": 0.005367277643692575,
            "data": 0.9184084574884499,
            "posture": 0.31664763709996435,
            "total": 1.41544308210924,
            "velocity": 0.1750197098771334
          },
          {
            "acceleration": 0.006213267117790243,
            "data": 0.7825364059734417,
            "posture": 0.3209163815351826,
            "total": 1.3210396184218178,
            "velocity": 0.21137356379540342
          },
          {
            "acceleration": 0.006887003670423428,
            "data": 0.6588030381153347,
            "posture": 0.32516947066853125,
            "total": 1.2348107318653279,
            "velocity": 0.24395121941103848
          },
          {
            "acceleration": 0.008106142036614755,
            "data": 0.437479486194406,
            "posture": 0.338704521159324,
            "total": 1.090587051644718,
            "velocity": 0.30629690225437334
          },
          {
            "acceleration": 0.008498222124689038,
            "data": 0.28983454694827976,
            "posture": 0.3543173481043746,
            "total": 0.9924774278312771,
            "velocity": 0.3398273106539338
          },
          {
            "acceleration": 0.008501634037072655,
            "data": 0.22909121651819783,
            "posture": 0.3638193725815319,
            "total": 0.9453960836250579,
            "velocity": 0.3439838604882555
          },
          {
            "acceleration": 0.008445069906610498,
            "data": 0.21873818033500608,
            "posture": 0.3663644431717334,
            "total": 0.930635850175644,
            "velocity": 0.33708815676229403
          },
          {
            "acceleration": 0.008430623418425863,
            "data": 0.21779518265539433,
            "posture": 0.3663705278394457,
            "total": 0.9289396021436824,
            "velocity": 0.33634326823041666
          },
          {
            "acceleration": 0.00842430999824522,
            "data": 0.21729404946609227,
            "posture": 0.3664055607923575,
            "total": 0.9279268788190802,
            "velocity": 0.33580295856238523
          },
          {
            "acceleration": 0.008420632909407864,
            "data": 0.21699310171753503,
            "posture": 0.366447778081136,
            "total": 0.9272766146070444,
            "velocity": 0.33541510189896556
          },
          {
            "acceleration": 0.008419320523626022,
            "data": 0.2168906387591652,
            "posture": 0.3664737609913972,
            "total": 0.9270201242267004,
            "velocity": 0.3352364039525119
          },
          {
            "acceleration": 0.008418293167880215,
            "data": 0.21677461639955742,
            "posture": 0.36651300853009305,
            "total": 0.9266765416269269,
            "velocity": 0.3349706235293962
          },
          {
            "acceleration": 0.008416954969614086,
            "data": 0.2166776999422017,
            "posture": 0.3665563145626439,
            "total": 0.9263715345597042,
            "velocity": 0.3347205650852446
          },
          {
            "acceleration": 0.008413958009847834,
            "data": 0.21652999384276422,
            "posture": 0.3666187914925827,
            "total": 0.9259715862185729,
            "velocity": 0.3344088428733782
          },
          {
            "acceleration": 0.008411014023608184,
            "data": 0.2163894456913919,
            "posture": 0.366699180842875,
            "total": 0.9255481002101447,
            "velocity": 0.33404845965226954
          },
          {
            "acceleration": 0.008408081814595915,
            "data": 0.216255040963791,
            "posture": 0.36678106198909,
            "total": 0.9251465722678189,
            "velocity": 0.333702387500342
          },
          {
            "acceleration": 0.00840789714632131,
            "data": 0.21624677355879432,
            "posture": 0.3667862082523697,
            "total": 0.9251228547086556,
            "velocity": 0.33368197575117037
          }
        ],
        "objective_terms": {
          "acceleration": 0.00840789714632131,
          "data": 0.21624677355879432,
          "posture": 0.3667862082523697,
          "total": 0.9251228547086556,
          "velocity": 0.33368197575117037
        },
        "orientation_rmse_deg": 16.289722916032463,
        "position_p95_mm": 20.02972037267494,
        "position_rmse_mm": 11.649977487112416,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.893719806763285,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9444444444444444,
        "wrist_target_height_median_mm": 84.83607649844548
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_2_20231022_201942_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          -175.675,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1561,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9628443305573351,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 32,
        "max_joint_acceleration_rad_s2": 1.5236300320370397,
        "max_joint_velocity_rad_s": 1.642273737902753,
        "min_collision_margin": 0.03537371044301993,
        "min_floor_margin_mm": 0.011406024559235117,
        "min_joint_limit_margin_rad": 1.3651016319655096,
        "min_urdf_velocity_limit_margin_rad_s": 1.496729797988392,
        "min_velocity_limit_margin_rad_s": 1.496729797988392,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 5.922689137223537,
        "objective_history": [
          {
            "acceleration": 0.000531127052635667,
            "data": 48.95394664417757,
            "posture": 0.0335246686056793,
            "total": 49.00054245894487,
            "velocity": 0.012540019108982477
          },
          {
            "acceleration": 0.00046090135274737926,
            "data": 39.21287799332474,
            "posture": 0.12401091935116301,
            "total": 39.375243764796444,
            "velocity": 0.03789395076779721
          },
          {
            "acceleration": 0.0005383446274780982,
            "data": 34.888326145174176,
            "posture": 0.1805911626127958,
            "total": 35.114623870466346,
            "velocity": 0.04516821805189724
          },
          {
            "acceleration": 0.0007879891744733555,
            "data": 30.711799919300912,
            "posture": 0.2341364706903387,
            "total": 31.01796591894466,
            "velocity": 0.07124153977893742
          },
          {
            "acceleration": 0.001017400352885044,
            "data": 26.480566686536296,
            "posture": 0.27938140889463475,
            "total": 26.86123886483977,
            "velocity": 0.10027336905595048
          },
          {
            "acceleration": 0.0013727472592346462,
            "data": 22.262027963480776,
            "posture": 0.3440261190727027,
            "total": 22.74432417337564,
            "velocity": 0.13689734356292968
          },
          {
            "acceleration": 0.0019200398670290975,
            "data": 18.184961239956298,
            "posture": 0.4178981581306891,
            "total": 18.78998515416881,
            "velocity": 0.18520571621479345
          },
          {
            "acceleration": 0.0022852344880178407,
            "data": 16.311232418095273,
            "posture": 0.4545044936896404,
            "total": 16.980822237202545,
            "velocity": 0.21280009092961555
          },
          {
            "acceleration": 0.0035059370924480017,
            "data": 12.90942474403752,
            "posture": 0.531607228744589,
            "total": 13.733100776973508,
            "velocity": 0.28856286709895085
          },
          {
            "acceleration": 0.0051208946508312205,
            "data": 10.269891036997254,
            "posture": 0.6051760180933323,
            "total": 11.272290392986271,
            "velocity": 0.39210244324485305
          },
          {
            "acceleration": 0.007024491760567788,
            "data": 8.646284175815769,
            "posture": 0.6577890447015562,
            "total": 9.822763594639559,
            "velocity": 0.5116658823616663
          },
          {
            "acceleration": 0.00786122831335681,
            "data": 8.130468718536997,
            "posture": 0.6791831273870772,
            "total": 9.38595185263591,
            "velocity": 0.5684387783984785
          },
          {
            "acceleration": 0.009753862949962012,
            "data": 7.355608215353742,
            "posture": 0.7235501395143885,
            "total": 8.774061041749144,
            "velocity": 0.6851488239310506
          },
          {
            "acceleration": 0.011398295280927217,
            "data": 6.83208524130378,
            "posture": 0.7667331169508462,
            "total": 8.395922786522753,
            "velocity": 0.7857061329871995
          },
          {
            "acceleration": 0.012695323011311248,
            "data": 6.401232335856919,
            "posture": 0.8113647103497563,
            "total": 8.097990417821913,
            "velocity": 0.872698048603927
          },
          {
            "acceleration": 0.01317683539622062,
            "data": 6.203515934810753,
            "posture": 0.834758450012618,
            "total": 7.9617550476253625,
            "velocity": 0.9103038274057713
          },
          {
            "acceleration": 0.014206664855415147,
            "data": 5.818710328852513,
            "posture": 0.8860414808505245,
            "total": 7.7061150848603495,
            "velocity": 0.9871566103018975
          },
          {
            "acceleration": 0.014588609115109775,
            "data": 5.6300228581639615,
            "posture": 0.9134859404744492,
            "total": 7.58074743242277,
            "velocity": 1.0226500246692491
          },
          {
            "acceleration": 0.015002902315579651,
            "data": 5.442530105891347,
            "posture": 0.9423273583912662,
            "total": 7.4576642576135015,
            "velocity": 1.0578038910153085
          },
          {
            "acceleration": 0.015881830699043368,
            "data": 5.067685165591467,
            "posture": 1.004515152563733,
            "total": 7.217206917679165,
            "velocity": 1.1291247688249222
          },
          {
            "acceleration": 0.016246293822046853,
            "data": 4.879768658556824,
            "posture": 1.0379459574275522,
            "total": 7.098265077690616,
            "velocity": 1.1643041678841943
          },
          {
            "acceleration": 0.016612096060969424,
            "data": 4.691977757765714,
            "posture": 1.0729158853686467,
            "total": 6.981453963506242,
            "velocity": 1.1999482243109123
          },
          {
            "acceleration": 0.017449552350498804,
            "data": 4.318739952771061,
            "posture": 1.147523608414286,
            "total": 6.759003571428564,
            "velocity": 1.2752904578927182
          },
          {
            "acceleration": 0.019290191232145926,
            "data": 3.5978086852830167,
            "posture": 1.312972411733699,
            "total": 6.370221119003523,
            "velocity": 1.4401498307546614
          },
          {
            "acceleration": 0.020867916051505297,
            "data": 2.9667914208213473,
            "posture": 1.480004437093122,
            "total": 6.084948954363169,
            "velocity": 1.6172851803971935
          },
          {
            "acceleration": 0.022457793206373127,
            "data": 2.4910985785430597,
            "posture": 1.631499265302232,
            "total": 5.955254354363801,
            "velocity": 1.8101987173121357
          },
          {
            "acceleration": 0.023093997967092255,
            "data": 2.316718421976291,
            "posture": 1.6865509497843938,
            "total": 5.928506563797927,
            "velocity": 1.9021431940701503
          },
          {
            "acceleration": 0.02373314715106607,
            "data": 2.1763565235127884,
            "posture": 1.7332557679789737,
            "total": 5.923083902988496,
            "velocity": 1.989738464345667
          },
          {
            "acceleration": 0.02386723292463306,
            "data": 2.148435699955,
            "posture": 1.742370462577171,
            "total": 5.9227339818802225,
            "velocity": 2.0080605864234187
          },
          {
            "acceleration": 0.02393285077538316,
            "data": 2.1352277203681953,
            "posture": 1.7466835718993372,
            "total": 5.922691283083086,
            "velocity": 2.01684714004017
          },
          {
            "acceleration": 0.023964955551188105,
            "data": 2.1288173431215025,
            "posture": 1.748774105858731,
            "total": 5.922689136899927,
            "velocity": 2.021132732368506
          },
          {
            "acceleration": 0.02396495945456377,
            "data": 2.1288165692201018,
            "posture": 1.7487743579058639,
            "total": 5.922689137223537,
            "velocity": 2.021133250643007
          }
        ],
        "objective_terms": {
          "acceleration": 0.02396495945456377,
          "data": 2.1288165692201018,
          "posture": 1.7487743579058639,
          "total": 5.922689137223537,
          "velocity": 2.021133250643007
        },
        "orientation_rmse_deg": 57.026038393141334,
        "position_p95_mm": 33.1539635976545,
        "position_rmse_mm": 16.571446403691766,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9410634208840487,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9762972453555413,
        "wrist_target_height_median_mm": 126.17467929325093
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_2_20231022_202617_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1271,
      "metrics": {
        "achieved_tool_axis_down_fraction": 1.0,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 29,
        "max_joint_acceleration_rad_s2": 1.6365326654785393,
        "max_joint_velocity_rad_s": 0.6330719063243193,
        "min_collision_margin": 0.03928559967091515,
        "min_floor_margin_mm": 1.8104563195842083e-06,
        "min_joint_limit_margin_rad": 1.762625100474225,
        "min_urdf_velocity_limit_margin_rad_s": 1.6444495884084531,
        "min_velocity_limit_margin_rad_s": 1.6444495884084531,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.144594551841628,
        "objective_history": [
          {
            "acceleration": 0.0003007106850721743,
            "data": 40.746547021966464,
            "posture": 0.023512493438975473,
            "total": 40.77089319445451,
            "velocity": 0.0005329683639923975
          },
          {
            "acceleration": 4.712508343979526e-05,
            "data": 31.04666932663583,
            "posture": 0.08822447639599264,
            "total": 31.135395544067844,
            "velocity": 0.00045461595258100074
          },
          {
            "acceleration": 8.299934703080814e-05,
            "data": 22.93114538343665,
            "posture": 0.20188814093453297,
            "total": 23.135660734837614,
            "velocity": 0.002544211119404716
          },
          {
            "acceleration": 0.00018349139556003588,
            "data": 14.426172142307685,
            "posture": 0.2514297564560066,
            "total": 14.681244746045826,
            "velocity": 0.0034593558865729872
          },
          {
            "acceleration": 0.0010408027656811521,
            "data": 7.57468657924208,
            "posture": 0.3542133747449602,
            "total": 7.955954119506736,
            "velocity": 0.02601336275401511
          },
          {
            "acceleration": 0.0015897045345389087,
            "data": 5.450391120472373,
            "posture": 0.39719624593766406,
            "total": 5.891713731344306,
            "velocity": 0.042536660399729496
          },
          {
            "acceleration": 0.0025181293473471146,
            "data": 3.8832495738459576,
            "posture": 0.43241656933301065,
            "total": 4.391960741849318,
            "velocity": 0.07377646932300275
          },
          {
            "acceleration": 0.00353035901484902,
            "data": 2.8570112912448193,
            "posture": 0.46085567747394673,
            "total": 3.437890561249407,
            "velocity": 0.11649323351579181
          },
          {
            "acceleration": 0.004489921575771977,
            "data": 2.2438138318231244,
            "posture": 0.47895377722561716,
            "total": 2.885370259465991,
            "velocity": 0.15811272884147737
          },
          {
            "acceleration": 0.005369085170115905,
            "data": 1.84001228663203,
            "posture": 0.491581092633697,
            "total": 2.531809163479738,
            "velocity": 0.19484669904389504
          },
          {
            "acceleration": 0.006086712461997256,
            "data": 1.5150691978194804,
            "posture": 0.5026544655488728,
            "total": 2.250311066920959,
            "velocity": 0.22650069109060833
          },
          {
            "acceleration": 0.006739543309755332,
            "data": 1.2277494424530924,
            "posture": 0.5150563410600082,
            "total": 2.0029478005208396,
            "velocity": 0.2534024736979839
          },
          {
            "acceleration": 0.0070092068378164165,
            "data": 1.0921431269173412,
            "posture": 0.5215411115074466,
            "total": 1.8845914599499185,
            "velocity": 0.2638980146873143
          },
          {
            "acceleration": 0.0075468740678922945,
            "data": 0.8370367782342663,
            "posture": 0.5368017735094781,
            "total": 1.6663134308149836,
            "velocity": 0.28492800500334703
          },
          {
            "acceleration": 0.008525033650305215,
            "data": 0.46423989879279676,
            "posture": 0.5739249479546118,
            "total": 1.3730515254887479,
            "velocity": 0.3263616450910341
          },
          {
            "acceleration": 0.008524364813895043,
            "data": 0.2716445003101348,
            "posture": 0.6034141880310344,
            "total": 1.220586933467939,
            "velocity": 0.3370038803128745
          },
          {
            "acceleration": 0.008443509919930986,
            "data": 0.2285081435489637,
            "posture": 0.6090060494987457,
            "total": 1.1797981135302877,
            "velocity": 0.33384041056264746
          },
          {
            "acceleration": 0.008381242772946827,
            "data": 0.2071911839417484,
            "posture": 0.6132319310264386,
            "total": 1.1598774233082998,
            "velocity": 0.33107306556716604
          },
          {
            "acceleration": 0.008344804912826161,
            "data": 0.19725165267669936,
            "posture": 0.6162856501447075,
            "total": 1.150317703828002,
            "velocity": 0.3284355960937688
          },
          {
            "acceleration": 0.008336349985912104,
            "data": 0.19519698732326143,
            "posture": 0.61634020172198,
            "total": 1.1478497331448487,
            "velocity": 0.32797619411369516
          },
          {
            "acceleration": 0.008331269462484582,
            "data": 0.1942250616996642,
            "posture": 0.6163950540229585,
            "total": 1.14660690659265,
            "velocity": 0.32765552140754295
          },
          {
            "acceleration": 0.008326706923080747,
            "data": 0.193726428277268,
            "posture": 0.6164568101353666,
            "total": 1.1458927219035022,
            "velocity": 0.32738277656778675
          },
          {
            "acceleration": 0.008322271124122732,
            "data": 0.19336023802525978,
            "posture": 0.6165106150214678,
            "total": 1.1453835278412852,
            "velocity": 0.32719040367043495
          },
          {
            "acceleration": 0.008318797076102152,
            "data": 0.19324348653358264,
            "posture": 0.6165770013461609,
            "total": 1.1451174162753435,
            "velocity": 0.3269781313194977
          },
          {
            "acceleration": 0.008315450312442118,
            "data": 0.19315962613883061,
            "posture": 0.6166443605657687,
            "total": 1.1449096812381874,
            "velocity": 0.32679024422114605
          },
          {
            "acceleration": 0.00831234388761173,
            "data": 0.1930925925794007,
            "posture": 0.6167117191838496,
            "total": 1.1447388679015433,
            "velocity": 0.3266222122506812
          },
          {
            "acceleration": 0.008310921059565409,
            "data": 0.1930630005137132,
            "posture": 0.6167451153828707,
            "total": 1.1446654231969897,
            "velocity": 0.3265463862408405
          },
          {
            "acceleration": 0.008309564812085802,
            "data": 0.19303464181726288,
            "posture": 0.6167783004471535,
            "total": 1.1445965897314734,
            "velocity": 0.3264740826549713
          },
          {
            "acceleration": 0.008309524513811575,
            "data": 0.19303377256174895,
            "posture": 0.6167793300321377,
            "total": 1.144594551841628,
            "velocity": 0.3264719247339298
          }
        ],
        "objective_terms": {
          "acceleration": 0.008309524513811575,
          "data": 0.19303377256174895,
          "posture": 0.6167793300321377,
          "total": 1.144594551841628,
          "velocity": 0.3264719247339298
        },
        "orientation_rmse_deg": 11.05262453050294,
        "position_p95_mm": 19.27912904472018,
        "position_rmse_mm": 11.953507169562993,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9441384736428009,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 1.0,
        "wrist_target_height_median_mm": 103.82920157850687
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_2_20231023_163929_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1129,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9326837909654562,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 24,
        "max_joint_acceleration_rad_s2": 1.4565153231948416,
        "max_joint_velocity_rad_s": 0.6557672282766622,
        "min_collision_margin": 0.0393403709010085,
        "min_floor_margin_mm": 8.417781451053319e-06,
        "min_joint_limit_margin_rad": 1.9064126878829533,
        "min_urdf_velocity_limit_margin_rad_s": 1.7568845584977453,
        "min_velocity_limit_margin_rad_s": 1.7568845584977453,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.0162330431205313,
        "objective_history": [
          {
            "acceleration": 0.0003937828458976507,
            "data": 33.68025736075998,
            "posture": 0.02142829916450229,
            "total": 33.704015866737805,
            "velocity": 0.0019364239674207284
          },
          {
            "acceleration": 0.00013013791780311173,
            "data": 25.643770395486264,
            "posture": 0.0823082280978399,
            "total": 25.730973685099915,
            "velocity": 0.004764923598005719
          },
          {
            "acceleration": 0.00015182035831901145,
            "data": 19.00891789991895,
            "posture": 0.17947400241114686,
            "total": 19.1939450958377,
            "velocity": 0.005401373149286226
          },
          {
            "acceleration": 0.00025464730196866973,
            "data": 11.891600496329062,
            "posture": 0.2196816263615882,
            "total": 12.119473000288245,
            "velocity": 0.007936230295626916
          },
          {
            "acceleration": 0.0010548718201980024,
            "data": 5.935096195755905,
            "posture": 0.29631995465965416,
            "total": 6.271027930771512,
            "velocity": 0.0385569085357551
          },
          {
            "acceleration": 0.0013754739653988395,
            "data": 4.275913507908232,
            "posture": 0.3355413731013692,
            "total": 4.663836659501379,
            "velocity": 0.05100630452637856
          },
          {
            "acceleration": 0.001956222097365807,
            "data": 3.2873515402900946,
            "posture": 0.3698451075288023,
            "total": 3.7270181817095684,
            "velocity": 0.06786531179330633
          },
          {
            "acceleration": 0.0026803576914201355,
            "data": 2.755431004699198,
            "posture": 0.39057311375829057,
            "total": 3.238879037472904,
            "velocity": 0.0901945613239953
          },
          {
            "acceleration": 0.003335707097992426,
            "data": 2.423506826303907,
            "posture": 0.39853571883724903,
            "total": 2.9430448370591966,
            "velocity": 0.11766658482004806
          },
          {
            "acceleration": 0.0035836842423681366,
            "data": 2.2981261427657103,
            "posture": 0.4008244992031826,
            "total": 2.831453591669319,
            "velocity": 0.12891926545805846
          },
          {
            "acceleration": 0.00382857164076291,
            "data": 2.180067930386465,
            "posture": 0.40342500533103354,
            "total": 2.7267361362377533,
            "velocity": 0.13941462887949138
          },
          {
            "acceleration": 0.0040601530301493955,
            "data": 2.066047723204814,
            "posture": 0.4066174075494969,
            "total": 2.6257713967938496,
            "velocity": 0.1490461130093891
          },
          {
            "acceleration": 0.004282953509220889,
            "data": 1.9525804194484597,
            "posture": 0.4105271635722159,
            "total": 2.525262276366223,
            "velocity": 0.15787173983632646
          },
          {
            "acceleration": 0.00449782344435882,
            "data": 1.8387230574524291,
            "posture": 0.41512866187999253,
            "total": 2.424223082487149,
            "velocity": 0.1658735397103686
          },
          {
            "acceleration": 0.004921087233187311,
            "data": 1.6058907303845205,
            "posture": 0.42726137486457194,
            "total": 2.2193053143723707,
            "velocity": 0.18123212189009108
          },
          {
            "acceleration": 0.00581230171499904,
            "data": 1.135403975780937,
            "posture": 0.46191619845673026,
            "total": 1.813773761221542,
            "velocity": 0.21064128526887566
          },
          {
            "acceleration": 0.006602852446534079,
            "data": 0.689330956989429,
            "posture": 0.5023603865943201,
            "total": 1.4322721600385715,
            "velocity": 0.23397796400828816
          },
          {
            "acceleration": 0.007206774357460019,
            "data": 0.3634081125316095,
            "posture": 0.5446453949277712,
            "total": 1.1648080589627734,
            "velocity": 0.24954777714593274
          },
          {
            "acceleration": 0.007169769308590022,
            "data": 0.2677568077732687,
            "posture": 0.5606686222222044,
            "total": 1.0833069890466758,
            "velocity": 0.24771178974261268
          },
          {
            "acceleration": 0.007344846728310126,
            "data": 0.1769599013634677,
            "posture": 0.5893391888269677,
            "total": 1.0211733913331962,
            "velocity": 0.2475294544144506
          },
          {
            "acceleration": 0.007313834524998668,
            "data": 0.17353534101004042,
            "posture": 0.5903284481753012,
            "total": 1.0173247992373096,
            "velocity": 0.24614717552696924
          },
          {
            "acceleration": 0.007307418730722915,
            "data": 0.17277132018652996,
            "posture": 0.5905615959467111,
            "total": 1.0164862993888832,
            "velocity": 0.24584596452491927
          },
          {
            "acceleration": 0.007305892656351013,
            "data": 0.17258430925052443,
            "posture": 0.5906194693172829,
            "total": 1.0162833719735478,
            "velocity": 0.24577370074938945
          },
          {
            "acceleration": 0.007305515748694448,
            "data": 0.17253780852230005,
            "posture": 0.5906339086268707,
            "total": 1.0162330431205313,
            "velocity": 0.2457558102226661
          }
        ],
        "objective_terms": {
          "acceleration": 0.007305515748694448,
          "data": 0.17253780852230005,
          "posture": 0.5906339086268707,
          "total": 1.0162330431205313,
          "velocity": 0.2457558102226661
        },
        "orientation_rmse_deg": 11.836882825971292,
        "position_p95_mm": 20.95798220869555,
        "position_rmse_mm": 11.072810172907305,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9149689991142604,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.933569530558016,
        "wrist_target_height_median_mm": 128.57619250510825
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_154531_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1686,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.7841043890865955,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 25,
        "max_joint_acceleration_rad_s2": 2.0409645634458524,
        "max_joint_velocity_rad_s": 0.8489505654253384,
        "min_collision_margin": 0.03981393495609306,
        "min_floor_margin_mm": 3.169825486815103e-05,
        "min_joint_limit_margin_rad": 1.823013756205625,
        "min_urdf_velocity_limit_margin_rad_s": 1.5678852207524683,
        "min_velocity_limit_margin_rad_s": 1.5678852207524683,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.6001450440804583,
        "objective_history": [
          {
            "acceleration": 0.0003419951280507359,
            "data": 58.09695278100874,
            "posture": 0.03269044063758692,
            "total": 58.13350604999127,
            "velocity": 0.003520833216888271
          },
          {
            "acceleration": 0.00035163174442217525,
            "data": 45.33221121725293,
            "posture": 0.12974640189610054,
            "total": 45.47605340406927,
            "velocity": 0.013744153175816868
          },
          {
            "acceleration": 0.0005270325449924652,
            "data": 34.59519867247577,
            "posture": 0.2886695985792252,
            "total": 34.907745556139716,
            "velocity": 0.023350252539725138
          },
          {
            "acceleration": 0.0005687324843383523,
            "data": 23.329207762727354,
            "posture": 0.3644034617367218,
            "total": 23.724340498700723,
            "velocity": 0.03016054175230786
          },
          {
            "acceleration": 0.0008428417128943376,
            "data": 12.887797386218615,
            "posture": 0.5048105648831556,
            "total": 13.430690639757245,
            "velocity": 0.03723984694257925
          },
          {
            "acceleration": 0.0012917951275457772,
            "data": 8.682794382304778,
            "posture": 0.5548783341359075,
            "total": 9.287079762767695,
            "velocity": 0.04811525119946471
          },
          {
            "acceleration": 0.002459280853011151,
            "data": 5.857723738627039,
            "posture": 0.594858169920557,
            "total": 6.534978438963885,
            "velocity": 0.07993724956327902
          },
          {
            "acceleration": 0.0040422815812907065,
            "data": 4.034218895794458,
            "posture": 0.6258342481323608,
            "total": 4.796988823800714,
            "velocity": 0.1328933982926031
          },
          {
            "acceleration": 0.005715183088820843,
            "data": 3.0019014349072495,
            "posture": 0.6429254995173217,
            "total": 3.8505408584430576,
            "velocity": 0.19999874092966555
          },
          {
            "acceleration": 0.00730617865335544,
            "data": 2.409636537773572,
            "posture": 0.6536183655793499,
            "total": 3.338776637325136,
            "velocity": 0.26821555531885893
          },
          {
            "acceleration": 0.008615026104066203,
            "data": 1.9955522778792647,
            "posture": 0.6632218979187469,
            "total": 2.9939336638075336,
            "velocity": 0.32654446190545566
          },
          {
            "acceleration": 0.01113439147401237,
            "data": 1.3274270630980824,
            "posture": 0.6932700308058554,
            "total": 2.467382497880698,
            "velocity": 0.43555101250274825
          },
          {
            "acceleration": 0.01284371117619186,
            "data": 0.8121111845836004,
            "posture": 0.7288737535166274,
            "total": 2.05136080235733,
            "velocity": 0.49753215308091053
          },
          {
            "acceleration": 0.013415679678620432,
            "data": 0.46202232057204207,
            "posture": 0.7608612149973311,
            "total": 1.759613115395001,
            "velocity": 0.5233139001470074
          },
          {
            "acceleration": 0.01350647249029045,
            "data": 0.2987283008875605,
            "posture": 0.7894984032614681,
            "total": 1.6428278384039476,
            "velocity": 0.5410946617646285
          },
          {
            "acceleration": 0.013465483572786104,
            "data": 0.267451292099841,
            "posture": 0.794371778450867,
            "total": 1.6186815553838523,
            "velocity": 0.543393001260358
          },
          {
            "acceleration": 0.013465192444458929,
            "data": 0.25427982140983985,
            "posture": 0.7963890535853773,
            "total": 1.6087461196747117,
            "velocity": 0.5446120522350356
          },
          {
            "acceleration": 0.013469053837397322,
            "data": 0.24934752669121912,
            "posture": 0.7965640429844617,
            "total": 1.6041627542358294,
            "velocity": 0.5447821307227514
          },
          {
            "acceleration": 0.01347267842555978,
            "data": 0.24691830734876086,
            "posture": 0.7967157759597409,
            "total": 1.6020209126279932,
            "velocity": 0.5449141508939317
          },
          {
            "acceleration": 0.013475117332413366,
            "data": 0.24565069668469147,
            "posture": 0.7968550777363315,
            "total": 1.6010089775791942,
            "velocity": 0.5450280858257579
          },
          {
            "acceleration": 0.013476485842938524,
            "data": 0.24492446248433417,
            "posture": 0.7969857284011647,
            "total": 1.6005173129059287,
            "velocity": 0.5451306361774914
          },
          {
            "acceleration": 0.013477281924322616,
            "data": 0.2444719882520294,
            "posture": 0.7971088427478404,
            "total": 1.600283298994735,
            "velocity": 0.5452251860705426
          },
          {
            "acceleration": 0.013477740564070454,
            "data": 0.2441650149336889,
            "posture": 0.7972248757451373,
            "total": 1.6001803235087204,
            "velocity": 0.5453126922658237
          },
          {
            "acceleration": 0.013478004848766947,
            "data": 0.2439392966799893,
            "posture": 0.7973341844420065,
            "total": 1.6001454409686926,
            "velocity": 0.5453939549979299
          },
          {
            "acceleration": 0.013478111553736483,
            "data": 0.24384971080028808,
            "posture": 0.7973856633102999,
            "total": 1.6001450440804583,
            "velocity": 0.5454315584161339
          }
        ],
        "objective_terms": {
          "acceleration": 0.013478111553736483,
          "data": 0.24384971080028808,
          "posture": 0.7973856633102999,
          "total": 1.6001450440804583,
          "velocity": 0.5454315584161339
        },
        "orientation_rmse_deg": 10.753640477476901,
        "position_p95_mm": 20.838259797732405,
        "position_rmse_mm": 12.255022439339356,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7313167259786477,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.7846975088967971,
        "wrist_target_height_median_mm": 89.49722257178786
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_155008_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1046,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8690248565965584,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 22,
        "max_joint_acceleration_rad_s2": 3.430343082318238,
        "max_joint_velocity_rad_s": 1.1146515789265832,
        "min_collision_margin": 0.04025215141827089,
        "min_floor_margin_mm": 0.003187534270303327,
        "min_joint_limit_margin_rad": 1.826099978012889,
        "min_urdf_velocity_limit_margin_rad_s": 1.47624178343476,
        "min_velocity_limit_margin_rad_s": 1.47624178343476,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.3379543822841977,
        "objective_history": [
          {
            "acceleration": 0.0005104284338871407,
            "data": 37.43197265541492,
            "posture": 0.02197646832159169,
            "total": 37.45859886879833,
            "velocity": 0.0041393166279415925
          },
          {
            "acceleration": 0.000655824087085446,
            "data": 29.029134507017428,
            "posture": 0.08658830944513164,
            "total": 29.13066203541955,
            "velocity": 0.014283394869902554
          },
          {
            "acceleration": 0.0007622912574232577,
            "data": 22.008696433082424,
            "posture": 0.17765218293031212,
            "total": 22.20653846193231,
            "velocity": 0.019427554662150057
          },
          {
            "acceleration": 0.001033961193172213,
            "data": 14.662114345115512,
            "posture": 0.2234867220565475,
            "total": 14.91190412771174,
            "velocity": 0.025269099346507517
          },
          {
            "acceleration": 0.0017816764187703412,
            "data": 8.02856710198014,
            "posture": 0.31338690316038537,
            "total": 8.382834242058774,
            "velocity": 0.03909856049947755
          },
          {
            "acceleration": 0.00624019278972574,
            "data": 3.632673663919378,
            "posture": 0.39160964130041176,
            "total": 4.114899714910782,
            "velocity": 0.08437621690126729
          },
          {
            "acceleration": 0.008846182972152412,
            "data": 2.5196449318348835,
            "posture": 0.4085663230091961,
            "total": 3.0658360213741314,
            "velocity": 0.12877858355789962
          },
          {
            "acceleration": 0.011826530976714461,
            "data": 1.8509666929071336,
            "posture": 0.4220253435182386,
            "total": 2.480144442132823,
            "velocity": 0.1953258747307364
          },
          {
            "acceleration": 0.01317412507820202,
            "data": 1.6557212553392788,
            "posture": 0.4241247150927218,
            "total": 2.3204925364782345,
            "velocity": 0.2274724409680318
          },
          {
            "acceleration": 0.01590698915564598,
            "data": 1.3573436276588158,
            "posture": 0.42644442906447033,
            "total": 2.094247304470018,
            "velocity": 0.29455225859108586
          },
          {
            "acceleration": 0.01697976363894377,
            "data": 1.2575128555353228,
            "posture": 0.42655504585134923,
            "total": 2.0209836458533914,
            "velocity": 0.3199359808277754
          },
          {
            "acceleration": 0.01902644773532883,
            "data": 1.0804102111777394,
            "posture": 0.42804246917196737,
            "total": 1.8958577146846536,
            "velocity": 0.3683785865996181
          },
          {
            "acceleration": 0.020631217594259538,
            "data": 0.9281126705067824,
            "posture": 0.43042325588060415,
            "total": 1.7852801555855433,
            "velocity": 0.40611301160389707
          },
          {
            "acceleration": 0.023554555274133603,
            "data": 0.6598540488246066,
            "posture": 0.4403414134317551,
            "total": 1.5982938032606429,
            "velocity": 0.47454378573014744
          },
          {
            "acceleration": 0.025339411198170916,
            "data": 0.4432512342588838,
            "posture": 0.4543667116312701,
            "total": 1.4361836794179152,
            "velocity": 0.5132263223295903
          },
          {
            "acceleration": 0.026222892510473723,
            "data": 0.3093522686226251,
            "posture": 0.4700614671831985,
            "total": 1.364892620631458,
            "velocity": 0.5592559923151608
          },
          {
            "acceleration": 0.02612053374147881,
            "data": 0.2822642844220899,
            "posture": 0.47374006827628995,
            "total": 1.3485287237541175,
            "velocity": 0.5664038373142589
          },
          {
            "acceleration": 0.026118644591097362,
            "data": 0.27022366077919796,
            "posture": 0.47563656849285735,
            "total": 1.342936227683737,
            "velocity": 0.5709573538205844
          },
          {
            "acceleration": 0.026131838579154857,
            "data": 0.2668209626745977,
            "posture": 0.47576740889226904,
            "total": 1.340075480322802,
            "velocity": 0.5713552701767806
          },
          {
            "acceleration": 0.026139343006987063,
            "data": 0.26507060794786186,
            "posture": 0.4758745481120082,
            "total": 1.3387445630867332,
            "velocity": 0.571660064019876
          },
          {
            "acceleration": 0.02613865370973919,
            "data": 0.2640403096090631,
            "posture": 0.47596053755317047,
            "total": 1.3380357028534777,
            "velocity": 0.5718962019815048
          },
          {
            "acceleration": 0.02613816135736219,
            "data": 0.2638848313949468,
            "posture": 0.47598109936501615,
            "total": 1.3379543822841977,
            "velocity": 0.5719502901668726
          }
        ],
        "objective_terms": {
          "acceleration": 0.02613816135736219,
          "data": 0.2638848313949468,
          "posture": 0.47598109936501615,
          "total": 1.3379543822841977,
          "velocity": 0.5719502901668726
        },
        "orientation_rmse_deg": 13.344901208419744,
        "position_p95_mm": 28.508854768083992,
        "position_rmse_mm": 16.10968494807964,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.8212237093690249,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8652007648183556,
        "wrist_target_height_median_mm": 59.67377454171506
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_161209_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1071,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8608776844070962,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 23,
        "max_joint_acceleration_rad_s2": 3.3470089717256233,
        "max_joint_velocity_rad_s": 1.0999136949776185,
        "min_collision_margin": 0.03996908085420496,
        "min_floor_margin_mm": 9.126962063898958e-05,
        "min_joint_limit_margin_rad": 1.7481054438303243,
        "min_urdf_velocity_limit_margin_rad_s": 1.5281597952252883,
        "min_velocity_limit_margin_rad_s": 1.5281597952252883,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.2367417823208795,
        "objective_history": [
          {
            "acceleration": 0.0002907106600409566,
            "data": 32.66803383210995,
            "posture": 0.0203845821044221,
            "total": 32.69120648009229,
            "velocity": 0.002497355217883247
          },
          {
            "acceleration": 0.0005779722183807565,
            "data": 25.27261570137582,
            "posture": 0.08360739257267184,
            "total": 25.369285132112125,
            "velocity": 0.01248406594525656
          },
          {
            "acceleration": 0.0007567894946163547,
            "data": 19.090647338985455,
            "posture": 0.17801474112871363,
            "total": 19.289750919688316,
            "velocity": 0.020332050079530747
          },
          {
            "acceleration": 0.0010631643062969133,
            "data": 12.703763492257554,
            "posture": 0.22523119549608397,
            "total": 12.959735244492466,
            "velocity": 0.02967739243253012
          },
          {
            "acceleration": 0.0019823444812200612,
            "data": 6.733913473443621,
            "posture": 0.31373544926649627,
            "total": 7.103591405134611,
            "velocity": 0.05396013794327368
          },
          {
            "acceleration": 0.003048344793237709,
            "data": 4.595750574386681,
            "posture": 0.3517468907328033,
            "total": 5.025612364693137,
            "velocity": 0.07506655478041437
          },
          {
            "acceleration": 0.0049656423762162,
            "data": 3.1934959806918117,
            "posture": 0.3849017811838869,
            "total": 3.6954621852763285,
            "velocity": 0.1120987810244135
          },
          {
            "acceleration": 0.007388577894912381,
            "data": 2.3892688661246373,
            "posture": 0.4066274844886447,
            "total": 2.965946588935211,
            "velocity": 0.16266166042701674
          },
          {
            "acceleration": 0.009669986579182254,
            "data": 1.884446146610773,
            "posture": 0.4202495100166256,
            "total": 2.530317039897985,
            "velocity": 0.2159513966914045
          },
          {
            "acceleration": 0.011593620519492974,
            "data": 1.5628560170447225,
            "posture": 0.4277329470365155,
            "total": 2.2654403667238068,
            "velocity": 0.26325778212307605
          },
          {
            "acceleration": 0.013298067077957143,
            "data": 1.3332993309992935,
            "posture": 0.43217411051369803,
            "total": 2.080634447660849,
            "velocity": 0.30186293906990025
          },
          {
            "acceleration": 0.01400275338481117,
            "data": 1.2378229890599293,
            "posture": 0.4343353791377219,
            "total": 2.0039311186082656,
            "velocity": 0.3177699970258031
          },
          {
            "acceleration": 0.015388819731311051,
            "data": 1.055437194678769,
            "posture": 0.4403369714870282,
            "total": 1.8611388795077481,
            "velocity": 0.34997589361063997
          },
          {
            "acceleration": 0.018087600679451053,
            "data": 0.7286156479813681,
            "posture": 0.45816509036354375,
            "total": 1.6174418755235198,
            "velocity": 0.4125735364991569
          },
          {
            "acceleration": 0.019718673103536154,
            "data": 0.46534047073145823,
            "posture": 0.4780335716600329,
            "total": 1.4128672091296028,
            "velocity": 0.44977449363457545
          },
          {
            "acceleration": 0.020274974666551423,
            "data": 0.30031525272886006,
            "posture": 0.4968249219332178,
            "total": 1.289390927339804,
            "velocity": 0.47197577801117496
          },
          {
            "acceleration": 0.02020401718219009,
            "data": 0.23008960622902602,
            "posture": 0.5113718510675376,
            "total": 1.2521224680845413,
            "velocity": 0.49045699360578754
          },
          {
            "acceleration": 0.020239226287592164,
            "data": 0.22118804839569534,
            "posture": 0.5115708146739781,
            "total": 1.2434933256449034,
            "velocity": 0.49049523628763786
          },
          {
            "acceleration": 0.020259684871830275,
            "data": 0.21707622877655633,
            "posture": 0.5116644894636743,
            "total": 1.2395737323017046,
            "velocity": 0.49057332918964386
          },
          {
            "acceleration": 0.020265267059252242,
            "data": 0.2160610655375883,
            "posture": 0.5117008363501433,
            "total": 1.2386205301173776,
            "velocity": 0.4905933611703937
          },
          {
            "acceleration": 0.02027250966303362,
            "data": 0.2145326651754238,
            "posture": 0.511781028647269,
            "total": 1.2372044060929757,
            "velocity": 0.49061820260724914
          },
          {
            "acceleration": 0.020274718746153643,
            "data": 0.21415818496366373,
            "posture": 0.5118123163754577,
            "total": 1.2368688511049397,
            "velocity": 0.49062363101966483
          },
          {
            "acceleration": 0.020275427753417216,
            "data": 0.21401291860042698,
            "posture": 0.5118276696973206,
            "total": 1.2367417823208795,
            "velocity": 0.49062576626971466
          }
        ],
        "objective_terms": {
          "acceleration": 0.020275427753417216,
          "data": 0.21401291860042698,
          "posture": 0.5118276696973206,
          "total": 1.2367417823208795,
          "velocity": 0.49062576626971466
        },
        "orientation_rmse_deg": 12.94931587078539,
        "position_p95_mm": 24.148670742602146,
        "position_rmse_mm": 14.862035015122792,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7871148459383753,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8590102707749766,
        "wrist_target_height_median_mm": 81.27266892113694
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_161937_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1269,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9078014184397163,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 24,
        "max_joint_acceleration_rad_s2": 2.3756812952372606,
        "max_joint_velocity_rad_s": 0.8072386563031313,
        "min_collision_margin": 0.04019510166341676,
        "min_floor_margin_mm": 4.452109735042509e-05,
        "min_joint_limit_margin_rad": 1.7630074322954319,
        "min_urdf_velocity_limit_margin_rad_s": 1.5564359685670168,
        "min_velocity_limit_margin_rad_s": 1.5564359685670168,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.2249311578037734,
        "objective_history": [
          {
            "acceleration": 0.0005967594050576205,
            "data": 38.65529136469881,
            "posture": 0.026017899811172477,
            "total": 38.68809221913341,
            "velocity": 0.006186195218368321
          },
          {
            "acceleration": 0.0005068367382902445,
            "data": 29.926558329713515,
            "posture": 0.10307153003947601,
            "total": 30.046964911916948,
            "velocity": 0.01682821542566693
          },
          {
            "acceleration": 0.0007159453069055623,
            "data": 22.652040589692064,
            "posture": 0.21363845613802965,
            "total": 22.88921527150661,
            "velocity": 0.022820280369615555
          },
          {
            "acceleration": 0.0007910621867177606,
            "data": 15.04624274163192,
            "posture": 0.26999079091218375,
            "total": 15.348742544428415,
            "velocity": 0.03171794969759346
          },
          {
            "acceleration": 0.001615088272872038,
            "data": 7.987752067517194,
            "posture": 0.3727171434489262,
            "total": 8.415322487547824,
            "velocity": 0.05323818830883164
          },
          {
            "acceleration": 0.004865267973300857,
            "data": 3.410842130607399,
            "posture": 0.4717401355875093,
            "total": 3.9900424938954293,
            "velocity": 0.10259495972722017
          },
          {
            "acceleration": 0.006353590380296919,
            "data": 2.535530207047229,
            "posture": 0.49232867573655614,
            "total": 3.159330691767516,
            "velocity": 0.12511821860343375
          },
          {
            "acceleration": 0.008102915182074238,
            "data": 1.9777685657649346,
            "posture": 0.5036858216459288,
            "total": 2.647359513559259,
            "velocity": 0.1578022109663217
          },
          {
            "acceleration": 0.009744276255664533,
            "data": 1.6118122632503744,
            "posture": 0.5109801877819804,
            "total": 2.3242935817675727,
            "velocity": 0.1917568544795531
          },
          {
            "acceleration": 0.011196600398287314,
            "data": 1.3421709776614184,
            "posture": 0.5191536343324423,
            "total": 2.095596006811369,
            "velocity": 0.22307479441922065
          },
          {
            "acceleration": 0.01241077362026353,
            "data": 1.1131334762680536,
            "posture": 0.5289417293136348,
            "total": 1.9066834328809517,
            "velocity": 0.25219745367899965
          },
          {
            "acceleration": 0.014787485713101296,
            "data": 0.7264995324563333,
            "posture": 0.5541488602528262,
            "total": 1.6122660690204398,
            "velocity": 0.31683019059817896
          },
          {
            "acceleration": 0.015832957646556702,
            "data": 0.4328852452002921,
            "posture": 0.5785685662079273,
            "total": 1.3892458075196903,
            "velocity": 0.3619590384649143
          },
          {
            "acceleration": 0.016325494687935985,
            "data": 0.26301381023911347,
            "posture": 0.5994924624347533,
            "total": 1.2696749968796468,
            "velocity": 0.39084322951784406
          },
          {
            "acceleration": 0.01634470061805034,
            "data": 0.1962797016587337,
            "posture": 0.6126719256818127,
            "total": 1.2369821780454053,
            "velocity": 0.4116858500868086
          },
          {
            "acceleration": 0.016326288750780564,
            "data": 0.18955779262057723,
            "posture": 0.6128284068879721,
            "total": 1.2305901542640456,
            "velocity": 0.4118776660047157
          },
          {
            "acceleration": 0.01631815474928457,
            "data": 0.18631991510836443,
            "posture": 0.6129349718208892,
            "total": 1.2276220295650566,
            "velocity": 0.4120489878865184
          },
          {
            "acceleration": 0.01631389312665205,
            "data": 0.18475423589265622,
            "posture": 0.612990406206576,
            "total": 1.2262158843327913,
            "velocity": 0.4121573491069069
          },
          {
            "acceleration": 0.016311455833753628,
            "data": 0.18392321261895395,
            "posture": 0.613039516753609,
            "total": 1.225535254292093,
            "velocity": 0.4122610690857763
          },
          {
            "acceleration": 0.016310009227798128,
            "data": 0.18343930090431115,
            "posture": 0.6130849845475707,
            "total": 1.2251967873620955,
            "velocity": 0.4123624926824156
          },
          {
            "acceleration": 0.01630923210323755,
            "data": 0.18312914832127175,
            "posture": 0.6131281572278724,
            "total": 1.225029387783064,
            "velocity": 0.4124628501306824
          },
          {
            "acceleration": 0.0163092572956947,
            "data": 0.18291396048982717,
            "posture": 0.6131686485985314,
            "total": 1.224952079062593,
            "velocity": 0.41256021267853976
          },
          {
            "acceleration": 0.016309511688919393,
            "data": 0.1828348911476948,
            "posture": 0.6131871132406864,
            "total": 1.2249361242174204,
            "velocity": 0.41260460814011973
          },
          {
            "acceleration": 0.016309663941351846,
            "data": 0.1827994408280097,
            "posture": 0.6131960519008824,
            "total": 1.2249311578037734,
            "velocity": 0.41262600113352943
          }
        ],
        "objective_terms": {
          "acceleration": 0.016309663941351846,
          "data": 0.1827994408280097,
          "posture": 0.6131960519008824,
          "total": 1.2249311578037734,
          "velocity": 0.41262600113352943
        },
        "orientation_rmse_deg": 10.969658376518641,
        "position_p95_mm": 21.22836751848207,
        "position_rmse_mm": 12.221345518921876,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.8754925137903862,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9078014184397163,
        "wrist_target_height_median_mm": 94.61415000376616
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_162327_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 878,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8207762557077626,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 19,
        "max_joint_acceleration_rad_s2": 3.041519471174866,
        "max_joint_velocity_rad_s": 1.0393911729827612,
        "min_collision_margin": 0.04040500130722781,
        "min_floor_margin_mm": 0.00035094688227016446,
        "min_joint_limit_margin_rad": 1.8632731120272281,
        "min_urdf_velocity_limit_margin_rad_s": 1.4682555086945537,
        "min_velocity_limit_margin_rad_s": 1.4682555086945537,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.0858931500862443,
        "objective_history": [
          {
            "acceleration": 0.0004230317462999378,
            "data": 27.299045644753036,
            "posture": 0.01946653557218338,
            "total": 27.320937588351992,
            "velocity": 0.002002376280473988
          },
          {
            "acceleration": 0.0004715726955336115,
            "data": 21.072234115601557,
            "posture": 0.07471197775929758,
            "total": 21.157904179815358,
            "velocity": 0.010486513758969706
          },
          {
            "acceleration": 0.0005890055403004211,
            "data": 15.867959883956544,
            "posture": 0.14990572300482718,
            "total": 16.03091400772794,
            "velocity": 0.01245939522626829
          },
          {
            "acceleration": 0.00073275062624006,
            "data": 10.38991306364607,
            "posture": 0.18835862651840563,
            "total": 10.595861638287507,
            "velocity": 0.01685719749679197
          },
          {
            "acceleration": 0.0015618291718828556,
            "data": 5.33405537651838,
            "posture": 0.2614936152918032,
            "total": 5.6356068836312945,
            "velocity": 0.038496062649228203
          },
          {
            "acceleration": 0.005430054455959524,
            "data": 2.0980692243405628,
            "posture": 0.31472624256988874,
            "total": 2.520336594230373,
            "velocity": 0.10211107286396179
          },
          {
            "acceleration": 0.007570587886586513,
            "data": 1.5713162388895916,
            "posture": 0.32630254300534456,
            "total": 2.058347792533162,
            "velocity": 0.15315842275163963
          },
          {
            "acceleration": 0.009808800853835644,
            "data": 1.239112357852435,
            "posture": 0.3346116445356938,
            "total": 1.7884940930958777,
            "velocity": 0.20496128985391313
          },
          {
            "acceleration": 0.01178155376486975,
            "data": 1.0209460068555822,
            "posture": 0.340858457694114,
            "total": 1.6204271032209807,
            "velocity": 0.24684108490641476
          },
          {
            "acceleration": 0.013415476747650572,
            "data": 0.8574913981777995,
            "posture": 0.3457808860747947,
            "total": 1.4981062076854157,
            "velocity": 0.28141844668517085
          },
          {
            "acceleration": 0.016384349628479424,
            "data": 0.5967111314361937,
            "posture": 0.3578435320632724,
            "total": 1.3176344920359002,
            "velocity": 0.3466954789079547
          },
          {
            "acceleration": 0.018167014292357817,
            "data": 0.400056986388468,
            "posture": 0.37178837444774027,
            "total": 1.1795700970554552,
            "velocity": 0.38955772192688903
          },
          {
            "acceleration": 0.019154058079942415,
            "data": 0.27077255948915746,
            "posture": 0.385691517660612,
            "total": 1.1061149499299519,
            "velocity": 0.43049681470024
          },
          {
            "acceleration": 0.019240514505441817,
            "data": 0.21211819631210377,
            "posture": 0.39549752326401166,
            "total": 1.0944263153148326,
            "velocity": 0.4675700812332753
          },
          {
            "acceleration": 0.01921573090778219,
            "data": 0.20771024921716205,
            "posture": 0.3955210557126944,
            "total": 1.0900055730387688,
            "velocity": 0.46755853720113005
          },
          {
            "acceleration": 0.01920157341828428,
            "data": 0.20563890367208978,
            "posture": 0.3955431809894478,
            "total": 1.0879285637230391,
            "velocity": 0.46754490564321716
          },
          {
            "acceleration": 0.019190747021133815,
            "data": 0.2046262205040486,
            "posture": 0.3955662636083071,
            "total": 1.0869083533406223,
            "velocity": 0.4675251222071327
          },
          {
            "acceleration": 0.019173316158947237,
            "data": 0.20360691100294465,
            "posture": 0.3956159903716795,
            "total": 1.0858977911085674,
            "velocity": 0.467501573574996
          },
          {
            "acceleration": 0.019172605325663793,
            "data": 0.20359888653287048,
            "posture": 0.39561947588874335,
            "total": 1.0858931500862443,
            "velocity": 0.4675021823389666
          }
        ],
        "objective_terms": {
          "acceleration": 0.019172605325663793,
          "data": 0.20359888653287048,
          "posture": 0.39561947588874335,
          "total": 1.0858931500862443,
          "velocity": 0.4675021823389666
        },
        "orientation_rmse_deg": 14.268061282229821,
        "position_p95_mm": 27.3600620264305,
        "position_rmse_mm": 15.393633688323805,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7294520547945206,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8211845102505695,
        "wrist_target_height_median_mm": 71.17816357694431
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_3_20231024_162756_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 946,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9344608879492601,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 24,
        "max_joint_acceleration_rad_s2": 3.331881989109407,
        "max_joint_velocity_rad_s": 0.9977525225011116,
        "min_collision_margin": 0.039718591111389534,
        "min_floor_margin_mm": 2.486684573532738e-06,
        "min_joint_limit_margin_rad": 1.74946682738302,
        "min_urdf_velocity_limit_margin_rad_s": 1.573924957721985,
        "min_velocity_limit_margin_rad_s": 1.573924957721985,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.0843118134417258,
        "objective_history": [
          {
            "acceleration": 0.00045307508376413874,
            "data": 31.337257221688745,
            "posture": 0.01838346748275328,
            "total": 31.359669034080532,
            "velocity": 0.0035752698252681423
          },
          {
            "acceleration": 0.00043754596032741526,
            "data": 24.61835343071549,
            "posture": 0.07339057167786286,
            "total": 24.70298454792147,
            "velocity": 0.010802999567788985
          },
          {
            "acceleration": 0.0005828692008632164,
            "data": 18.98180662853172,
            "posture": 0.15429531770631863,
            "total": 19.148687598264793,
            "velocity": 0.012002782825890995
          },
          {
            "acceleration": 0.0008453143716391343,
            "data": 13.08377292470259,
            "posture": 0.19242383202750868,
            "total": 13.295743825470007,
            "velocity": 0.018701754368269548
          },
          {
            "acceleration": 0.0014912555773783445,
            "data": 7.415516695261245,
            "posture": 0.27675131229871347,
            "total": 7.736908757365094,
            "velocity": 0.043149494227756664
          },
          {
            "acceleration": 0.0021250109986100103,
            "data": 5.103862523730423,
            "posture": 0.3165181001532495,
            "total": 5.478727697963037,
            "velocity": 0.05622206308075422
          },
          {
            "acceleration": 0.003687969013861701,
            "data": 3.5740033372818267,
            "posture": 0.35226904166681083,
            "total": 4.010268819971336,
            "velocity": 0.0803084720088367
          },
          {
            "acceleration": 0.006056727363162441,
            "data": 2.6327177283728362,
            "posture": 0.3766360100681782,
            "total": 3.1345109457597458,
            "velocity": 0.11910047995556847
          },
          {
            "acceleration": 0.008610495269979979,
            "data": 2.025963459931682,
            "posture": 0.39107004294600506,
            "total": 2.592350398686241,
            "velocity": 0.16670640053857408
          },
          {
            "acceleration": 0.010949582687521015,
            "data": 1.6319342771450396,
            "posture": 0.3993368146178413,
            "total": 2.256882536832482,
            "velocity": 0.21466186238208018
          },
          {
            "acceleration": 0.01310741708538108,
            "data": 1.36785928357526,
            "posture": 0.4051835338049654,
            "total": 2.0406499121248727,
            "velocity": 0.25449967765926595
          },
          {
            "acceleration": 0.014904345002313685,
            "data": 1.152929993945743,
            "posture": 0.4113709809203434,
            "total": 1.8656823221632328,
            "velocity": 0.2864770022948328
          },
          {
            "acceleration": 0.017979887332885563,
            "data": 0.7885490939420527,
            "posture": 0.4292579240060466,
            "total": 1.581055599773845,
            "velocity": 0.34526869449286
          },
          {
            "acceleration": 0.01922466799357107,
            "data": 0.5082586608147388,
            "posture": 0.4480606924242072,
            "total": 1.3372563934293114,
            "velocity": 0.3617123721967942
          },
          {
            "acceleration": 0.019922916985574395,
            "data": 0.30450567322556243,
            "posture": 0.46697072200398054,
            "total": 1.1727906312288796,
            "velocity": 0.38139131901376233
          },
          {
            "acceleration": 0.020002324705441617,
            "data": 0.20049622814925894,
            "posture": 0.48190656441726,
            "total": 1.107609746606804,
            "velocity": 0.4052046293348436
          },
          {
            "acceleration": 0.0198863676574971,
            "data": 0.17932934122026944,
            "posture": 0.4849456367939659,
            "total": 1.094156451677063,
            "velocity": 0.4099951060053306
          },
          {
            "acceleration": 0.019867880093975224,
            "data": 0.1735022297194474,
            "posture": 0.48513223932187177,
            "total": 1.0888776362839208,
            "velocity": 0.4103752871486263
          },
          {
            "acceleration": 0.01985686712419432,
            "data": 0.17047808026283073,
            "posture": 0.485304599464214,
            "total": 1.0863937517523161,
            "velocity": 0.4107542049010771
          },
          {
            "acceleration": 0.019847261743793143,
            "data": 0.1686905762264932,
            "posture": 0.4854809365078122,
            "total": 1.0851714528330016,
            "velocity": 0.41115267835490316
          },
          {
            "acceleration": 0.019842764185569895,
            "data": 0.16768414657141345,
            "posture": 0.4856218754846433,
            "total": 1.0846387414397156,
            "velocity": 0.41148995519808895
          },
          {
            "acceleration": 0.019840781710668713,
            "data": 0.16699288095255035,
            "posture": 0.48575390657486317,
            "total": 1.084400641867574,
            "velocity": 0.41181307262949174
          },
          {
            "acceleration": 0.019840145166694082,
            "data": 0.16647142061929168,
            "posture": 0.4858790216908185,
            "total": 1.0843144653086627,
            "velocity": 0.4121238778318585
          },
          {
            "acceleration": 0.019840250435567607,
            "data": 0.166364324855929,
            "posture": 0.48590881138045444,
            "total": 1.0843118134417258,
            "velocity": 0.41219842676977475
          }
        ],
        "objective_terms": {
          "acceleration": 0.019840250435567607,
          "data": 0.166364324855929,
          "posture": 0.48590881138045444,
          "total": 1.0843118134417258,
          "velocity": 0.41219842676977475
        },
        "orientation_rmse_deg": 11.307717125869242,
        "position_p95_mm": 23.98086179412931,
        "position_rmse_mm": 14.10371620062099,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9090909090909091,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9355179704016914,
        "wrist_target_height_median_mm": 107.66848923048245
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_4_20231026_163223_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 939,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.6666666666666666,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 17,
        "max_joint_acceleration_rad_s2": 2.02568088527905,
        "max_joint_velocity_rad_s": 0.7716570843895408,
        "min_collision_margin": 0.04005039101153267,
        "min_floor_margin_mm": 3.0872997908160116e-07,
        "min_joint_limit_margin_rad": 1.9867496978979042,
        "min_urdf_velocity_limit_margin_rad_s": 1.527398541604769,
        "min_velocity_limit_margin_rad_s": 1.527398541604769,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 0.8384540226664143,
        "objective_history": [
          {
            "acceleration": 0.000249305442422829,
            "data": 25.344056115282434,
            "posture": 0.020994493493845405,
            "total": 25.365956439329878,
            "velocity": 0.0006565251111737243
          },
          {
            "acceleration": 0.00017783171102657845,
            "data": 19.509205916652256,
            "posture": 0.07911962837985352,
            "total": 19.59262055673651,
            "velocity": 0.004117179993371934
          },
          {
            "acceleration": 0.0002782541753612742,
            "data": 14.584050709302417,
            "posture": 0.15888318771299245,
            "total": 14.747985794471054,
            "velocity": 0.004773643280281966
          },
          {
            "acceleration": 0.0003685176093549393,
            "data": 9.338748496901626,
            "posture": 0.1985186781029574,
            "total": 9.54588026891642,
            "velocity": 0.008244576302482097
          },
          {
            "acceleration": 0.0008532075325689982,
            "data": 4.569332730685083,
            "posture": 0.2731024146376818,
            "total": 4.8739107241917985,
            "velocity": 0.030622371336464085
          },
          {
            "acceleration": 0.0014025121924208174,
            "data": 2.962108658832936,
            "posture": 0.2889335118107192,
            "total": 3.2958508900813577,
            "velocity": 0.0434062072452821
          },
          {
            "acceleration": 0.00241018695419529,
            "data": 1.8480553340214119,
            "posture": 0.29866236738603447,
            "total": 2.213776092879879,
            "velocity": 0.0646482045182373
          },
          {
            "acceleration": 0.0035678300919272233,
            "data": 1.1241309858332946,
            "posture": 0.30674256868349753,
            "total": 1.52587500743856,
            "velocity": 0.09143362282984063
          },
          {
            "acceleration": 0.004824018965761777,
            "data": 0.7155214164200777,
            "posture": 0.31442545216508005,
            "total": 1.1547246911125038,
            "velocity": 0.11995380356158426
          },
          {
            "acceleration": 0.005367422868153671,
            "data": 0.6052527387988424,
            "posture": 0.3170259672371825,
            "total": 1.0610325570772128,
            "velocity": 0.1333864281730342
          },
          {
            "acceleration": 0.0077920499273074805,
            "data": 0.32926530160565703,
            "posture": 0.3292281896409749,
            "total": 0.8696716224368493,
            "velocity": 0.20338608126290983
          },
          {
            "acceleration": 0.008935221234153953,
            "data": 0.23858631817446826,
            "posture": 0.33585700331198853,
            "total": 0.8424628787883831,
            "velocity": 0.25908433606777237
          },
          {
            "acceleration": 0.009221226776331337,
            "data": 0.20651990850632768,
            "posture": 0.33802541177593864,
            "total": 0.8410988770037602,
            "velocity": 0.28733232994516245
          },
          {
            "acceleration": 0.009163541282651103,
            "data": 0.20594738735530088,
            "posture": 0.33799997294311124,
            "total": 0.839844641337304,
            "velocity": 0.2867337397562408
          },
          {
            "acceleration": 0.009126334957698996,
            "data": 0.20569756776476997,
            "posture": 0.3380007621808133,
            "total": 0.8391773648095783,
            "velocity": 0.28635269990629597
          },
          {
            "acceleration": 0.009077117823686576,
            "data": 0.20546007630190113,
            "posture": 0.3380341694248309,
            "total": 0.8384565506472756,
            "velocity": 0.28588518709685706
          },
          {
            "acceleration": 0.009076802059236725,
            "data": 0.2054581388494583,
            "posture": 0.33803526298557135,
            "total": 0.8384540226664143,
            "velocity": 0.2858838187721479
          }
        ],
        "objective_terms": {
          "acceleration": 0.009076802059236725,
          "data": 0.2054581388494583,
          "posture": 0.33803526298557135,
          "total": 0.8384540226664143,
          "velocity": 0.2858838187721479
        },
        "orientation_rmse_deg": 15.690477779093781,
        "position_p95_mm": 24.164852802889712,
        "position_rmse_mm": 11.437575799264321,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.6932907348242812,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.6666666666666666,
        "wrist_target_height_median_mm": 21.93968157853314
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_4_20231026_164812_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 974,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8798767967145791,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 21,
        "max_joint_acceleration_rad_s2": 1.7521351422005675,
        "max_joint_velocity_rad_s": 0.5680108178193173,
        "min_collision_margin": 0.03964078610540549,
        "min_floor_margin_mm": 0.0007724699085595588,
        "min_joint_limit_margin_rad": 1.899084755051192,
        "min_urdf_velocity_limit_margin_rad_s": 1.5902552864682526,
        "min_velocity_limit_margin_rad_s": 1.5902552864682526,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 0.7152511739221353,
        "objective_history": [
          {
            "acceleration": 0.00042479966512262224,
            "data": 26.402287323595242,
            "posture": 0.02068201535390486,
            "total": 26.425380427960846,
            "velocity": 0.001986289346576142
          },
          {
            "acceleration": 0.0002477609468698707,
            "data": 20.089777170217623,
            "posture": 0.07877708475938446,
            "total": 20.176913935308153,
            "velocity": 0.008111919384277428
          },
          {
            "acceleration": 0.00035495294862398596,
            "data": 14.816319362233056,
            "posture": 0.1619853872533441,
            "total": 14.987511837360874,
            "velocity": 0.008852134925848404
          },
          {
            "acceleration": 0.00042388913079845507,
            "data": 9.175395579354372,
            "posture": 0.19820577673830897,
            "total": 9.38783984768215,
            "velocity": 0.013814602458670448
          },
          {
            "acceleration": 0.0011315439176226396,
            "data": 4.133900429928657,
            "posture": 0.26820090817055375,
            "total": 4.440460422870564,
            "velocity": 0.03722754085373026
          },
          {
            "acceleration": 0.0018127338182272372,
            "data": 2.639717606697955,
            "posture": 0.2888092070767738,
            "total": 2.9848447079522558,
            "velocity": 0.054505160359299705
          },
          {
            "acceleration": 0.0027689727835121013,
            "data": 1.600606856123071,
            "posture": 0.3021549707402813,
            "total": 1.9834298217356476,
            "velocity": 0.07789902208878323
          },
          {
            "acceleration": 0.00368225204405502,
            "data": 0.9732882380038587,
            "posture": 0.313675591377261,
            "total": 1.3939602223074883,
            "velocity": 0.10331414088231355
          },
          {
            "acceleration": 0.004561031607711357,
            "data": 0.6360932119420966,
            "posture": 0.32149590548331997,
            "total": 1.0899344721717994,
            "velocity": 0.12778432313867152
          },
          {
            "acceleration": 0.004911875998590758,
            "data": 0.5330758573714147,
            "posture": 0.3249210103309338,
            "total": 1.0013070399215958,
            "velocity": 0.13839829622065652
          },
          {
            "acceleration": 0.005609250039880223,
            "data": 0.37692005471906975,
            "posture": 0.33311756850800356,
            "total": 0.8759760455390124,
            "velocity": 0.16032917227205887
          },
          {
            "acceleration": 0.006882542266863006,
            "data": 0.1989572851712755,
            "posture": 0.35242486726857813,
            "total": 0.761326757934593,
            "velocity": 0.20306206322787637
          },
          {
            "acceleration": 0.007515524770677164,
            "data": 0.12428167863123595,
            "posture": 0.3637750746269654,
            "total": 0.7281037713460277,
            "velocity": 0.23253149331714915
          },
          {
            "acceleration": 0.007533617496850083,
            "data": 0.1132565319141831,
            "posture": 0.36495694363334813,
            "total": 0.7204192695229463,
            "velocity": 0.234672176478565
          },
          {
            "acceleration": 0.007567952880783873,
            "data": 0.10556326441873023,
            "posture": 0.36626571536390296,
            "total": 0.7168241768413854,
            "velocity": 0.2374272441779684
          },
          {
            "acceleration": 0.007591134360109911,
            "data": 0.10196312467475818,
            "posture": 0.36691980697967574,
            "total": 0.7156140527416031,
            "velocity": 0.23913998672705916
          },
          {
            "acceleration": 0.007606427098537208,
            "data": 0.10140493685144153,
            "posture": 0.3669588331711478,
            "total": 0.715375581291094,
            "velocity": 0.23940538416996757
          },
          {
            "acceleration": 0.007615478492026255,
            "data": 0.10103234342400015,
            "posture": 0.36699753507045596,
            "total": 0.7152710176497317,
            "velocity": 0.23962566066324933
          },
          {
            "acceleration": 0.007622109485625358,
            "data": 0.10060570814158507,
            "posture": 0.36707172219678635,
            "total": 0.7152521260109115,
            "velocity": 0.23995258618691473
          },
          {
            "acceleration": 0.007624008192366508,
            "data": 0.10049867114288924,
            "posture": 0.3670912127360138,
            "total": 0.7152511738830385,
            "velocity": 0.24003728181176892
          },
          {
            "acceleration": 0.007624008217571283,
            "data": 0.1004986696164597,
            "posture": 0.3670912130321657,
            "total": 0.7152511739221353,
            "velocity": 0.24003728305593866
          }
        ],
        "objective_terms": {
          "acceleration": 0.007624008217571283,
          "data": 0.1004986696164597,
          "posture": 0.3670912130321657,
          "total": 0.7152511739221353,
          "velocity": 0.24003728305593866
        },
        "orientation_rmse_deg": 10.19912527808229,
        "position_p95_mm": 16.25184094055538,
        "position_rmse_mm": 7.868555658073565,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.8162217659137577,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8809034907597536,
        "wrist_target_height_median_mm": 55.7791380056147
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_5_20231027_112303_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 702,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.7549857549857549,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 20,
        "max_joint_acceleration_rad_s2": 1.6147415130272842,
        "max_joint_velocity_rad_s": 0.7831049840155117,
        "min_collision_margin": 0.03953056288221311,
        "min_floor_margin_mm": 0.00012233104817321871,
        "min_joint_limit_margin_rad": 1.8393336953835486,
        "min_urdf_velocity_limit_margin_rad_s": 1.7376617143971305,
        "min_velocity_limit_margin_rad_s": 1.7376617143971305,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.0692913252088774,
        "objective_history": [
          {
            "acceleration": 0.00031638184314036776,
            "data": 20.67714532551007,
            "posture": 0.013125442006300694,
            "total": 20.691814475112537,
            "velocity": 0.001227325753021702
          },
          {
            "acceleration": 0.00046861542337020464,
            "data": 15.762036030007497,
            "posture": 0.05284450084447302,
            "total": 15.821605718076322,
            "velocity": 0.0062565718009798986
          },
          {
            "acceleration": 0.0006130238575671396,
            "data": 11.655007853387168,
            "posture": 0.11294782654605101,
            "total": 11.774600188863133,
            "velocity": 0.006031485072345848
          },
          {
            "acceleration": 0.0009740693023365597,
            "data": 7.324990893769512,
            "posture": 0.14077524683278939,
            "total": 7.4797618552550444,
            "velocity": 0.013021645350406034
          },
          {
            "acceleration": 0.0017679631954008528,
            "data": 3.554107954604853,
            "posture": 0.20262297984138433,
            "total": 3.7948717665051386,
            "velocity": 0.036372868863500385
          },
          {
            "acceleration": 0.004778420422326048,
            "data": 1.5138962261008773,
            "posture": 0.25938126281540824,
            "total": 1.850043883085368,
            "velocity": 0.07198797374675658
          },
          {
            "acceleration": 0.006222019322338374,
            "data": 1.172607495018931,
            "posture": 0.26864395945125685,
            "total": 1.5618019153390303,
            "velocity": 0.11432844154650428
          },
          {
            "acceleration": 0.007988657903295816,
            "data": 0.9672351960154599,
            "posture": 0.27279516203623755,
            "total": 1.414449089117522,
            "velocity": 0.16643007316252872
          },
          {
            "acceleration": 0.009614298983028152,
            "data": 0.8255422644727008,
            "posture": 0.2740407725727714,
            "total": 1.3290053866287086,
            "velocity": 0.21980805060020833
          },
          {
            "acceleration": 0.01089987767731116,
            "data": 0.7148840029938708,
            "posture": 0.2753326905383724,
            "total": 1.2694909068916362,
            "velocity": 0.268374335682082
          },
          {
            "acceleration": 0.01194940565169467,
            "data": 0.6232736472508873,
            "posture": 0.27763474155801976,
            "total": 1.2223494446710041,
            "velocity": 0.3094916502104025
          },
          {
            "acceleration": 0.012779105624266048,
            "data": 0.5428303024159946,
            "posture": 0.2809942426851961,
            "total": 1.1799708689122845,
            "velocity": 0.3433672181868277
          },
          {
            "acceleration": 0.014393969802354762,
            "data": 0.4093463513650928,
            "posture": 0.2916343565702994,
            "total": 1.1213940604391155,
            "velocity": 0.40601938270136845
          },
          {
            "acceleration": 0.01473472710537594,
            "data": 0.34303002389149845,
            "posture": 0.30115951143791403,
            "total": 1.0956521900278946,
            "velocity": 0.43672792759310625
          },
          {
            "acceleration": 0.014712270464310189,
            "data": 0.3341020022410203,
            "posture": 0.30288403600979685,
            "total": 1.0815589985729208,
            "velocity": 0.4298606898577934
          },
          {
            "acceleration": 0.014659970431665774,
            "data": 0.33058584598185403,
            "posture": 0.30448953586022015,
            "total": 1.0719606868247835,
            "velocity": 0.42222533455104355
          },
          {
            "acceleration": 0.014656876305443337,
            "data": 0.3295513531708607,
            "posture": 0.3045496463767173,
            "total": 1.0707067081010058,
            "velocity": 0.4219488322479846
          },
          {
            "acceleration": 0.014653059889490712,
            "data": 0.32901915885504673,
            "posture": 0.3046183943864609,
            "total": 1.0698830236718657,
            "velocity": 0.42159241054086727
          },
          {
            "acceleration": 0.014651524567583531,
            "data": 0.32875357864832816,
            "posture": 0.30467700523425034,
            "total": 1.0693802711597002,
            "velocity": 0.421298162709538
          },
          {
            "acceleration": 0.014650783110676231,
            "data": 0.3287110062338665,
            "posture": 0.30468900303588925,
            "total": 1.0692913252088774,
            "velocity": 0.4212405328284454
          }
        ],
        "objective_terms": {
          "acceleration": 0.014650783110676231,
          "data": 0.3287110062338665,
          "posture": 0.30468900303588925,
          "total": 1.0692913252088774,
          "velocity": 0.4212405328284454
        },
        "orientation_rmse_deg": 23.08314855370802,
        "position_p95_mm": 26.19682086133618,
        "position_rmse_mm": 15.745596944947353,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.6837606837606838,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.7606837606837606,
        "wrist_target_height_median_mm": 52.68163701796106
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_6_20231025_110646_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1292,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8846749226006192,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 18,
        "max_joint_acceleration_rad_s2": 1.6098902389472483,
        "max_joint_velocity_rad_s": 0.7917395005481923,
        "min_collision_margin": 0.039364902536616994,
        "min_floor_margin_mm": -3.209234475387568e-07,
        "min_joint_limit_margin_rad": 2.0646558136019424,
        "min_urdf_velocity_limit_margin_rad_s": 1.7217393675904158,
        "min_velocity_limit_margin_rad_s": 1.7217393675904158,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.1518964248646537,
        "objective_history": [
          {
            "acceleration": 0.00023116744021344846,
            "data": 37.38725659317447,
            "posture": 0.02725542540759839,
            "total": 37.41680911362997,
            "velocity": 0.0020659276076913525
          },
          {
            "acceleration": 0.0004268921081586587,
            "data": 28.564164376666827,
            "posture": 0.10508331606808102,
            "total": 28.679342316630816,
            "velocity": 0.009667731787747613
          },
          {
            "acceleration": 0.0004656752774739352,
            "data": 21.197613253705477,
            "posture": 0.2148748000095786,
            "total": 21.423369104024548,
            "velocity": 0.010415375032018922
          },
          {
            "acceleration": 0.0007444287314438805,
            "data": 13.378061570841401,
            "posture": 0.2682784363100818,
            "total": 13.661869195164687,
            "velocity": 0.01478475928176034
          },
          {
            "acceleration": 0.0014381431450781216,
            "data": 6.404747554485677,
            "posture": 0.38106937556205117,
            "total": 6.823286294661174,
            "velocity": 0.03603122146836713
          },
          {
            "acceleration": 0.0021045438225081726,
            "data": 4.288816193166245,
            "posture": 0.4212484572821549,
            "total": 4.76172265921442,
            "velocity": 0.04955346494351125
          },
          {
            "acceleration": 0.0030672010615575367,
            "data": 2.984552327089941,
            "posture": 0.4532724563388465,
            "total": 3.5162822755193175,
            "velocity": 0.07539029102897316
          },
          {
            "acceleration": 0.004168282103897639,
            "data": 2.29889958751178,
            "posture": 0.47190072089579127,
            "total": 2.8797488761173358,
            "velocity": 0.104780285605867
          },
          {
            "acceleration": 0.005231160125479857,
            "data": 1.9150566435130463,
            "posture": 0.4799298107925923,
            "total": 2.535586609951918,
            "velocity": 0.1353689955207998
          },
          {
            "acceleration": 0.0056693254616000275,
            "data": 1.7806615551775962,
            "posture": 0.4829518285314251,
            "total": 2.4172464658554773,
            "velocity": 0.14796375668485595
          },
          {
            "acceleration": 0.006527688767703623,
            "data": 1.5351447708597945,
            "posture": 0.490825021428692,
            "total": 2.2055660695381443,
            "velocity": 0.17306858848195428
          },
          {
            "acceleration": 0.008101793481383825,
            "data": 1.0920589834271912,
            "posture": 0.5139656973451217,
            "total": 1.834373326527218,
            "velocity": 0.22024685227352117
          },
          {
            "acceleration": 0.00905976889055844,
            "data": 0.696985929309986,
            "posture": 0.543691497717742,
            "total": 1.4994456626090975,
            "velocity": 0.2497084666908112
          },
          {
            "acceleration": 0.009575817485355999,
            "data": 0.396392370553999,
            "posture": 0.5777026941358995,
            "total": 1.2576175711593405,
            "velocity": 0.2739466889840862
          },
          {
            "acceleration": 0.009466925459954355,
            "data": 0.24681140214713457,
            "posture": 0.6090640259367862,
            "total": 1.1636553552623599,
            "velocity": 0.2983130017184846
          },
          {
            "acceleration": 0.009392661464246853,
            "data": 0.2348254788973655,
            "posture": 0.6116154205051358,
            "total": 1.1536684737033198,
            "velocity": 0.2978349128365717
          },
          {
            "acceleration": 0.009378580782931314,
            "data": 0.2326749629704894,
            "posture": 0.6121320115624341,
            "total": 1.1519027679458653,
            "velocity": 0.29771721263001055
          },
          {
            "acceleration": 0.009378528680622795,
            "data": 0.23266723127938937,
            "posture": 0.6121339083801097,
            "total": 1.1518964248646537,
            "velocity": 0.2977167565245318
          }
        ],
        "objective_terms": {
          "acceleration": 0.009378528680622795,
          "data": 0.23266723127938937,
          "posture": 0.6121339083801097,
          "total": 1.1518964248646537,
          "velocity": 0.2977167565245318
        },
        "orientation_rmse_deg": 14.635195745029575,
        "position_p95_mm": 22.817886599884233,
        "position_rmse_mm": 12.230114139153825,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7902476780185759,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8777089783281734,
        "wrist_target_height_median_mm": 111.07870334993042
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_6_20231025_111118_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1272,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.809748427672956,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 21,
        "max_joint_acceleration_rad_s2": 1.7521191562732634,
        "max_joint_velocity_rad_s": 0.769866289792175,
        "min_collision_margin": 0.03970279232979183,
        "min_floor_margin_mm": 7.479846731578421e-05,
        "min_joint_limit_margin_rad": 2.0445415033833143,
        "min_urdf_velocity_limit_margin_rad_s": 1.5853218750112883,
        "min_velocity_limit_margin_rad_s": 1.5853218750112883,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.4109549596496995,
        "objective_history": [
          {
            "acceleration": 0.0002575018554150479,
            "data": 35.43363015710769,
            "posture": 0.02476710971435378,
            "total": 35.46186524146619,
            "velocity": 0.003210472788728846
          },
          {
            "acceleration": 0.0005338116476078055,
            "data": 27.135446254288613,
            "posture": 0.09867073822943383,
            "total": 27.246391851083207,
            "velocity": 0.011741046917551748
          },
          {
            "acceleration": 0.0008175878833133943,
            "data": 20.209129474086133,
            "posture": 0.20622258573518631,
            "total": 20.43306946503659,
            "velocity": 0.016899817331952997
          },
          {
            "acceleration": 0.001105963741404275,
            "data": 12.883290293078936,
            "posture": 0.2587104908758261,
            "total": 13.166708587734863,
            "velocity": 0.02360184003869721
          },
          {
            "acceleration": 0.0013531666705538744,
            "data": 9.54902480749703,
            "posture": 0.3089749421666142,
            "total": 9.889606438227379,
            "velocity": 0.03025352189318014
          },
          {
            "acceleration": 0.0018390306777600703,
            "data": 6.634467969090626,
            "posture": 0.3595373239142971,
            "total": 7.037048662601002,
            "velocity": 0.04120433891831833
          },
          {
            "acceleration": 0.002711920030214796,
            "data": 4.635172187561574,
            "posture": 0.4097042158978291,
            "total": 5.105859009878855,
            "velocity": 0.0582706863892372
          },
          {
            "acceleration": 0.0037803314690853453,
            "data": 3.2548944753423763,
            "posture": 0.44352913585046627,
            "total": 3.7880676687204695,
            "velocity": 0.08586372605854159
          },
          {
            "acceleration": 0.005054703700377878,
            "data": 2.4071957246636546,
            "posture": 0.462356257048447,
            "total": 2.996228650000036,
            "velocity": 0.12162196458755624
          },
          {
            "acceleration": 0.006436528253265148,
            "data": 1.9297799493733556,
            "posture": 0.47361874243142704,
            "total": 2.5743894082533987,
            "velocity": 0.16455418819535095
          },
          {
            "acceleration": 0.007777756576536885,
            "data": 1.6343110102900549,
            "posture": 0.48372217930380623,
            "total": 2.333348506808069,
            "velocity": 0.2075375606376707
          },
          {
            "acceleration": 0.01020689534710608,
            "data": 1.1924648127137742,
            "posture": 0.5073407565588397,
            "total": 2.0070247496696965,
            "velocity": 0.2970122850499763
          },
          {
            "acceleration": 0.011581329692393361,
            "data": 0.8414923381411956,
            "posture": 0.5306283470827712,
            "total": 1.7444787189882036,
            "velocity": 0.36077670407184337
          },
          {
            "acceleration": 0.012069355099560567,
            "data": 0.5803898745238807,
            "posture": 0.5550648632397465,
            "total": 1.5456321716717891,
            "velocity": 0.3981080788086016
          },
          {
            "acceleration": 0.012156288521044407,
            "data": 0.43728368652796157,
            "posture": 0.576972447113508,
            "total": 1.4487530381164557,
            "velocity": 0.42234061595394157
          },
          {
            "acceleration": 0.011905713825430566,
            "data": 0.38763775603651773,
            "posture": 0.5899430561804667,
            "total": 1.4174737364324173,
            "velocity": 0.4279872103900022
          },
          {
            "acceleration": 0.011893357972024133,
            "data": 0.38455777618643905,
            "posture": 0.5899911027651159,
            "total": 1.4141544888247082,
            "velocity": 0.4277122519011291
          },
          {
            "acceleration": 0.01188659636528731,
            "data": 0.3830859136442137,
            "posture": 0.5900316444727886,
            "total": 1.412494595679085,
            "velocity": 0.42749044119679547
          },
          {
            "acceleration": 0.011881210164160369,
            "data": 0.38240245213772006,
            "posture": 0.5900716905207398,
            "total": 1.4116321299275827,
            "velocity": 0.4272767771049625
          },
          {
            "acceleration": 0.011876750327722774,
            "data": 0.3820790641234516,
            "posture": 0.5901130655696825,
            "total": 1.4111425486743827,
            "velocity": 0.42707366865352575
          },
          {
            "acceleration": 0.011874316713445856,
            "data": 0.38195263754371356,
            "posture": 0.5901309844732782,
            "total": 1.4109549596496995,
            "velocity": 0.4269970209192618
          }
        ],
        "objective_terms": {
          "acceleration": 0.011874316713445856,
          "data": 0.38195263754371356,
          "posture": 0.5901309844732782,
          "total": 1.4109549596496995,
          "velocity": 0.4269970209192618
        },
        "orientation_rmse_deg": 20.858462241366112,
        "position_p95_mm": 20.98781665276088,
        "position_rmse_mm": 12.742967732267644,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.8270440251572327,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.815251572327044,
        "wrist_target_height_median_mm": 79.75369354652565
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_6_20231025_112546_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1311,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8764302059496567,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 25,
        "max_joint_acceleration_rad_s2": 2.1556560223591745,
        "max_joint_velocity_rad_s": 0.8516946275886542,
        "min_collision_margin": 0.03926933646975811,
        "min_floor_margin_mm": 0.0003636097646725345,
        "min_joint_limit_margin_rad": 2.1755428932039207,
        "min_urdf_velocity_limit_margin_rad_s": 1.595478672292918,
        "min_velocity_limit_margin_rad_s": 1.595478672292918,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.2315774777281248,
        "objective_history": [
          {
            "acceleration": 0.00037082092021778647,
            "data": 33.753446346924726,
            "posture": 0.024577073818155565,
            "total": 33.78002874161479,
            "velocity": 0.0016344999516905735
          },
          {
            "acceleration": 0.0006133635583425714,
            "data": 25.679182609152946,
            "posture": 0.09809804838089617,
            "total": 25.78900233324803,
            "velocity": 0.011108312155841989
          },
          {
            "acceleration": 0.0008235572394369129,
            "data": 18.930563176209745,
            "posture": 0.20633009631848223,
            "total": 19.15573324253791,
            "velocity": 0.018016412770247292
          },
          {
            "acceleration": 0.0011096965091701866,
            "data": 11.844948063148397,
            "posture": 0.25786779211456623,
            "total": 12.128812825608584,
            "velocity": 0.024887273836450355
          },
          {
            "acceleration": 0.0013173787779558596,
            "data": 8.687693397391115,
            "posture": 0.30458885233825567,
            "total": 9.022969279323137,
            "velocity": 0.029369650815810236
          },
          {
            "acceleration": 0.0018157196828663892,
            "data": 5.956427199916625,
            "posture": 0.3513123621321338,
            "total": 6.3492549673111895,
            "velocity": 0.039699685579563354
          },
          {
            "acceleration": 0.0026592913461352444,
            "data": 4.179589422179204,
            "posture": 0.39381565799360563,
            "total": 4.634173421976418,
            "velocity": 0.05810905045747344
          },
          {
            "acceleration": 0.003750138098498475,
            "data": 3.1031459813083737,
            "posture": 0.42194362908365296,
            "total": 3.6099471708740394,
            "velocity": 0.08110742238351423
          },
          {
            "acceleration": 0.004973078878269818,
            "data": 2.41213394008731,
            "posture": 0.43642484690991734,
            "total": 2.9626831704338725,
            "velocity": 0.10915130455837502
          },
          {
            "acceleration": 0.006223542825785083,
            "data": 1.9887464941918034,
            "posture": 0.44843614907908025,
            "total": 2.5855943018004828,
            "velocity": 0.14218811570381398
          },
          {
            "acceleration": 0.007411795588987305,
            "data": 1.6957419885054685,
            "posture": 0.46151055311330114,
            "total": 2.3398050018730348,
            "velocity": 0.17514066466527764
          },
          {
            "acceleration": 0.009495751338069706,
            "data": 1.2325012734011667,
            "posture": 0.493889350124357,
            "total": 1.974070997053718,
            "velocity": 0.23818462219012454
          },
          {
            "acceleration": 0.01072254037074912,
            "data": 0.8264107617702565,
            "posture": 0.5284680214705894,
            "total": 1.6327805281819265,
            "velocity": 0.2671792045703316
          },
          {
            "acceleration": 0.011720820593282978,
            "data": 0.5026712271310834,
            "posture": 0.5685063806806254,
            "total": 1.3786317496891516,
            "velocity": 0.2957333212841597
          },
          {
            "acceleration": 0.011909152648131124,
            "data": 0.3287110445884156,
            "posture": 0.6054013266196897,
            "total": 1.2668502661389272,
            "velocity": 0.32082874228269065
          },
          {
            "acceleration": 0.011816452703567995,
            "data": 0.26413919582034484,
            "posture": 0.6244406117959953,
            "total": 1.2360199094139506,
            "velocity": 0.33562364909404235
          },
          {
            "acceleration": 0.01180297994838521,
            "data": 0.2629449036666959,
            "posture": 0.6244751871159069,
            "total": 1.234643342937117,
            "velocity": 0.3354202722061292
          },
          {
            "acceleration": 0.011793123812072478,
            "data": 0.2623271657963938,
            "posture": 0.6245245313357508,
            "total": 1.2338814503478752,
            "velocity": 0.33523662940365806
          },
          {
            "acceleration": 0.01177714409760984,
            "data": 0.26163557130459936,
            "posture": 0.6246418699066049,
            "total": 1.2329633055882523,
            "velocity": 0.3349087202794382
          },
          {
            "acceleration": 0.011764223001009681,
            "data": 0.26142035790641555,
            "posture": 0.624781676708236,
            "total": 1.2326065277078608,
            "velocity": 0.3346402700921996
          },
          {
            "acceleration": 0.011754064066075551,
            "data": 0.26120881293594683,
            "posture": 0.6249248349103826,
            "total": 1.2323094315711365,
            "velocity": 0.3344217196587314
          },
          {
            "acceleration": 0.011746069837386052,
            "data": 0.2609949328811428,
            "posture": 0.6250674024598827,
            "total": 1.2320498832241003,
            "velocity": 0.33424147804568877
          },
          {
            "acceleration": 0.01173957741516541,
            "data": 0.26078027962807815,
            "posture": 0.6252083592098341,
            "total": 1.2318202152418543,
            "velocity": 0.3340919989887766
          },
          {
            "acceleration": 0.01173404703519083,
            "data": 0.26056613710765086,
            "posture": 0.6253471528924527,
            "total": 1.2316142335648066,
            "velocity": 0.3339668965295121
          },
          {
            "acceleration": 0.011733110740624634,
            "data": 0.2605265503332391,
            "posture": 0.6253611372940214,
            "total": 1.2315774777281248,
            "velocity": 0.3339566793602397
          }
        ],
        "objective_terms": {
          "acceleration": 0.011733110740624634,
          "data": 0.2605265503332391,
          "posture": 0.6253611372940214,
          "total": 1.2315774777281248,
          "velocity": 0.3339566793602397
        },
        "orientation_rmse_deg": 14.947589395477285,
        "position_p95_mm": 21.561294860154685,
        "position_rmse_mm": 12.675177855001246,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.8443935926773455,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8771929824561403,
        "wrist_target_height_median_mm": 123.34778018795103
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_7_20231022_193506_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1136,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9302120141342756,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 18,
        "max_joint_acceleration_rad_s2": 2.2622528656243257,
        "max_joint_velocity_rad_s": 0.6310431140564416,
        "min_collision_margin": 0.04002523021998727,
        "min_floor_margin_mm": 0.008944906608427533,
        "min_joint_limit_margin_rad": 1.8187771516578106,
        "min_urdf_velocity_limit_margin_rad_s": 1.587691214126517,
        "min_velocity_limit_margin_rad_s": 1.587691214126517,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.0652004650289688,
        "objective_history": [
          {
            "acceleration": 0.00016906480891655878,
            "data": 33.80468405251051,
            "posture": 0.02493491323776344,
            "total": 33.830703525301224,
            "velocity": 0.0009154947440356926
          },
          {
            "acceleration": 0.00023127920178836524,
            "data": 26.48148427897223,
            "posture": 0.09653633751077334,
            "total": 26.583442753960973,
            "velocity": 0.005190858276184315
          },
          {
            "acceleration": 0.00033004441451672513,
            "data": 20.23809192755064,
            "posture": 0.1911848376062417,
            "total": 20.438579100288393,
            "velocity": 0.008972290716992374
          },
          {
            "acceleration": 0.0005471521432087031,
            "data": 13.72375914982539,
            "posture": 0.24031371869052184,
            "total": 13.97736054907655,
            "velocity": 0.012740528417427864
          },
          {
            "acceleration": 0.0010681368573857424,
            "data": 7.907805196156249,
            "posture": 0.3245257419991298,
            "total": 8.262565053805961,
            "velocity": 0.029165978793195463
          },
          {
            "acceleration": 0.0013357534867554052,
            "data": 5.621668218056646,
            "posture": 0.34967442574296587,
            "total": 6.011044226094929,
            "velocity": 0.03836582880856142
          },
          {
            "acceleration": 0.0020899758024507348,
            "data": 3.853309598487394,
            "posture": 0.3687462262004082,
            "total": 4.276886878215663,
            "velocity": 0.05274107772541011
          },
          {
            "acceleration": 0.003174971173697821,
            "data": 2.499385074591866,
            "posture": 0.38586278868572765,
            "total": 2.9620111989393183,
            "velocity": 0.07358836448802661
          },
          {
            "acceleration": 0.004352942211721633,
            "data": 1.5535348944902891,
            "posture": 0.4042510750462289,
            "total": 2.0623553352400252,
            "velocity": 0.10021642349178544
          },
          {
            "acceleration": 0.005487687422872475,
            "data": 1.0170798983195053,
            "posture": 0.42140781229918506,
            "total": 1.5734934496753628,
            "velocity": 0.12951805163380004
          },
          {
            "acceleration": 0.006464606294446992,
            "data": 0.7113797422711943,
            "posture": 0.4379451869307113,
            "total": 1.3161350496864872,
            "velocity": 0.1603455141901347
          },
          {
            "acceleration": 0.008277288835819347,
            "data": 0.40115341606312505,
            "posture": 0.47219402357623913,
            "total": 1.109205647434826,
            "velocity": 0.22758091895964244
          },
          {
            "acceleration": 0.009152571141333873,
            "data": 0.2909840493056127,
            "posture": 0.49312415142065313,
            "total": 1.068587298140526,
            "velocity": 0.2753265262729264
          },
          {
            "acceleration": 0.00923064537777077,
            "data": 0.27394165155869293,
            "posture": 0.49526751533496227,
            "total": 1.0659426965436254,
            "velocity": 0.2875028842721994
          },
          {
            "acceleration": 0.00928588714313005,
            "data": 0.26894858458586557,
            "posture": 0.4959893083329297,
            "total": 1.0655380618601702,
            "velocity": 0.2913142817982449
          },
          {
            "acceleration": 0.009298075708015089,
            "data": 0.26819271542775647,
            "posture": 0.4960165478705485,
            "total": 1.0652895534834605,
            "velocity": 0.2917822144771405
          },
          {
            "acceleration": 0.009305480922591612,
            "data": 0.26777605442458735,
            "posture": 0.4960388542004732,
            "total": 1.065200464964113,
            "velocity": 0.29208007541646075
          },
          {
            "acceleration": 0.00930548117587094,
            "data": 0.26777604358695817,
            "posture": 0.4960388551056651,
            "total": 1.0652004650289688,
            "velocity": 0.2920800851604746
          }
        ],
        "objective_terms": {
          "acceleration": 0.00930548117587094,
          "data": 0.26777604358695817,
          "posture": 0.4960388551056651,
          "total": 1.0652004650289688,
          "velocity": 0.2920800851604746
        },
        "orientation_rmse_deg": 16.46206598269144,
        "position_p95_mm": 26.53956576392247,
        "position_rmse_mm": 12.931296149260826,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9257950530035336,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9304577464788732,
        "wrist_target_height_median_mm": 65.76848784844873
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_7_20231023_162803_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          -175.675,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1092,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9917431192660551,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 29,
        "max_joint_acceleration_rad_s2": 1.7173976338981347,
        "max_joint_velocity_rad_s": 1.5048948946531202,
        "min_collision_margin": 0.03932631108857233,
        "min_floor_margin_mm": 0.0001015667860180397,
        "min_joint_limit_margin_rad": 1.560268407223535,
        "min_urdf_velocity_limit_margin_rad_s": 1.6351175675941032,
        "min_velocity_limit_margin_rad_s": 1.6351175675941032,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 3.7225718662224474,
        "objective_history": [
          {
            "acceleration": 0.000635743971773701,
            "data": 29.410493549558844,
            "posture": 0.021917142101384332,
            "total": 29.445657278651616,
            "velocity": 0.012610843019616682
          },
          {
            "acceleration": 0.0008191882369540414,
            "data": 23.102282487412225,
            "posture": 0.08162457130923365,
            "total": 23.21486299554264,
            "velocity": 0.030136748584225815
          },
          {
            "acceleration": 0.0017145094447989372,
            "data": 17.689128013606197,
            "posture": 0.17004304354901295,
            "total": 17.923744272203194,
            "velocity": 0.06285870560318682
          },
          {
            "acceleration": 0.0018796904913342977,
            "data": 15.00436382975904,
            "posture": 0.19420231901634175,
            "total": 15.279989134685236,
            "velocity": 0.07954329541852079
          },
          {
            "acceleration": 0.0025152090536695332,
            "data": 12.397926721780781,
            "posture": 0.23127377542974464,
            "total": 12.741254196112033,
            "velocity": 0.10953848984783801
          },
          {
            "acceleration": 0.0036324530236411046,
            "data": 9.97082239191739,
            "posture": 0.2747041549052672,
            "total": 10.403050325092162,
            "velocity": 0.1538913252458636
          },
          {
            "acceleration": 0.005042079700326425,
            "data": 7.90639149019413,
            "posture": 0.32163454492770516,
            "total": 8.44578112802818,
            "velocity": 0.21271301320601865
          },
          {
            "acceleration": 0.006758722107237881,
            "data": 6.471462094695248,
            "posture": 0.36615734830983593,
            "total": 7.1287330081378695,
            "velocity": 0.28435484302554753
          },
          {
            "acceleration": 0.011501766616623683,
            "data": 5.011554156530353,
            "posture": 0.4412317042081436,
            "total": 5.941727855711857,
            "velocity": 0.47744022835673694
          },
          {
            "acceleration": 0.012349053022898035,
            "data": 4.711552072843681,
            "posture": 0.4630964589106152,
            "total": 5.732485853694662,
            "velocity": 0.5454882689174678
          },
          {
            "acceleration": 0.013084892899529906,
            "data": 4.462430455884598,
            "posture": 0.48519818950247473,
            "total": 5.555778989243291,
            "velocity": 0.595065450956689
          },
          {
            "acceleration": 0.013536342649038365,
            "data": 4.238362225815627,
            "posture": 0.5085019904064061,
            "total": 5.385530904767002,
            "velocity": 0.6251303458959308
          },
          {
            "acceleration": 0.013615140309144578,
            "data": 4.13216511978171,
            "posture": 0.5209494620491296,
            "total": 5.299380143608623,
            "velocity": 0.6326504214686385
          },
          {
            "acceleration": 0.013806245919765496,
            "data": 3.915190986848756,
            "posture": 0.5492135647838343,
            "total": 5.126570011323576,
            "velocity": 0.6483592137712203
          },
          {
            "acceleration": 0.013851911278726927,
            "data": 3.692069932149474,
            "posture": 0.5824297270330256,
            "total": 4.9524798441441575,
            "velocity": 0.6641282736829306
          },
          {
            "acceleration": 0.01380572207801262,
            "data": 3.6346186826955478,
            "posture": 0.5917021095979099,
            "total": 4.908108437999726,
            "velocity": 0.6679819236282549
          },
          {
            "acceleration": 0.013815079333724233,
            "data": 3.397665557343326,
            "posture": 0.632114200447194,
            "total": 4.731625926639923,
            "velocity": 0.6880310895156786
          },
          {
            "acceleration": 0.013832330016609145,
            "data": 3.2809645301973385,
            "posture": 0.6541871520019034,
            "total": 4.648614328956858,
            "velocity": 0.6996303167410077
          },
          {
            "acceleration": 0.014059557795076802,
            "data": 3.049907974075591,
            "posture": 0.7017415744111981,
            "total": 4.492139068162806,
            "velocity": 0.7264299618809398
          },
          {
            "acceleration": 0.01436787056884024,
            "data": 2.8217986703375555,
            "posture": 0.7531647250666759,
            "total": 4.346464970539411,
            "velocity": 0.7571337045663394
          },
          {
            "acceleration": 0.015193209948776725,
            "data": 2.3772142209310445,
            "posture": 0.8666672963271184,
            "total": 4.088316374814265,
            "velocity": 0.8292416476073255
          },
          {
            "acceleration": 0.015505093330068293,
            "data": 2.1628388885727374,
            "posture": 0.9234938644315793,
            "total": 3.972044386478708,
            "velocity": 0.8702065401443229
          },
          {
            "acceleration": 0.015929719846432627,
            "data": 1.9604267972401157,
            "posture": 0.9804092390744682,
            "total": 3.868383019845649,
            "velocity": 0.9116172636846325
          },
          {
            "acceleration": 0.0163855766604087,
            "data": 1.7720061742145148,
            "posture": 1.0359981680078856,
            "total": 3.775635720683954,
            "velocity": 0.951245801801145
          },
          {
            "acceleration": 0.016615177618588835,
            "data": 1.6857628432411902,
            "posture": 1.0622237313540344,
            "total": 3.7324528381865605,
            "velocity": 0.9678510859727473
          },
          {
            "acceleration": 0.016644676123790957,
            "data": 1.6754413551379994,
            "posture": 1.0653749833411743,
            "total": 3.727168769202178,
            "velocity": 0.9697077545992133
          },
          {
            "acceleration": 0.016659546899338566,
            "data": 1.6703003044636413,
            "posture": 1.0669497279415132,
            "total": 3.7245401460106056,
            "velocity": 0.9706305667061128
          },
          {
            "acceleration": 0.016666979113394147,
            "data": 1.6677403175388645,
            "posture": 1.067733098701142,
            "total": 3.7232274398672747,
            "velocity": 0.9710870445138738
          },
          {
            "acceleration": 0.016670700430687975,
            "data": 1.6664616381586725,
            "posture": 1.0681246541639449,
            "total": 3.7225718662224474,
            "velocity": 0.9713148734691418
          }
        ],
        "objective_terms": {
          "acceleration": 0.016670700430687975,
          "data": 1.6664616381586725,
          "posture": 1.0681246541639449,
          "total": 3.7225718662224474,
          "velocity": 0.9713148734691418
        },
        "orientation_rmse_deg": 56.75495087579851,
        "position_p95_mm": 37.12526044082383,
        "position_rmse_mm": 19.544816423903793,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 1.0,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9862637362637363,
        "wrist_target_height_median_mm": 138.7315346383191
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_8_20231024_180651_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 985,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.6206543967280164,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 19,
        "max_joint_acceleration_rad_s2": 1.557552926362419,
        "max_joint_velocity_rad_s": 0.5619054273997125,
        "min_collision_margin": 0.039605311352616426,
        "min_floor_margin_mm": 3.398975198520837e-08,
        "min_joint_limit_margin_rad": 1.8490738538547284,
        "min_urdf_velocity_limit_margin_rad_s": 1.5317370000002737,
        "min_velocity_limit_margin_rad_s": 1.5317370000002737,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 0.9288711498527639,
        "objective_history": [
          {
            "acceleration": 0.00043621193042173525,
            "data": 23.079248812349306,
            "posture": 0.021847760219377215,
            "total": 23.10314978278733,
            "velocity": 0.0016169982882285904
          },
          {
            "acceleration": 0.0004365740620828159,
            "data": 17.487604311481313,
            "posture": 0.08317303680172138,
            "total": 17.58074077119243,
            "velocity": 0.009526848847316761
          },
          {
            "acceleration": 0.0005880245800241248,
            "data": 12.761819157409569,
            "posture": 0.17478874034564737,
            "total": 12.950787815842153,
            "velocity": 0.01359189350691338
          },
          {
            "acceleration": 0.0007898932004038099,
            "data": 7.825818275899918,
            "posture": 0.22170829472736955,
            "total": 8.069006293990144,
            "velocity": 0.020689830162452544
          },
          {
            "acceleration": 0.0012892052162867223,
            "data": 3.4635603762326737,
            "posture": 0.3087614752048266,
            "total": 3.8080221865724733,
            "velocity": 0.03441112991868662
          },
          {
            "acceleration": 0.0019524548594411063,
            "data": 2.1814036051299794,
            "posture": 0.3298517092890082,
            "total": 2.5621401370480004,
            "velocity": 0.04893236776957184
          },
          {
            "acceleration": 0.0031644306105507677,
            "data": 1.3486490522212027,
            "posture": 0.34575388633249005,
            "total": 1.773489065363632,
            "velocity": 0.07592169619938843
          },
          {
            "acceleration": 0.0037268792880436807,
            "data": 1.090956237497173,
            "posture": 0.35028182341368524,
            "total": 1.5367171788076626,
            "velocity": 0.09175223860876056
          },
          {
            "acceleration": 0.004297291214309484,
            "data": 0.9025099602461383,
            "posture": 0.35372722172512877,
            "total": 1.368649066298145,
            "velocity": 0.10811459311256831
          },
          {
            "acceleration": 0.0054982118641357216,
            "data": 0.6432341310878424,
            "posture": 0.35830448166925266,
            "total": 1.150374344763046,
            "velocity": 0.14333752014181528
          },
          {
            "acceleration": 0.00644459615289479,
            "data": 0.5219106145827961,
            "posture": 0.35841542584365876,
            "total": 1.061442301100837,
            "velocity": 0.17467166452148716
          },
          {
            "acceleration": 0.00718684910013808,
            "data": 0.4540708885221671,
            "posture": 0.35853364744423677,
            "total": 1.0220315255295291,
            "velocity": 0.20224014046298713
          },
          {
            "acceleration": 0.008373879930531292,
            "data": 0.35943535765691514,
            "posture": 0.3628769232132411,
            "total": 0.9813242513341782,
            "velocity": 0.2506380905334907
          },
          {
            "acceleration": 0.008707135516055335,
            "data": 0.29896526934515544,
            "posture": 0.3677910765713868,
            "total": 0.94600549089188,
            "velocity": 0.2705420094592825
          },
          {
            "acceleration": 0.008735156422599201,
            "data": 0.28756359045069846,
            "posture": 0.3686591731312824,
            "total": 0.9360855313967898,
            "velocity": 0.27112761139220976
          },
          {
            "acceleration": 0.008750147698323753,
            "data": 0.28203305149137764,
            "posture": 0.369143918932535,
            "total": 0.9312573226163678,
            "velocity": 0.2713302044941314
          },
          {
            "acceleration": 0.008753825223299039,
            "data": 0.2806737157355198,
            "posture": 0.36926877791028395,
            "total": 0.9300624851644426,
            "velocity": 0.27136616629533983
          },
          {
            "acceleration": 0.008757515204881853,
            "data": 0.27931945792786544,
            "posture": 0.36939602930413673,
            "total": 0.9288734671028831,
            "velocity": 0.27140046466599904
          },
          {
            "acceleration": 0.008757522301266368,
            "data": 0.2793168258023573,
            "posture": 0.3693962786785935,
            "total": 0.9288711498527639,
            "velocity": 0.2714005230705468
          }
        ],
        "objective_terms": {
          "acceleration": 0.008757522301266368,
          "data": 0.2793168258023573,
          "posture": 0.3693962786785935,
          "total": 0.9288711498527639,
          "velocity": 0.2714005230705468
        },
        "orientation_rmse_deg": 19.502922859348523,
        "position_p95_mm": 24.28049433395339,
        "position_rmse_mm": 11.010370738613009,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7249488752556237,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.6152284263959391,
        "wrist_target_height_median_mm": 18.97199501333835
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_9_20231027_123403_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 992,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.8881048387096774,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 21,
        "max_joint_acceleration_rad_s2": 2.4594270799022344,
        "max_joint_velocity_rad_s": 0.9143487621232862,
        "min_collision_margin": 0.039302408162259245,
        "min_floor_margin_mm": 0.002381113726017403,
        "min_joint_limit_margin_rad": 1.6657899433054677,
        "min_urdf_velocity_limit_margin_rad_s": 1.4949068750979735,
        "min_velocity_limit_margin_rad_s": 1.4949068750979735,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.299714118749388,
        "objective_history": [
          {
            "acceleration": 0.00029590001626339757,
            "data": 26.60319137974507,
            "posture": 0.018438366163930534,
            "total": 26.622739698559883,
            "velocity": 0.0008140526346204916
          },
          {
            "acceleration": 0.00019524393567389164,
            "data": 20.625174868434712,
            "posture": 0.06914864553221453,
            "total": 20.696437287738508,
            "velocity": 0.0019185298359070094
          },
          {
            "acceleration": 0.00022332709471551692,
            "data": 15.658010569642727,
            "posture": 0.15428125278716007,
            "total": 15.81568734467556,
            "velocity": 0.0031721951509575386
          },
          {
            "acceleration": 0.0007629176890933899,
            "data": 10.6591578413043,
            "posture": 0.19206702483792418,
            "total": 10.866753921242816,
            "velocity": 0.014766137411501507
          },
          {
            "acceleration": 0.0013366163816148672,
            "data": 6.359097125446763,
            "posture": 0.2806537250317361,
            "total": 6.67760305037013,
            "velocity": 0.03651558351001504
          },
          {
            "acceleration": 0.004600709633070745,
            "data": 4.052565822172584,
            "posture": 0.35840622578732295,
            "total": 4.5067081495880394,
            "velocity": 0.09113539199506272
          },
          {
            "acceleration": 0.006074893887464589,
            "data": 3.37953635120582,
            "posture": 0.39041193375679956,
            "total": 3.897675676355,
            "velocity": 0.12165249750491612
          },
          {
            "acceleration": 0.008126024567891965,
            "data": 2.8730523989316907,
            "posture": 0.4209136092992423,
            "total": 3.4602326017388316,
            "velocity": 0.15814056894000658
          },
          {
            "acceleration": 0.01014070988707888,
            "data": 2.4447858069174786,
            "posture": 0.43880334664065324,
            "total": 3.0965849094157694,
            "velocity": 0.20285504597055878
          },
          {
            "acceleration": 0.011760471546447642,
            "data": 2.083795309749206,
            "posture": 0.4473987229078499,
            "total": 2.798249453369656,
            "velocity": 0.25529494916615264
          },
          {
            "acceleration": 0.013136146528287032,
            "data": 1.832333040078841,
            "posture": 0.4513649019766087,
            "total": 2.595978572420536,
            "velocity": 0.29914448383679904
          },
          {
            "acceleration": 0.013689215059672882,
            "data": 1.7335173061930302,
            "posture": 0.45144227439916035,
            "total": 2.5132521380936947,
            "velocity": 0.31460334244183125
          },
          {
            "acceleration": 0.014781495941064393,
            "data": 1.54160602755665,
            "posture": 0.4536973363618272,
            "total": 2.3535143094488133,
            "velocity": 0.3434294495892715
          },
          {
            "acceleration": 0.01666602302911121,
            "data": 1.1599825553140843,
            "posture": 0.4663973446725639,
            "total": 2.034216235309451,
            "velocity": 0.39117031229369154
          },
          {
            "acceleration": 0.018258747038410666,
            "data": 0.7994176576333915,
            "posture": 0.4852901329136989,
            "total": 1.7153078190847095,
            "velocity": 0.4123412814992086
          },
          {
            "acceleration": 0.0199366385461973,
            "data": 0.5086145800740827,
            "posture": 0.5117589989747223,
            "total": 1.466129788304802,
            "velocity": 0.42581957070979953
          },
          {
            "acceleration": 0.02026400944972996,
            "data": 0.3427109418185525,
            "posture": 0.5357481310332819,
            "total": 1.3437800429670403,
            "velocity": 0.44505696066547595
          },
          {
            "acceleration": 0.020258261051228268,
            "data": 0.3028895558160931,
            "posture": 0.5432084464862681,
            "total": 1.3152596270940062,
            "velocity": 0.44890336374041684
          },
          {
            "acceleration": 0.020415094910687274,
            "data": 0.2736427568298323,
            "posture": 0.551610916824866,
            "total": 1.301571167861293,
            "velocity": 0.4559023992959075
          },
          {
            "acceleration": 0.020441672288291978,
            "data": 0.2722296995719216,
            "posture": 0.5516448347560765,
            "total": 1.3003124835558348,
            "velocity": 0.45599627693954464
          },
          {
            "acceleration": 0.020459627793480463,
            "data": 0.27153426264645497,
            "posture": 0.5516723013416692,
            "total": 1.299714118749388,
            "velocity": 0.4560479269677834
          }
        ],
        "objective_terms": {
          "acceleration": 0.020459627793480463,
          "data": 0.27153426264645497,
          "posture": 0.5516723013416692,
          "total": 1.299714118749388,
          "velocity": 0.4560479269677834
        },
        "orientation_rmse_deg": 18.14301751422688,
        "position_p95_mm": 28.45577851619067,
        "position_rmse_mm": 18.400646538276636,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.7459677419354839,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.8901209677419355,
        "wrist_target_height_median_mm": 117.43583016248557
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_9_20231027_123725_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1134,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9320987654320988,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 21,
        "max_joint_acceleration_rad_s2": 2.103377134500627,
        "max_joint_velocity_rad_s": 0.8376729558533986,
        "min_collision_margin": 0.03932970325897105,
        "min_floor_margin_mm": 0.0001887289021937355,
        "min_joint_limit_margin_rad": 1.6772004805898908,
        "min_urdf_velocity_limit_margin_rad_s": 1.4377466066864155,
        "min_velocity_limit_margin_rad_s": 1.4377466066864155,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.1740469612527473,
        "objective_history": [
          {
            "acceleration": 0.0002766248872916912,
            "data": 33.029532674896565,
            "posture": 0.02096279266457221,
            "total": 33.051233493356065,
            "velocity": 0.00046140090764041806
          },
          {
            "acceleration": 0.00022511052874486104,
            "data": 25.585730481871884,
            "posture": 0.08219309311865208,
            "total": 25.672462015019942,
            "velocity": 0.004313329500661101
          },
          {
            "acceleration": 0.00020252386924652648,
            "data": 19.41879479415953,
            "posture": 0.178948610973041,
            "total": 19.602147955382133,
            "velocity": 0.00420202638031238
          },
          {
            "acceleration": 0.0006321727046445885,
            "data": 13.163084631796934,
            "posture": 0.22282393544092666,
            "total": 13.399655424767682,
            "velocity": 0.013114684825176
          },
          {
            "acceleration": 0.0010987423316332666,
            "data": 7.759677200216277,
            "posture": 0.3231379079432774,
            "total": 8.112532777645432,
            "velocity": 0.028618927154244973
          },
          {
            "acceleration": 0.0038522487824270177,
            "data": 4.6100729654080705,
            "posture": 0.42213833825412467,
            "total": 5.103829905329338,
            "velocity": 0.06776635288471601
          },
          {
            "acceleration": 0.004892492166622664,
            "data": 3.8167998001839645,
            "posture": 0.46388216390371795,
            "total": 4.378568967407475,
            "velocity": 0.09299451115317008
          },
          {
            "acceleration": 0.006445919983931317,
            "data": 3.2344775902518936,
            "posture": 0.49976903653824406,
            "total": 3.8610601796908806,
            "velocity": 0.12036763291681127
          },
          {
            "acceleration": 0.007966079167326192,
            "data": 2.7580092397766034,
            "posture": 0.5171107297604526,
            "total": 3.4347906245768187,
            "velocity": 0.15170457587243635
          },
          {
            "acceleration": 0.009126707007876992,
            "data": 2.3522351824991827,
            "posture": 0.5244091490411835,
            "total": 3.075630659657185,
            "velocity": 0.18985962110894175
          },
          {
            "acceleration": 0.010168993427214881,
            "data": 2.054723105791961,
            "posture": 0.5279923236018536,
            "total": 2.814646536963672,
            "velocity": 0.22176211414264257
          },
          {
            "acceleration": 0.010584989110411024,
            "data": 1.92525625011746,
            "posture": 0.5286606189198177,
            "total": 2.6974406968572007,
            "velocity": 0.2329388387095116
          },
          {
            "acceleration": 0.010998687127907913,
            "data": 1.798824979116858,
            "posture": 0.5301174067469671,
            "total": 2.5834480755982643,
            "velocity": 0.2435070026065314
          },
          {
            "acceleration": 0.011814422072679834,
            "data": 1.541024424151455,
            "posture": 0.5356078091500802,
            "total": 2.352803050024068,
            "velocity": 0.26435639464985333
          },
          {
            "acceleration": 0.013487292644748806,
            "data": 1.041885386904001,
            "posture": 0.5570312852317839,
            "total": 1.9172990066007483,
            "velocity": 0.30489504182021443
          },
          {
            "acceleration": 0.014819837168079386,
            "data": 0.6151217185282918,
            "posture": 0.5822627014166822,
            "total": 1.5431499952034495,
            "velocity": 0.330945738090396
          },
          {
            "acceleration": 0.015290291993815536,
            "data": 0.3326463541931468,
            "posture": 0.6102494864709607,
            "total": 1.2952507381799645,
            "velocity": 0.3370646055220415
          },
          {
            "acceleration": 0.015197338000285358,
            "data": 0.20292753680798997,
            "posture": 0.6334785103680937,
            "total": 1.2000191046864765,
            "velocity": 0.3484157195101075
          },
          {
            "acceleration": 0.015134550922558257,
            "data": 0.16059883608515968,
            "posture": 0.6429091553366091,
            "total": 1.1755738339977357,
            "velocity": 0.35693129165340864
          },
          {
            "acceleration": 0.015129374803663291,
            "data": 0.15962122411730584,
            "posture": 0.6429262051981082,
            "total": 1.174552119979288,
            "velocity": 0.3568753158602107
          },
          {
            "acceleration": 0.015127093402115292,
            "data": 0.1591458534857241,
            "posture": 0.6429438030848639,
            "total": 1.1740469612527473,
            "velocity": 0.35683021128004416
          }
        ],
        "objective_terms": {
          "acceleration": 0.015127093402115292,
          "data": 0.1591458534857241,
          "posture": 0.6429438030848639,
          "total": 1.1740469612527473,
          "velocity": 0.35683021128004416
        },
        "orientation_rmse_deg": 11.11362746496737,
        "position_p95_mm": 20.98688945823313,
        "position_rmse_mm": 12.790475064012071,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9029982363315696,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9329805996472663,
        "wrist_target_height_median_mm": 118.56606360401508
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_9_20231027_124057_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1136,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.9234154929577465,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 24,
        "max_joint_acceleration_rad_s2": 1.9533122024653566,
        "max_joint_velocity_rad_s": 0.809498288353086,
        "min_collision_margin": 0.03940673507844711,
        "min_floor_margin_mm": 0.0018797948242374463,
        "min_joint_limit_margin_rad": 1.728190713835994,
        "min_urdf_velocity_limit_margin_rad_s": 1.5028617693110218,
        "min_velocity_limit_margin_rad_s": 1.5028617693110218,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.3332815393852804,
        "objective_history": [
          {
            "acceleration": 0.00034885484596563506,
            "data": 34.51079627181621,
            "posture": 0.021032291849557322,
            "total": 34.53272327229064,
            "velocity": 0.0005458537789034982
          },
          {
            "acceleration": 0.0004193949558746987,
            "data": 27.051924954071556,
            "posture": 0.08360267061336063,
            "total": 27.14310039407177,
            "velocity": 0.007153374430981378
          },
          {
            "acceleration": 0.0004945821642115458,
            "data": 20.824260300175887,
            "posture": 0.18039919687003855,
            "total": 21.01346408398084,
            "velocity": 0.008310004770704175
          },
          {
            "acceleration": 0.0009337742813987643,
            "data": 14.512197764003641,
            "posture": 0.22749089346281623,
            "total": 14.757648369707187,
            "velocity": 0.01702593795933142
          },
          {
            "acceleration": 0.0013849919300166023,
            "data": 8.8423593083042,
            "posture": 0.33085713412180584,
            "total": 9.201766841930276,
            "velocity": 0.027165407574252967
          },
          {
            "acceleration": 0.0022333265303570785,
            "data": 6.741122872446255,
            "posture": 0.38076406094453696,
            "total": 7.159085640350185,
            "velocity": 0.03496538042903642
          },
          {
            "acceleration": 0.006747709148240511,
            "data": 3.7551454611224813,
            "posture": 0.47895027801421514,
            "total": 4.330995326339966,
            "velocity": 0.09015187805502983
          },
          {
            "acceleration": 0.008297837849341055,
            "data": 3.056492680602706,
            "posture": 0.5092223665895254,
            "total": 3.6970876541649336,
            "velocity": 0.12307476912336104
          },
          {
            "acceleration": 0.010067840795806995,
            "data": 2.547229131895932,
            "posture": 0.5251082213523188,
            "total": 3.246544067123862,
            "velocity": 0.16413887307980396
          },
          {
            "acceleration": 0.011530040106658116,
            "data": 2.18942981914822,
            "posture": 0.5349183931141228,
            "total": 2.9465231207673943,
            "velocity": 0.21064486839839353
          },
          {
            "acceleration": 0.01282211096159285,
            "data": 1.939855156441148,
            "posture": 0.5370475096513949,
            "total": 2.737380940782189,
            "velocity": 0.24765616372805374
          },
          {
            "acceleration": 0.013352994881256346,
            "data": 1.8304451813880251,
            "posture": 0.5369718618880621,
            "total": 2.642591357625358,
            "velocity": 0.26182131946801473
          },
          {
            "acceleration": 0.013840640929701446,
            "data": 1.723809851033903,
            "posture": 0.5374600418213649,
            "total": 2.5502437551045487,
            "velocity": 0.2751332213195795
          },
          {
            "acceleration": 0.014772689329392954,
            "data": 1.5050453164024904,
            "posture": 0.5409329217012051,
            "total": 2.3618492981224453,
            "velocity": 0.3010983706893569
          },
          {
            "acceleration": 0.01663605152625116,
            "data": 1.0672448848100888,
            "posture": 0.5582426228275734,
            "total": 1.9918031104041372,
            "velocity": 0.34967955124022365
          },
          {
            "acceleration": 0.018157079384689325,
            "data": 0.6778392310759369,
            "posture": 0.5797582077639251,
            "total": 1.6490643232625302,
            "velocity": 0.37330980503797906
          },
          {
            "acceleration": 0.019018910623309494,
            "data": 0.40889429252447307,
            "posture": 0.6086796349498899,
            "total": 1.426853883663798,
            "velocity": 0.3902610455661256
          },
          {
            "acceleration": 0.0188977991828583,
            "data": 0.3008354052414492,
            "posture": 0.6325703631113222,
            "total": 1.356071031641184,
            "velocity": 0.4037674641055543
          },
          {
            "acceleration": 0.01886462881129609,
            "data": 0.2880083449024975,
            "posture": 0.633916614697866,
            "total": 1.3432512516867163,
            "velocity": 0.4024616632750568
          },
          {
            "acceleration": 0.018864058417191982,
            "data": 0.2827627135555904,
            "posture": 0.6339952872196749,
            "total": 1.337977478867336,
            "velocity": 0.40235541967487853
          },
          {
            "acceleration": 0.018865465799298292,
            "data": 0.2802034826296469,
            "posture": 0.6340693421622853,
            "total": 1.3354104471357156,
            "velocity": 0.402272156544485
          },
          {
            "acceleration": 0.018867108480927777,
            "data": 0.2789009068755013,
            "posture": 0.6341630336166727,
            "total": 1.3341057595454267,
            "velocity": 0.40217471057232484
          },
          {
            "acceleration": 0.018868921745041043,
            "data": 0.2782756356471834,
            "posture": 0.6342420934222114,
            "total": 1.3334868501537573,
            "velocity": 0.4021001993393216
          },
          {
            "acceleration": 0.01886802083316193,
            "data": 0.2780641963133389,
            "posture": 0.6342750958161375,
            "total": 1.3332815393852804,
            "velocity": 0.40207422642264184
          }
        ],
        "objective_terms": {
          "acceleration": 0.01886802083316193,
          "data": 0.2780641963133389,
          "posture": 0.6342750958161375,
          "total": 1.3332815393852804,
          "velocity": 0.40207422642264184
        },
        "orientation_rmse_deg": 15.55688569232866,
        "position_p95_mm": 25.53498695447907,
        "position_rmse_mm": 15.394811521197582,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9242957746478874,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.920774647887324,
        "wrist_target_height_median_mm": 119.62990070642395
      },
      "solver_status": "converged",
      "status": "complete"
    },
    "subject_9_20231027_125315_105322251564-043422252387-037522251142-105322251225-108222250342-117222250549-046122250168-115422250549": {
      "config": {
        "adapter": {
          "kind": "user_cylinder",
          "radius_m": 0.035,
          "rpy_deg": [
            0.0,
            0.0,
            0.0
          ],
          "translation_m": [
            0.0,
            0.0,
            0.011
          ]
        },
        "approach_sec": 0.0,
        "base_position": [
          0.0,
          -1.0,
          0.0
        ],
        "base_rpy_deg": [
          0.0,
          0.0,
          0.0
        ],
        "collision_min_distance_m": 0.02,
        "collision_pairs": 8,
        "gripper": "robotiq_2f85",
        "hand_to_ee_rpy_deg": [
          4.325,
          15.45,
          -47.922
        ],
        "hand_to_ee_translation": [
          0.0,
          0.0,
          0.0
        ],
        "object_cube_side_mm": 40.0,
        "object_grasp": false,
        "pose_source": "hand_fit",
        "reference_policy": "zero",
        "robot": "ur10",
        "sequential_baseline": false,
        "solver": {
          "collision_enabled": true,
          "confidence_floor": 0.05,
          "convergence_rad": 0.001,
          "huber_delta": 0.05,
          "lm_damping": 1e-05,
          "max_iterations": 40,
          "max_step_rad": 0.2,
          "qp_solver": "osqp"
        },
        "table_plane_z_m": 0.0,
        "target_position_anchor": "pinch_center",
        "velocity_limit_scale": 1.0,
        "weights": {
          "acceleration": 2e-05,
          "orientation": 0.01,
          "position": 1.0,
          "posture": 0.0001,
          "velocity": 0.002
        }
      },
      "coordinate_frame": "hocap_tag_1",
      "frames": 1136,
      "metrics": {
        "achieved_tool_axis_down_fraction": 0.983274647887324,
        "adapter_length_mm": 11.0,
        "approach_duration_sec": 0.0,
        "converged": true,
        "iterations": 23,
        "max_joint_acceleration_rad_s2": 1.466467911993297,
        "max_joint_velocity_rad_s": 0.8269236095173436,
        "min_collision_margin": 0.03961133676775676,
        "min_floor_margin_mm": 0.00021554257972206115,
        "min_joint_limit_margin_rad": 1.702632162580971,
        "min_urdf_velocity_limit_margin_rad_s": 1.5638732647937237,
        "min_velocity_limit_margin_rad_s": 1.5638732647937237,
        "object_grasp": null,
        "object_grasp_carry_width_m": null,
        "objective": 1.077802647897971,
        "objective_history": [
          {
            "acceleration": 0.0003441982809085741,
            "data": 33.03180377759792,
            "posture": 0.02104243934431074,
            "total": 33.05381606838092,
            "velocity": 0.0006256531577791091
          },
          {
            "acceleration": 0.00015945018628303926,
            "data": 25.590135763343923,
            "posture": 0.08309022891901226,
            "total": 25.677190658811902,
            "velocity": 0.0038052163626815017
          },
          {
            "acceleration": 0.00020028031103337104,
            "data": 19.447441103760077,
            "posture": 0.1799137011872437,
            "total": 19.631131602808633,
            "velocity": 0.003576517550280562
          },
          {
            "acceleration": 0.0003979096536532639,
            "data": 13.112473098972226,
            "posture": 0.2223638486679965,
            "total": 13.343661494418116,
            "velocity": 0.008426637124241859
          },
          {
            "acceleration": 0.001011344077016038,
            "data": 7.659428245241958,
            "posture": 0.3213154237620508,
            "total": 8.011288395328043,
            "velocity": 0.029533382247017098
          },
          {
            "acceleration": 0.0017475520273944195,
            "data": 5.900642612298158,
            "posture": 0.3637743284890936,
            "total": 6.314803455752666,
            "velocity": 0.04863896293801926
          },
          {
            "acceleration": 0.002959158613490175,
            "data": 4.539947344439501,
            "posture": 0.4084639121243794,
            "total": 5.02386863556322,
            "velocity": 0.0724982203858493
          },
          {
            "acceleration": 0.0042478892077744065,
            "data": 3.585062713877918,
            "posture": 0.44912416977753095,
            "total": 4.1325484261736305,
            "velocity": 0.0941136533104071
          },
          {
            "acceleration": 0.005515420891049291,
            "data": 2.9434921023721365,
            "posture": 0.4742467709528748,
            "total": 3.543299949901663,
            "velocity": 0.12004565568560199
          },
          {
            "acceleration": 0.006593374720636071,
            "data": 2.459931739821123,
            "posture": 0.48522727417634043,
            "total": 3.1049868577775435,
            "velocity": 0.15323446905944385
          },
          {
            "acceleration": 0.0073376455824205565,
            "data": 2.092052643748282,
            "posture": 0.4942184945088092,
            "total": 2.7808415789110428,
            "velocity": 0.18723279507153126
          },
          {
            "acceleration": 0.00807249884862255,
            "data": 1.8108363062549495,
            "posture": 0.5007135994102164,
            "total": 2.5315822094162344,
            "velocity": 0.21195980490244573
          },
          {
            "acceleration": 0.008749827633149361,
            "data": 1.5566232166273177,
            "posture": 0.5078864945703324,
            "total": 2.3058184416937406,
            "velocity": 0.232558902862941
          },
          {
            "acceleration": 0.010096194836897188,
            "data": 1.067080603852606,
            "posture": 0.530799296205282,
            "total": 1.881257940506491,
            "velocity": 0.27328184561170593
          },
          {
            "acceleration": 0.011245494312515337,
            "data": 0.6320522145092141,
            "posture": 0.5584388896339019,
            "total": 1.5042175073367559,
            "velocity": 0.30248090888112455
          },
          {
            "acceleration": 0.011697027406355117,
            "data": 0.32970398165273396,
            "posture": 0.5848724530365387,
            "total": 1.2371655976256146,
            "velocity": 0.31089213552998685
          },
          {
            "acceleration": 0.011326992788748958,
            "data": 0.1836901916313282,
            "posture": 0.6063898933588221,
            "total": 1.114856759686345,
            "velocity": 0.31344968190744593
          },
          {
            "acceleration": 0.011132107180067135,
            "data": 0.15741350426472803,
            "posture": 0.6118221814380711,
            "total": 1.091477175289484,
            "velocity": 0.3111093824066179
          },
          {
            "acceleration": 0.011073020525629392,
            "data": 0.14718012712492926,
            "posture": 0.6144306226152952,
            "total": 1.0831180117169472,
            "velocity": 0.3104342414510934
          },
          {
            "acceleration": 0.011068202082617494,
            "data": 0.14414764576059325,
            "posture": 0.6145522135304131,
            "total": 1.0801216765765995,
            "velocity": 0.3103536152029756
          },
          {
            "acceleration": 0.011066244887185385,
            "data": 0.14262943525029353,
            "posture": 0.6146519365328488,
            "total": 1.0786466241699042,
            "velocity": 0.3102990074995765
          },
          {
            "acceleration": 0.011065152825667578,
            "data": 0.14184606623469911,
            "posture": 0.6147497519613427,
            "total": 1.0779140354250885,
            "velocity": 0.310253064403379
          },
          {
            "acceleration": 0.011064729259963917,
            "data": 0.1417224681980888,
            "posture": 0.6147696190870146,
            "total": 1.077802647897971,
            "velocity": 0.31024583135290357
          }
        ],
        "objective_terms": {
          "acceleration": 0.011064729259963917,
          "data": 0.1417224681980888,
          "posture": 0.6147696190870146,
          "total": 1.077802647897971,
          "velocity": 0.31024583135290357
        },
        "orientation_rmse_deg": 10.11496172169536,
        "position_p95_mm": 21.020995122616497,
        "position_rmse_mm": 12.017119952236769,
        "reference_policy": "zero",
        "reference_q": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "sequential_baseline": null,
        "table_plane_coordinate_frame": "hocap_tag_1",
        "table_plane_z_calibration_m": 0.0,
        "target_position_anchor": "pinch_center",
        "target_tool_axis_down_fraction": 0.9269366197183099,
        "velocity_limit_scale": 1.0,
        "velocity_repair_max_rad": 0.0,
        "wrist_above_target_fraction": 0.9841549295774648,
        "wrist_target_height_median_mm": 123.08023190977036
      },
      "solver_status": "converged",
      "status": "complete"
    }
  },
  "protocol": "HO-Cap hand-only UR10/Robotiq, base [0,-1,0] in hocap_tag_1, table z=0, zero IK reference, no home approach",
  "workers": 4
}
```
