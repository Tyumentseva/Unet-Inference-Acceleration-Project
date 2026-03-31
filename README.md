# Описание проекта

Описание проекта можно найти на [странице описания проектов](https://docs.google.com/spreadsheets/d/1qPBzSfBVwHwDUPWAHXLmdiwRpu3rGWMuPPPK7FR7lVk/edit?gid=0#gid=0) курса ШАД [Эффективные модели ML и архитектуры нейросетей](https://lk.dataschool.yandex.ru/courses/2026-spring/7.1689-efficient-ml-model/).

Цель проекта — ускорение инференса UNet-модели. В качестве бейзлайна рассматривается инференс на PyTorch с вычислениями в формате fp16. Для ускорения инференса относительно бейзлайна применяются: 
- PyTorch torch.compile (JIT-компиляция графа вычислений); 
- компилятор TVM (аппаратно-независимая оптимизация через Relax и TIR); 
- квантизация (снижение точности весов и активаций); 
- спарсификация: pruning / 2:4 semi-structured (разреживание весовой матрицы).

Вклад квантизации и прунинга оценивается отдельно посредством профилирования и ablation-study.

# Структура проекта

```
project
├── notebooks/
│   ├── experiments/  # блокноты с кодом экспериментов
│   │   ├── images/
│   │   └── torch_compile.ipynb
│   └── model/  # блокноты по обучению модели и её тестовом запуске
│       ├── test/
│       │   └── load_baseline.ipynb
│       └── train/
│           └── train_COCO.ipynb
├── src/  # скрипты по замеру параметров инференса модели
│   ├── benchmark/
│   │   ├── benchmarker.py
│   │   ├── measurement_strategy.py
│   │   └── __init__.py
│   ├── profile/
│   │   ├── profiler.py
│   │   └── __init__.py
│   ├── flops/
│   │   ├── flops_counter.py
│   │   └── __init__.py
│   └── __init__.py
├── weights/  # веса модели
│   └── checkpoint_coco_fp16.pt
├── main.py
├── pyroject.toml
├── uv.lock
├── SETUP.md
├── RESULT.md
└── README.md
```
# Настройка окружения
Инструкция по запуску скриптов подробно изложена в ```SETUP.md``` и содержит:
1. Как поставить менеджера пакектов [uv](https://github.com/astral-sh/uv) и управлять окружением и зависимостями через `pyproject.toml`
2. Как создать виртуальное окружение с нужной версией питона
3. Как установить питон пакеты через ```uv```.

# Пререквизиты
Модель Unet [получена дообучением](notebooks/model/train/train_COCO.ipynb) предобученной модели из smp на COCO-датасете.
Обучение проходило c точностью bfloat16.
Веса энкодера в процессе обучения не менялись (были заморожены).
Оценка качества полученной модели [проводилась](notebooks/model/test/load_baseline.ipynb) на валидационном COCO-датасете.

1. Unet из [segmentation_models_pytorch](https://pypi.org/project/segmentation-models-pytorch/0.5.0/):
   - энкодер: resnet101
   - датасет предобучения энкодера: imagenet
   - обучаемые веса: декодер 
2. Датасет COCO:
   - train: [изображения](http://images.cocodataset.org/zips/train2017.zip), [аннотации](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
   - valid: [изображения](http://images.cocodataset.org/zips/val2017.zip), [аннотации](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
3. Аппаратное обеспечение:
   - CPU:
   - GPU:

# Описание экспериментов
В таблицах ниже представлены группы планируемых экспериментов по измерению end-to-end latency и throughput для всех конфигураций проекта. Качество модели будет оцениваться относительно fp16 baseline с использованием метрики IoU.

TODO: обновлять по мере погружения в материал

### 1. Компиляторные оптимизации

| Конфигурация | Latency (ms) | Throughput (img/s) при batch_size=16 | IoU  |
|--------------|--------------|--------------------------------------|------|
| **Baseline** |              |                                      |      |
| PyTorch fp16 (baseline) | 19.6         | 585                                  | 45.2 |
| **Компиляторные оптимизации** |              |                                      |      |
| PyTorch torch.compile (mode="default") | 5.7          | 972                                  | 45.2 |
| PyTorch torch.compile (mode="max-autotune") | 2.6          | 1076                                 | 45.2 |
| TVM (Relax/TIR) | 26.7         | 83                                   | 45.2 |
| |              |                                      |      |

### 2. Пост-тренировочные оптимизации

| Конфигурация | Latency (ms) | Throughput (img/s) при batch_size=16  | IoU |
|--------------|--------------|---------------------------------------|--------|
| **Baseline** | |                                       | |
| PyTorch fp16 (baseline) | 19.6 | 585                                | 45.2 |
| **Пост-тренировочная оптимизация** | |                                       | |
| Квантизация (int8, per-tensor) | |                                       | |
| Квантизация (int8, per-channel) | |                                       | |
| Спарсификация (2:4 semi-structured) | |                                       | |
| Спарсификация + квантизация | |                                       | |
| | |                                       | |

### 3. Комбинированные методы

| Конфигурация | Latency (ms) | Throughput (img/s)  при batch_size=16 | IoU |
|--------------|--------------|---------------------------------------|--------|
| **Baseline** | |                                       | |
| PyTorch fp16 (baseline) | 19.6 | 585                                   | 45.2 |
| **Комбинированные методы** | |                                       | |
| torch.compile + квантизация | |                                       | |
| TVM + спарсификация | |                                       | |
| | |                                       | |

# Результаты
Результаты всех экспериментов подробно изложены в [файле](RESULT.md).

Эксперимент со следующей конфигурацией (TODO: описать конфигурацию):
- параметр 1
- параметр 2
- ...
  
показал лучшие результаты: (TODO: описать результаты лучшего эксперимента)
