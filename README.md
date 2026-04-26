# Описание проекта

Описание проекта можно найти на [странице описания проектов](https://docs.google.com/spreadsheets/d/1qPBzSfBVwHwDUPWAHXLmdiwRpu3rGWMuPPPK7FR7lVk/edit?gid=0#gid=0) курса ШАД [Эффективные модели ML и архитектуры нейросетей](https://lk.dataschool.yandex.ru/courses/2026-spring/7.1689-efficient-ml-model/).

Цель проекта — ускорение инференса UNet-модели. В качестве бейзлайна рассматривается инференс на PyTorch с вычислениями в формате bf16. Для ускорения инференса относительно бейзлайна применяются: 
- PyTorch torch.compile (JIT-компиляция графа вычислений); 
- компилятор TVM (аппаратно-независимая оптимизация через Relax и TIR); 
- квантизация (снижение точности весов и активаций); 
- спарсификация: pruning / 2:4 semi-structured (разреживание весовой матрицы).

Вклад компиляции, квантизации, прунинга и их комбинаций оценивается посредством профилирования, измерения качества и времени обработки изображения.

# Структура проекта

```
project
├── notebooks/
│   ├── experiments/  # блокноты с кодом экспериментов
│   │   ├── images/
│   │   ├── traces/
│   │   ├── LSQ_inference.ipynb
│   │   ├── LSQuantization_8bit_conv2d_inference.ipynb
│   │   ├── LSQuantization_8bit_triton_inference.ipynb
│   │   ├── torch_compile.ipynb
│   │   └── torch_compile_bf16_pruning_structured.ipynb
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
│   │   ├── profile.py
│   │   └── __init__.py
│   ├── model/
│   │   ├── lsq_quantization/
│       │   ├── lsq_inference_conv2d.py
│       │   ├── lsq_inference_triton.py
│       │   ├── lsq_train_conv2d.py
│       │   └── __init__.py
│   │   └── __init__.py
│   └── __init__.py
├── weights/  # веса модели
│   └── checkpoint_coco_fp16.pt
├── unet_inference.pptx
├── unet_inference.pdf
├── pyroject.toml
├── uv.lock
├── SETUP.md
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

# Описание экспериментов
В таблицах ниже представлены группы планируемых экспериментов по измерению end-to-end latency и throughput для всех конфигураций проекта. Качество модели оценивается относительно fp16 baseline с использованием метрики IoU.

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

### 2. Оптимизации aware training

| Конфигурация | Latency (ms) | Throughput (img/s) при batch_size=16 | IoU |
|--------------|--------------|---------------------------------------|-----|
| **Baseline** |              |                                       |     |
| PyTorch fp16 (baseline) | 19.6 | 585 | 45.2 |
| **Aware-training оптимизация** | | | |
| LSQ - conv2d -квантизация | 29.2 | 400 | 0.87 |
| LSQ - triton -квантизация | 400 | 150 | 47.5 |
| Прунинг | 19.6 | 600 | 46.0 |

### 3. Комбинированные методы

| Конфигурация | Latency (ms) | Throughput (img/s) при batch_size=16 | IoU |
|--------------|--------------|---------------------------------------|-----|
| **Baseline** |              |                                       |     |
| PyTorch fp16 (baseline) | 19.6 | 585 | 45.2 |
| **Комбинированные методы** | | | |
| torch.compile(mode="default") + triton-lsq-квантизация | 17.8 | 205 | 47.5 |
| torch.compile(mode="max-autotune") + triton-lsq-квантизация | 17.3 | 205 | 47.5 |
| torch.compile(mode="default") + conv2d-lsq-квантизация | 5.2 | 870 | 0.87 |
| torch.compile(mode="max-autotune") + conv2d-lsq-квантизация | 4.3 | 880 | 0.87 |
| torch.compile(mode="default") + прунинг | 5.4 | 1000 | 46.0 |
| torch.compile(mode="max-autotune") + прунинг | 2.3 | 1200 | 45.9 |
