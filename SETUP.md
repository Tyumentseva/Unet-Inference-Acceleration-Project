[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)

### Установка интерпретатора и пакетов

Для воспроизвоимости экспериментов рекомендуется использовать версию питона 3.13.7.  
В качестве пакетного менеджера - uv.

1. Установите пакетный менеджер uv, [официальная инструкция](https://docs.astral.sh/uv/getting-started/installation/#installing-uv)

2. Разверните виртуальное окружение с нужной версией питона при помощи [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) в репозитории с задачами
  ```bash
  $ cd <путь к склонированному репозиторию>
  $ uv sync
  ```

3. Активируйте виртуальное окружение (будет активным, пока не закроете консоль, либо не выполните `deactivate`)
  ```bash
  $ source .venv/bin/activate
  ```  
  Появится следующий префикс:
  ```bash
  (unet_env)$ ...
  ```

4. Проверьте, что всё активировалось и ссылка `python` ведёт на папку с `unet_env`
  ```bash
  (unet_env)$ which python
  <директория с репозиторием>/unet_env/bin/python
  (unet_env)$ python --version
  Python 3.13.7
  ```

5. Работа с пакетами

```bash
uv add <package> или с версией uv add requests==2.31.0
uv remove <package>
uv lock
uv sync
```

### Запуск линтера

```bash
uv run ruff check .
uv run ruff check . --fix
```