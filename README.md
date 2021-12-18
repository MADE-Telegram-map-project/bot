# Анализ данных

## Установить зависимости

```
pip install -r ./requirements.txt
```

## Скачать данные из DVC

[Добавить в локальный конфигурационный файл DVC ключ с секретом](https://dvc.org/doc/user-guide/setup-google-drive-remote#using-service-accounts)

**Устанавливать повторно gdrive_use_service_account не нужно**

Выполнить команду:
```
dvc pull
```

## Data

raw data is in data/raw_16k

[SQL Запрос на вычисление пересечения каналов по пользователям](https://gist.github.com/KernelA/22a65f1631c3b586c34d32c63fbd6141)


## Пакет для логгирования

[См. README.md](logger/README.md)


## Вычисление эмбеддингов
```
python calculate_embeddings_transformer.py
```