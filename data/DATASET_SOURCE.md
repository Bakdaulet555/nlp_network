# Источник NLP-данных

Источники train:

* [Morzeux/HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset/),
  `payload_full.csv`, лицензия MIT;
* [fuzzdb](https://github.com/fuzzdb-project/fuzzdb), XSS-файлы,
  New BSD / CC BY с атрибуцией.

Для проекта используются записи `norm`, `sqli` и `xss`. Категории `cmdi` и
`path-traversal` исключены, поскольку модель решает трёхклассовую задачу:
`Normal / SQL Injection / XSS`.

После нормализации формируется по 900 строк каждого класса. Разбиение выполняется
внутри этого набора по группам похожих шаблонов: одна группа не может одновременно
попасть в train и test. Выборка детерминирована (`random_state=42`).

Подготовка воспроизводится командой:

```powershell
python scripts/prepare_http_params_dataset.py
```

`fp_benign_requests.csv` является отдельным набором трудных нормальных запросов
и используется только для независимой проверки ложных срабатываний.
