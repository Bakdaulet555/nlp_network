"""
Общая конфигурация проекта AI Security Monitor.

Все пути, имена столбцов и пороги собраны в одном месте, чтобы оба модуля
(сетевой и NLP) и панель мониторинга использовали одинаковые названия.
Это и есть «контракт» между Студентом 1 и Студентом 2.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Пути
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

NETWORK_DATA_PATH = DATA_DIR / "network_traffic.csv"
TEXT_DATA_PATH = DATA_DIR / "http_requests.csv"

ISOFOREST_MODEL_PATH = MODELS_DIR / "network_isoforest.joblib"
RF_MODEL_PATH = MODELS_DIR / "network_random_forest.joblib"
NLP_MODEL_PATH = MODELS_DIR / "nlp_sgd_classifier.joblib"
NLP_NB_MODEL_PATH = MODELS_DIR / "nlp_complement_nb.joblib"

NETWORK_EVENTS_PATH = OUTPUTS_DIR / "network_events.csv"
HTTP_EVENTS_PATH = OUTPUTS_DIR / "http_events.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# --------------------------------------------------------------------------
# Модуль 1. Сетевой трафик (подмножество столбцов CIC-IDS2017)
# В оригинальных CSV CIC-IDS2017 у многих столбцов есть ведущий пробел
# (" Flow Duration"), поэтому загрузчик всегда делает strip() имён.
# --------------------------------------------------------------------------
NETWORK_LABEL_COL = "Label"
NETWORK_TIME_COL = "Timestamp"          # необязательный столбец
NETWORK_BENIGN_LABEL = "BENIGN"

NETWORK_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
]

# Isolation Forest обучается только на нормальном трафике, поэтому
# contamination ≈ допустимая доля ложных срабатываний на норме.
# Значение по умолчанию; при наличии меток подбирается на валидационной
# части train (см. tune_isoforest_contamination), НЕ на тестовой выборке.
ISOFOREST_CONTAMINATION = 0.05

# --------------------------------------------------------------------------
# Модуль 2. Текстовые HTTP-запросы
# --------------------------------------------------------------------------
TEXT_COL = "request"
TEXT_LABEL_COL = "label"
TEXT_TIME_COL = "timestamp"             # необязательный столбец
TEXT_SPLIT_COL = "split"                # необязательный: train / test
TEXT_CLASSES = ["Normal", "SQL Injection", "XSS"]
MAX_REQUEST_LENGTH = 4096               # длиннее — обрезаем перед анализом

# Если уверенность модели ниже порога, событие помечается «требует проверки»
NLP_REVIEW_THRESHOLD = 0.60

# --------------------------------------------------------------------------
# Единая схема событий для панели (контракт между модулями)
# --------------------------------------------------------------------------
EVENT_COLUMNS = [
    "event_id",       # str: уникальный идентификатор (net-000001 / http-000001)
    "timestamp",      # str ISO 8601 или пусто, если времени нет в данных
    "source_module",  # "network" | "http"
    "model_name",     # IsolationForest | RandomForest | SGDClassifier
    "prediction",     # категория по мнению модели
    "score",          # float 0..1: уверенность / нормированная аномальность
    "is_alert",       # bool: модель считает событие подозрительным
    "needs_review",   # bool: низкая уверенность → ручная проверка
    "ground_truth",   # метка из датасета, если есть (иначе пусто)
    "details",        # краткое описание (порт, фрагмент запроса)
    "status",         # всегда «Прогноз модели» — не подтверждённая атака
]
PREDICTION_STATUS = "Прогноз модели (не подтверждено)"

# --------------------------------------------------------------------------
# Безопасность загрузки файлов
# --------------------------------------------------------------------------
MAX_UPLOAD_MB = 20
MAX_UPLOAD_ROWS = 200_000
ALLOWED_EXTENSIONS = {".csv"}
