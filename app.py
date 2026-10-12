"""
Модуль 3. Единая панель мониторинга AI Security Monitor (Streamlit).

Запуск:  streamlit run app.py

Принципы:
* все результаты моделей подписаны как «Прогноз модели (не подтверждено)»;
* пользовательский текст выводится только через st.dataframe / st.code /
  st.text (экранирование), unsafe_allow_html НЕ используется;
* загружаются только CSV с ограничением размера и проверкой столбцов;
* модели читаются только из локального каталога models/, загрузка
  моделей пользователем не предусмотрена (pickle небезопасен).
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import config
from src.common.io_utils import DataValidationError, load_events, to_csv_bytes

st.set_page_config(page_title="AI Security Monitor", page_icon="🛡️", layout="wide")
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Загрузка моделей и событий
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path_str: str):
    from pathlib import Path
    path = Path(path_str)
    # загружаем только из каталога models/ проекта
    if path.parent.resolve() != config.MODELS_DIR.resolve() or not path.exists():
        return None
    return joblib.load(path)


def get_events(module: str) -> pd.DataFrame:
    key = f"{module}_events"
    if key not in st.session_state:
        path = config.NETWORK_EVENTS_PATH if module == "network" else config.HTTP_EVENTS_PATH
        st.session_state[key] = load_events(path)
    return st.session_state[key]


def set_events(module: str, df: pd.DataFrame) -> None:
    st.session_state[f"{module}_events"] = df


iso_model = load_model(str(config.ISOFOREST_MODEL_PATH))
rf_model = load_model(str(config.RF_MODEL_PATH))
nlp_model = load_model(str(config.NLP_MODEL_PATH))

st.title("🛡️ AI Security Monitor")
st.caption("Учебный прототип: мониторинг сетевой активности и NLP-классификация HTTP-запросов. "
           "Все результаты — прогнозы ML-моделей и требуют проверки аналитиком.")

tab_overview, tab_net, tab_http, tab_eval, tab_about = st.tabs(
    ["📊 Обзор", "🌐 Сетевой трафик", "🔤 HTTP-запросы (NLP)", "📈 Качество моделей", "ℹ️ О системе"])


# ==========================================================================
# Обзор
# ==========================================================================
with tab_overview:
    net = get_events("network")
    http = get_events("http")
    events = pd.concat([net, http], ignore_index=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сетевых записей", f"{len(net):,}".replace(",", " "))
    c2.metric("Аномалий (IF)", int(net["is_alert"].sum()) if len(net) else 0)
    c3.metric("Запросов Normal", int((http["prediction"] == "Normal").sum()) if len(http) else 0)
    c4.metric("SQL Injection", int((http["prediction"] == "SQL Injection").sum()) if len(http) else 0)
    c5.metric("XSS", int((http["prediction"] == "XSS").sum()) if len(http) else 0)

    if events.empty:
        st.info("Событий пока нет. Обучите модели (`python train_all.py`) или загрузите CSV "
                "на вкладках «Сетевой трафик» и «HTTP-запросы».")
    else:
        st.markdown("##### Распределение событий по категориям (прогноз моделей)")
        g1, g2 = st.columns(2)
        for col, module, title in [(g1, "network", "Сетевой модуль"), (g2, "http", "HTTP-модуль")]:
            sub = events[events["source_module"] == module]
            if sub.empty:
                col.caption(f"{title}: нет данных")
                continue
            fig, ax = plt.subplots(figsize=(5, 2.8))
            counts = sub["prediction"].value_counts()
            sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4C72B0")
            ax.set_title(title); ax.set_xlabel("Количество"); ax.set_ylabel("")
            fig.tight_layout(); col.pyplot(fig); plt.close(fig)

        st.markdown("##### Журнал событий")
        f1, f2, f3, f4 = st.columns([1, 1.4, 1, 1.6])
        modules = f1.multiselect("Модуль", ["network", "http"], default=["network", "http"])
        cats = f2.multiselect("Категория", sorted(events["prediction"].unique()))
        result = f3.selectbox("Результат", ["Все", "Только предупреждения", "Требуют проверки"])
        ts = pd.to_datetime(events["timestamp"], errors="coerce")
        time_range = None
        if ts.notna().any():
            lo, hi = ts.min().to_pydatetime(), ts.max().to_pydatetime()
            if lo < hi:
                time_range = f4.slider("Время", min_value=lo, max_value=hi, value=(lo, hi),
                                       format="DD.MM HH:mm")

        view = events[events["source_module"].isin(modules)]
        if cats:
            view = view[view["prediction"].isin(cats)]
        if result == "Только предупреждения":
            view = view[view["is_alert"]]
        elif result == "Требуют проверки":
            view = view[view["needs_review"]]
        if time_range:
            vts = pd.to_datetime(view["timestamp"], errors="coerce")
            view = view[vts.between(*time_range) | vts.isna()]

        view = view.sort_values("timestamp", ascending=False)
        st.caption(f"Показано {min(len(view), 500)} из {len(view)} событий (сначала последние).")
        st.dataframe(view.head(500), width="stretch", hide_index=True)
        st.download_button("⬇️ Выгрузить отфильтрованные события (CSV)", to_csv_bytes(view),
                           "events_filtered.csv", "text/csv")


# ==========================================================================
# Сетевой трафик
# ==========================================================================
with tab_net:
    from src.network.models import analyze_network, network_events
    from src.network.preprocess import clean_network_data, load_network_csv

    st.subheader("Мониторинг сетевой активности")
    if iso_model is None:
        st.warning("Модель Isolation Forest не найдена. Выполните `python train_all.py --only network`.")
    else:
        up = st.file_uploader("Загрузите CSV с сетевыми потоками (столбцы CIC-IDS2017)",
                              type=["csv"], key="net_upload",
                              help=f"До {config.MAX_UPLOAD_MB} МБ. Обязательные столбцы: "
                                   + ", ".join(config.NETWORK_FEATURES))
        if up is not None and st.button("Проанализировать сетевой файл"):
            try:
                raw = load_network_csv(up, filename=up.name)
                clean, quality = clean_network_data(raw)
                analyzed = analyze_network(clean, iso_model, rf_model)
                st.session_state["net_analyzed"] = analyzed
                st.session_state["net_quality"] = quality
                set_events("network", network_events(analyzed))
                st.success(f"Проанализировано {len(analyzed)} записей.")
            except DataValidationError as exc:
                st.error(f"Файл отклонён: {exc}")

        analyzed = st.session_state.get("net_analyzed")
        if analyzed is None and config.NETWORK_DATA_PATH.exists():
            if st.button("Показать анализ обучающего набора"):
                clean, quality = clean_network_data(load_network_csv(config.NETWORK_DATA_PATH))
                st.session_state["net_analyzed"] = analyze_network(clean, iso_model, rf_model)
                st.session_state["net_quality"] = quality
                st.rerun()

        if analyzed is not None:
            q = st.session_state.get("net_quality", {})
            a, b, c, d = st.columns(4)
            a.metric("Строк до очистки", q.get("rows", "—"))
            b.metric("Удалено дубликатов", q.get("duplicates_removed", "—"))
            c.metric("Пропусков", q.get("missing_total", "—"))
            d.metric("Значений inf", q.get("infinite_values", "—"))

            g1, g2 = st.columns(2)
            if config.NETWORK_LABEL_COL in analyzed.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                vc = analyzed[config.NETWORK_LABEL_COL].value_counts()
                sns.barplot(x=vc.values, y=vc.index, ax=ax, color="#55A868")
                ax.set_title("Метки в данных (фактические)"); ax.set_xlabel("Записей")
                fig.tight_layout(); g1.pyplot(fig); plt.close(fig)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(data=analyzed, x="if_score", hue="if_anomaly", bins=40, ax=ax,
                         palette={False: "#4C72B0", True: "#C44E52"})
            ax.set_title("Оценка аномальности (Isolation Forest)"); ax.set_xlabel("if_score")
            fig.tight_layout(); g2.pyplot(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 3.2))
            sns.scatterplot(data=analyzed.sample(min(3000, len(analyzed)), random_state=1),
                            x="Flow Duration", y="Flow Packets/s", hue="if_anomaly", s=12, ax=ax,
                            palette={False: "#4C72B0", True: "#C44E52"})
            ax.set_xscale("symlog"); ax.set_yscale("symlog")
            ax.set_title("Потоки: длительность vs пакетов/с (красные — аномалии по прогнозу)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("##### Записи, помеченные как аномальные")
            cols = (["Timestamp"] if "Timestamp" in analyzed.columns else []) + \
                ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Packets/s",
                 "if_score"] + [c for c in ["rf_prediction", "rf_confidence",
                                            config.NETWORK_LABEL_COL] if c in analyzed.columns]
            anomalies = analyzed[analyzed["if_anomaly"]].sort_values("if_score", ascending=False)
            st.dataframe(anomalies[cols].head(300), width="stretch", hide_index=True)
            st.download_button("⬇️ Полный результат анализа (CSV)", to_csv_bytes(analyzed),
                               "network_analysis.csv", "text/csv")


# ==========================================================================
# HTTP-запросы
# ==========================================================================
with tab_http:
    from src.nlp.models import classify_requests, explain_request, http_events
    from src.nlp.preprocess import load_text_csv, normalize_request

    st.subheader("Классификация HTTP-запросов: Normal / SQL Injection / XSS")
    if nlp_model is None:
        st.warning("NLP-модель не найдена. Подготовьте датасет (`scripts/prepare_text_dataset.py`) "
                   "и выполните `python train_all.py --only nlp`.")
    else:
        text = st.text_area("Проверить один запрос (строка запроса, параметр или тело формы)",
                            height=90, max_chars=config.MAX_REQUEST_LENGTH)
        if st.button("Классифицировать") and text.strip():
            res = classify_requests(nlp_model, [text]).iloc[0]
            c1, c2 = st.columns([1, 2])
            c1.metric("Прогноз модели", res["prediction"], f"уверенность {res['confidence']:.2f}")
            if res["needs_review"]:
                c1.warning("Низкая уверенность — нужна ручная проверка.")
            probs = {c: res[f"prob_{c}"] for c in nlp_model.classes_}
            c2.bar_chart(pd.Series(probs, name="вероятность"))
            st.markdown("Нормализованный запрос (после декодирования):")
            st.code(normalize_request(text), language=None)
            st.markdown("Наиболее повлиявшие символьные n-граммы:")
            st.dataframe(explain_request(nlp_model, text), hide_index=True)

        st.divider()
        up = st.file_uploader("Пакетная проверка: CSV со столбцом `request` "
                              "(необязательно `label`, `timestamp`)", type=["csv"], key="http_upload")
        if up is not None and st.button("Проанализировать запросы"):
            try:
                df = load_text_csv(up, filename=up.name)
                res = classify_requests(nlp_model, df[config.TEXT_COL])
                set_events("http", http_events(df, res))
                st.session_state["http_batch"] = pd.concat(
                    [res, df.drop(columns=[config.TEXT_COL])], axis=1)
                st.success(f"Классифицировано {len(res)} запросов. Результаты добавлены на «Обзор».")
            except (DataValidationError, ValueError) as exc:
                st.error(f"Файл отклонён: {exc}")

        batch = st.session_state.get("http_batch")
        if batch is not None:
            st.dataframe(batch.head(500), width="stretch", hide_index=True)
            st.download_button("⬇️ Результаты классификации (CSV)", to_csv_bytes(batch),
                               "http_classification.csv", "text/csv")


# ==========================================================================
# Качество моделей
# ==========================================================================
with tab_eval:
    st.subheader("Метрики на тестовой выборке")
    if not config.METRICS_PATH.exists():
        st.info("Метрик нет — выполните `python train_all.py`.")
    else:
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        rows = []
        for module, models in [("network", ["isolation_forest", "random_forest"]),
                               ("nlp", ["sgd_classifier", "complement_naive_bayes"])]:
            for m in models:
                mm = metrics.get(module, {}).get(m)
                if mm:
                    rows.append({"Модуль": module, "Модель": m, "Accuracy": mm["accuracy"],
                                 "Precision (macro)": mm["precision_macro"],
                                 "Recall (macro)": mm["recall_macro"], "F1 (macro)": mm["f1_macro"],
                                 "FPR": mm.get("false_positive_rate"), "Test size": mm["support"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("FPR — доля нормальных объектов, ошибочно помеченных как атака/аномалия. "
                   "При дисбалансе классов ориентируйтесь на Recall/F1 по классам атак, а не на Accuracy.")

        imgs = sorted(config.REPORTS_DIR.glob("cm_*.png"))
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for col, img in zip(cols, imgs[i:i + 2]):
                col.image(str(img), caption=img.stem)

        iso = metrics.get("network", {}).get("isolation_forest")
        if iso:
            st.markdown("##### Isolation Forest: доля помеченных записей по фактической метке")
            st.bar_chart(pd.Series(iso["detection_rate_by_label"], name="доля помеченных"))
            st.caption("Для BENIGN это доля ложных срабатываний. Типы атак, похожие на норму "
                       "по статистике потока, детектор аномалий пропускает — аномальность "
                       "и вредоносность не одно и то же.")
        with st.expander("Полный metrics.json"):
            st.json(metrics)


# ==========================================================================
# О системе
# ==========================================================================
with tab_about:
    st.markdown(f"""
**Что делает система.** Анализирует подготовленные наборы данных: сетевые потоки
(Isolation Forest + Random Forest) и тексты HTTP-запросов (TF-IDF + SGD Classifier)
и формирует предупреждения. Система ничего не блокирует и не атакует.

**Ограничения.**
* Прогноз модели — не подтверждённый инцидент. Возможны ложные срабатывания и пропуски атак.
* Модели обучены на учебных данных; на реальном трафике качество будет ниже.
* Новые техники обфускации и типы атак, которых не было в обучении, могут не распознаваться.
* Порог ручной проверки NLP-модели: уверенность < {config.NLP_REVIEW_THRESHOLD}.

**Безопасность.** Принимаются только CSV до {config.MAX_UPLOAD_MB} МБ и
{config.MAX_UPLOAD_ROWS:,} строк; содержимое не исполняется; при выгрузке ячейки,
начинающиеся с `= + - @`, экранируются.
""")
