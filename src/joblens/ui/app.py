"""Simple Streamlit frontend backed by the REST API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("JOBLENS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="JobLens ML", page_icon="🔎", layout="wide")
st.title("JobLens ML")
st.caption("Анализ IT-вакансий и соответствия резюме")

page = st.sidebar.radio("Раздел", ["Анализ", "О модели"])
if page == "О модели":
    st.header("О модели")
    st.markdown("""
    **Данные:** CSV с русскоязычными IT-вакансиями. Синтетический пример — только smoke-тест.

    **Алгоритмы:** TF-IDF + Logistic Regression, Ridge с текстовыми и табличными признаками.  

    **Метрики:** macro F1 для ролей; MAE, MdAPE и R² для зарплаты.

    **Ограничения:** слабые метки, смещение данных и эвристический match score.

    Результат не гарантирует трудоустройство и не является решением о найме.
    """)
else:
    left, right = st.columns(2)
    resume = left.text_area("Текст резюме", height=300)
    vacancy = right.text_area("Текст вакансии", height=300)
    c1, c2, c3, c4 = st.columns(4)
    area = c1.text_input("Город")
    experience = c2.text_input("Опыт")
    employment = c3.text_input("Занятость")
    schedule = c4.text_input("График")
    if st.button("Проанализировать", type="primary", use_container_width=True):
        if not resume.strip() or not vacancy.strip():
            st.error("Заполните текст резюме и вакансии.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/analyze-vacancy",
                    json={
                        "resume_text": resume,
                        "vacancy_text": vacancy,
                        "area": area,
                        "experience": experience,
                        "employment": employment,
                        "schedule": schedule,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                role = result.get("role") or {}
                m1, m2, m3 = st.columns(3)
                m1.metric("Направление", role.get("predicted_class", "модель недоступна"))
                m2.metric(
                    "Confidence",
                    f"{role.get('confidence', 0):.1%}"
                    if role.get("confidence") is not None
                    else "—",
                )
                m3.metric("Match score", f"{result['match']['score']:.1f}/100")
                st.write("Навыки вакансии:", result["match"]["vacancy_skills"] or "не найдены")
                st.write("Совпавшие:", result["match"]["matched_skills"] or "нет")
                st.write("Отсутствующие:", result["match"]["missing_skills"] or "нет")
                st.metric(
                    "Прогноз зарплаты",
                    f"{result['salary']:,.0f} ₽"
                    if result["salary_available"]
                    else "модель недоступна",
                )
                for warning in result["warnings"]:
                    st.warning(warning)
            except requests.RequestException as error:
                st.error(f"API недоступен: {error}")
