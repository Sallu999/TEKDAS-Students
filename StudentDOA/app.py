"""Streamlit teaching dashboard: Student Academic Retention -> BI -> ML -> LLM.

Run from this folder:
    streamlit run app.py
"""
from pathlib import Path
import json
import os
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
import streamlit as st

from data_prep import (
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    COURSE_MAP,
    DATA_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET,
    clean_and_prepare_features,
    load_source_data,
)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ==========================================
# LLM ANALYST FUNCTIONS (INTEGRATED)
# ==========================================
def build_prompt(
    student_profile: Dict[str, Any],
    prediction_probabilities: Dict[str, float],
    model_factors: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Membangun grounded prompt yang dapat diinspeksi pengguna sebelum memanggil LLM."""
    profile_summary = {
        "student_id": student_profile.get("student_id", "MHS-DEMO"),
        "program_studi": student_profile.get("course_name", "Informatics Engineering"),
        "umur_masuk": student_profile.get("Age at enrollment", 20),
        "status_spp": "Lunas (Up to date)" if student_profile.get("Tuition fees up to date", 1) == 1 else "Menunggak",
        "penerima_beasiswa": "Ya" if student_profile.get("Scholarship holder", 0) == 1 else "Tidak",
        "sks_sem1_diambil": student_profile.get("Curricular units 1st sem (enrolled)", 6),
        "sks_sem1_lulus": student_profile.get("Curricular units 1st sem (approved)", 5),
        "nilai_sem1": round(float(student_profile.get("Curricular units 1st sem (grade)", 13.5)), 1),
        "sks_sem2_diambil": student_profile.get("Curricular units 2nd sem (enrolled)", 6),
        "sks_sem2_lulus": student_profile.get("Curricular units 2nd sem (approved)", 2),
        "nilai_sem2": round(float(student_profile.get("Curricular units 2nd sem (grade)", 10.2)), 1),
    }

    return f"""
Anda adalah Konselor dan Analis Retensi Akademik Perguruan Tinggi.
Gunakan HANYA bukti dan fakta yang diberikan di bawah ini. Skor machine learning adalah probabilitas statistik, bukan vonis mutlak atau hubungan sebab-akibat langsung.
Jangan mengarang fakta pribadi, keluarga, atau alasan di luar data yang diberikan.

PROFIL MAHASISWA
{json.dumps(profile_summary, indent=2, ensure_ascii=False)}

SKOR PREDIKSI MACHINE LEARNING
- Probabilitas Dropout: {prediction_probabilities.get('Dropout', 0.0):.1%}
- Probabilitas Enrolled: {prediction_probabilities.get('Enrolled', 0.0):.1%}
- Probabilitas Graduate: {prediction_probabilities.get('Graduate', 0.0):.1%}

FAKTOR MODEL GLOBAL UTAMA
{json.dumps(model_factors or [], indent=2, ensure_ascii=False)}

TUGAS ANDA
Tulis dalam Bahasa Indonesia terstruktur untuk Dosen Pembimbing Akademik / Dekanat:
1. Satu kalimat interpretasi risiko dropout mahasiswa berdasarkan skor ML.
2. Dua observasi berbasis bukti konkret dari riwayat akademik atau status biaya mahasiswa di atas.
3. Dua rekomendasi intervensi akademik/finansial yang proporsional dan realistis untuk membantu mahasiswa ini menyelesaikan studinya.
4. Satu catatan kehati-hatian etis mengenai keterbatasan model prediktif (karena korelasi bukan sebab-akibat).
Gunakan format poin (bullet points) yang ringkas, jelas, dan profesional.
""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Mengirim prompt ke Ollama lokal."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def generate_student_strategy(
    student_profile: Dict[str, Any],
    prediction_probabilities: Dict[str, float],
    model_factors: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Menghubungi Ollama untuk menghasilkan rencana aksi atau fallback terarah."""
    prompt = build_prompt(
        student_profile=student_profile,
        prediction_probabilities=prediction_probabilities,
        model_factors=model_factors,
    )
    try:
        return call_ollama(prompt=prompt, model=model)
    except Exception as exc:
        p_drop = prediction_probabilities.get("Dropout", 0.0)
        spp_status = "menunggak" if student_profile.get("Tuition fees up to date") == 0 else "lunas"
        sks2_app = student_profile.get("Curricular units 2nd sem (approved)", 0)
        sks2_enr = student_profile.get("Curricular units 2nd sem (enrolled)", 0)

        return (
            f"*(Catatan: Ollama `{model}` tidak terhubung: {exc}. Menampilkan sintesis analisis berbasis bukti:)*\n\n"
            f"### Analisis Retensi & Rencana Aksi Akademik ({student_profile.get('student_id')} - {student_profile.get('course_name')})\n\n"
            f"1. **Interpretasi Risiko ML**:\n"
            f"   - Mahasiswa ini memiliki probabilitas dropout sebesar **{p_drop:.1%}**, yang menunjukkan tingkat risiko "
            f"   {'tinggi' if p_drop > 0.4 else 'moderat/rendah'} terhadap kelangsungan studinya.\n\n"
            f"2. **Observasi Berbasis Bukti**:\n"
            f"   - Pada semester 2, mahasiswa hanya berhasil meluluskan **{sks2_app} dari {sks2_enr} SKS** yang diambil, dengan rata-rata nilai {student_profile.get('Curricular units 2nd sem (grade)', 0):.1f}.\n"
            f"   - Status pembayaran SPP tercatat **{spp_status}**, yang menjadi salah satu faktor risiko finansial yang relevan.\n\n"
            f"3. **Rekomendasi Intervensi Proporsional**:\n"
            f"   - **Konseling Akademik & Remediasi**: Jadwalkan pertemuan dengan dosen pembimbing untuk menyusun rencana perbaikan mata kuliah semester 1 & 2 yang belum lulus.\n"
            f"   - **Dukungan Finansial**: Hubungi bagian kemahasiswaan untuk skema restrukturisasi cicilan SPP atau informasi bantuan beasiswa agar tidak terkendala administrasi.\n\n"
            f"4. **Catatan Etis & Keterbatasan Model**:\n"
            f"   - Model ini mendeteksi pola asosiasi retrospektif dari data akademik lampau, bukan vonis mutlak. Keputusan pembinaan harus didasarkan pada dialog langsung dengan mahasiswa."
        )


# ==========================================
# STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Student Academic Retention: BI + ML + LLM", layout="wide")
st.title("Student Academic Retention — dari Data ke BI, ML, dan LLM")
st.caption(
    "Proyek pembelajaran: analisis profil akademik mahasiswa dari dataset_StudentDOA.xlsx, "
    "prediksi risiko dropout dengan ML, dan gunakan LLM untuk menyusun rencana aksi intervensi."
)


@st.cache_data
def load_data():
    """Memuat data model yang telah disiapkan atau membersihkan langsung dari dataset Excel."""
    ml_dataset_path = BASE_DIR / "student_ml_dataset.csv"
    if ml_dataset_path.exists():
        return pd.read_csv(ml_dataset_path)

    raw_df = load_source_data()
    return clean_and_prepare_features(raw_df)


@st.cache_resource
def load_model():
    """Memuat pipeline model yang sudah dilatih oleh ml_model.py."""
    model_path = BASE_DIR / "student_dropout_pipeline.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data
def load_optional_outputs():
    """Memuat metrik evaluasi dan feature importance yang dihasilkan ml_model.py."""
    fi_path = BASE_DIR / "feature_importance.csv"
    metrics_path = BASE_DIR / "model_metrics.json"

    fi = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return fi, metrics


# Muat data dan artefak
try:
    data = load_data()
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

model = load_model()
feature_importance, model_metrics = load_optional_outputs()

# Headline KPI Metrics di atas dashboard
dropout_rate = float(data[TARGET].eq("Dropout").mean())
graduate_rate = float(data[TARGET].eq("Graduate").mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Mahasiswa", f"{len(data):,}")
col2.metric("Jumlah Fitur", f"{len(FEATURE_COLUMNS)}")
col3.metric("Observed Dropout", f"{dropout_rate:.1%}")
col4.metric("Observed Graduate", f"{graduate_rate:.1%}")

# Sidebar navigasi mahasiswa dan pengaturan threshold
st.sidebar.header("Pilih Mahasiswa")
selected_id = st.sidebar.selectbox("Student ID", data[ID_COLUMN].tolist())
decision_threshold = st.sidebar.slider(
    "Threshold Peringatan Risiko Dropout",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Jika probabilitas dropout melebihi nilai threshold ini, sistem memicu peringatan intervensi dini."
)

student_row = data.loc[data[ID_COLUMN].eq(selected_id)].iloc[0]
X_student = data.loc[data[ID_COLUMN].eq(selected_id)][FEATURE_COLUMNS]

# Ringkasan cepat profil mahasiswa di sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Profil Ringkas")
st.sidebar.write(f"**Program Studi:** {student_row.get('course_name', 'N/A')}")
st.sidebar.write(f"**Status Aktual:** `{student_row[TARGET]}`")
st.sidebar.write(f"**Umur saat Masuk:** {student_row['Age at enrollment']} tahun")
st.sidebar.write(f"**Status SPP:** {'Lunas' if student_row['Tuition fees up to date'] == 1 else 'Menunggak'}")
st.sidebar.write(f"**Beasiswa:** {'Ya' if student_row['Scholarship holder'] == 1 else 'Tidak'}")
st.sidebar.write(
    f"**SKS Lulus Sem 1 & 2:** "
    f"{student_row['Curricular units 1st sem (approved)']}/{student_row['Curricular units 1st sem (enrolled)']} | "
    f"{student_row['Curricular units 2nd sem (approved)']}/{student_row['Curricular units 2nd sem (enrolled)']}"
)

# 4 Tab Lapisan: Data -> BI -> ML -> LLM
tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · Data Mahasiswa",
    "2 · Academic Intelligence",
    "3 · Machine Learning",
    "4 · LLM Academic Analyst",
])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab_data:
    st.subheader("Inspeksi Dataset & Struktur Fitur")
    st.markdown(
        "- **dataset_StudentDOA.xlsx** — 4.424 baris data mahasiswa (UCI Student Dropout & Academic Success).\n"
        "- **Fitur Demografis & Finansial**: Umur, status pendaftaran, pekerjaan orang tua, SPP, dan beasiswa.\n"
        "- **Fitur Akademik**: SKS terdaftar, evaluasi, SKS lulus, dan nilai semester 1 & 2.\n"
        "- **Target ML**: Kolom `Target` dengan 3 status: **Dropout**, **Enrolled**, dan **Graduate**."
    )

    st.write(f"Dimensi Data: **{data.shape[0]:,} baris × {data.shape[1]} kolom**")
    default_cols = [ID_COLUMN, "course_name", "Age at enrollment", "Tuition fees up to date",
                    "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)", TARGET]
    selected_cols = st.multiselect(
        "Pilih kolom untuk ditampilkan",
        options=list(data.columns),
        default=[c for c in default_cols if c in data.columns],
    )
    st.dataframe(data[selected_cols].head(100), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribusi Status Target**")
        target_counts = data[TARGET].value_counts()
        target_summary = pd.DataFrame({
            "Jumlah": target_counts,
            "Persentase": (data[TARGET].value_counts(normalize=True) * 100).round(1).astype(str) + "%",
        })
        st.dataframe(target_summary, use_container_width=True)

    with c2:
        st.warning(
            "**Data-Quality Lesson**:\n"
            "1. File Excel asli `dataset_StudentDOA.xlsx` kehilangan pemisah desimal saat ekspor (misal nilai 140 untuk 14.0). "
            "Pipeline `data_prep.py` melakukan normalisasi otomatis agar skala penilaian kembali ke 0-20 dan makroekonomi ke persentase riil.\n"
            "2. Fitur semester 1 & 2 bersifat retrospektif (diketahui setelah perkuliahan berjalan), sehingga "
            "tidak boleh disamakan dengan prediksi pada saat pendaftaran awal mahasiswa baru."
        )

# ==========================================
# TAB 2: ACADEMIC INTELLIGENCE (BI)
# ==========================================
with tab_bi:
    st.subheader("Descriptive Academic BI sebelum Predictive ML")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**Proporsi Dropout Berdasarkan Status Pembayaran SPP**")
        fee_dropout = (
            data.groupby("Tuition fees up to date")[TARGET]
            .apply(lambda s: (s == "Dropout").mean())
            .rename("Rasio Dropout")
            .to_frame()
        )
        fee_dropout.index = ["Menunggak SPP", "SPP Lunas"]
        st.bar_chart(fee_dropout)
        st.caption("Menunggak SPP memiliki korelasi kuat dengan tingginya rasio dropout.")

    with b_col2:
        st.markdown("**Rata-rata SKS Lulus Semester 2 per Status Akhir**")
        approved_target = (
            data.groupby(TARGET)["Curricular units 2nd sem (approved)"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(approved_target)
        st.caption("Mahasiswa yang dropout rata-rata hanya meluluskan kurang dari 2 SKS pada semester 2.")

    if "course_name" in data.columns:
        st.markdown("**Distribusi Status Kelulusan per Program Studi (Top 10)**")
        top_courses = data["course_name"].value_counts().head(10).index
        course_data = data[data["course_name"].isin(top_courses)]
        course_cross = pd.crosstab(course_data["course_name"], course_data[TARGET], normalize="index")
        st.bar_chart(course_cross)

    st.markdown(f"**Profil Akademik {selected_id} vs Rata-Rata Angkatan**")
    comparison_df = pd.DataFrame({
        "Metrik": [
            "Umur saat Pendaftaran",
            "SKS Lulus Sem 1",
            "Nilai Rata-rata Sem 1",
            "SKS Lulus Sem 2",
            "Nilai Rata-rata Sem 2",
        ],
        f"{selected_id} (Terpilih)": [
            f"{student_row['Age at enrollment']:.1f}",
            f"{student_row['Curricular units 1st sem (approved)']:.1f}",
            f"{student_row['Curricular units 1st sem (grade)']:.1f}",
            f"{student_row['Curricular units 2nd sem (approved)']:.1f}",
            f"{student_row['Curricular units 2nd sem (grade)']:.1f}",
        ],
        "Rata-rata Angkatan": [
            f"{data['Age at enrollment'].mean():.1f}",
            f"{data['Curricular units 1st sem (approved)'].mean():.1f}",
            f"{data['Curricular units 1st sem (grade)'].mean():.1f}",
            f"{data['Curricular units 2nd sem (approved)'].mean():.1f}",
            f"{data['Curricular units 2nd sem (grade)'].mean():.1f}",
        ],
    })
    st.dataframe(comparison_df.set_index("Metrik"), use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING
# ==========================================
with tab_ml:
    st.subheader("Prediksi Status Akademik & Evaluasi Model")

    if model is None:
        st.error(
            "Model belum dibuat atau artefak belum lengkap. "
            "Jalankan langkah berikut di terminal:\n"
            "```powershell\n"
            "python data_prep.py\n"
            "python ml_model.py\n"
            "```"
        )
        if st.button("Latih Model Sekarang di Dashboard"):
            with st.spinner("Menjalankan data_prep dan ml_model..."):
                from ml_model import main as run_train
                run_train()
                st.success("Model berhasil dilatih! Silakan muat ulang halaman.")
                st.rerun()
    else:
        classes = list(model.named_steps["model"].classes_)
        probs = model.predict_proba(X_student)[0]
        prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
        dropout_prob = prob_dict.get("Dropout", 0.0)
        predicted_label = model.predict(X_student)[0]

        is_dropout_alert = dropout_prob >= decision_threshold
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(
            "Probabilitas Dropout",
            f"{dropout_prob:.1%}",
            delta=f"{'+' if is_dropout_alert else ''}{(dropout_prob - decision_threshold):.1%} vs Threshold",
            delta_color="inverse",
        )
        m_col2.metric("Prediksi Model", str(predicted_label), help=f"Threshold peringatan dropout = {decision_threshold:.2f}")
        m_col3.metric("Status Aktual (Dataset)", str(student_row[TARGET]))

        st.markdown("**Distribusi Probabilitas Prediksi Tiap Status**")
        st.bar_chart(pd.Series(prob_dict))

        st.info(
            f"**Decision Threshold Risiko Dropout saat ini: {decision_threshold:.2f}**. "
            "Karena target bersifat multi-class dan distribusinya tidak seimbang, evaluasi kinerja model harus "
            "memperhatikan Macro F1 dan Weighted F1, bukan sekadar Accuracy."
        )

        if model_metrics:
            met1, met2, met3 = st.columns(3)
            met1.metric("Accuracy Global", f"{model_metrics.get('accuracy', 0):.3f}")
            met2.metric("Macro F1", f"{model_metrics.get('macro_f1', 0):.3f}")
            met3.metric("Weighted F1", f"{model_metrics.get('weighted_f1', 0):.3f}")

        if not feature_importance.empty:
            st.markdown("**15 Fitur Paling Berpengaruh (Feature Importance Global)**")
            top_fi = feature_importance.head(15).set_index("feature")
            st.bar_chart(top_fi["importance"])

# ==========================================
# TAB 4: LLM ACADEMIC ANALYST
# ==========================================
with tab_llm:
    st.subheader("LLM sebagai Konselor & Analyst Akademik — Berbasis Bukti (Grounded)")
    st.caption(f"Ollama model: {DEFAULT_MODEL} (atur via environment variable OLLAMA_MODEL)")

    if model is None:
        st.error("Train model terlebih dahulu (`python ml_model.py`) agar LLM menerima skor ML yang nyata.")
    else:
        top_factors = feature_importance.head(8).to_dict("records") if not feature_importance.empty else []
        prompt = build_prompt(
            student_profile=student_row.to_dict(),
            prediction_probabilities=prob_dict,
            model_factors=top_factors,
        )

        with st.expander("Lihat prompt terstruktur yang dikirim ke LLM"):
            st.code(prompt, language="text")

        if st.button("Generate Rekomendasi Rencana Aksi Akademik"):
            with st.spinner("LLM sedang menganalisis bukti akademik dan probabilitas ML..."):
                strategy = generate_student_strategy(
                    student_profile=student_row.to_dict(),
                    prediction_probabilities=prob_dict,
                    model_factors=top_factors,
                    model=DEFAULT_MODEL,
                )
            st.markdown(strategy)

st.markdown("---")
st.caption(
    "Teaching principle: dataset_StudentDOA.xlsx menyediakan data; BI mendeskripsikan pola; "
    "ML memprediksi risiko; LLM mengomunikasikan bukti dan menyusun rekomendasi intervensi terarah."
)
