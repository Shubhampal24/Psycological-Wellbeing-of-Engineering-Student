import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# ==========================================
#  PAGE CONFIG & CSS
# ==========================================
st.set_page_config(page_title="Student Well-Being Analytics", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 800;
        color: #3b82f6;
        margin: 0;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .hero-title {
        font-weight: 900;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
#  HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Animations
lottie_dashboard = load_lottieurl("https://lottie.host/81b22e11-dbab-4318-8686-ef635f6dd3d6/sZ0zEebzVf.json")
lottie_test = load_lottieurl("https://lottie.host/9e4d0d0f-6fb6-4660-8457-3f309d4ebc85/lRkQ9mC9xM.json")
lottie_success = load_lottieurl("https://lottie.host/a82d0ad3-6df1-4a57-8b0d-587ff02dd8c9/Fq9WqM1h0n.json")
lottie_warning = load_lottieurl("https://lottie.host/4a2db7eb-9878-4389-9a25-c65507d8b5c9/P2v6Gf5WJp.json")
lottie_danger = load_lottieurl("https://lottie.host/1c8f4951-60a6-4dc4-b770-07ceea3db711/wD39vX8J0Q.json")


@st.cache_data(ttl=3600)
def fetch_live_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ykJtioZB3IrKzYKH5oab1Wf_WnCSQ6e7pLZT8nA1uCY/export?format=csv"
    try:
        raw_df = pd.read_csv(sheet_url)
    except:
        raw_df = pd.read_csv('data.csv') # Fallback if offline
    return raw_df

@st.cache_resource(ttl=3600)
def load_and_fit_encoders():
    # Keep the raw data for dashboard
    raw_df = fetch_live_data()
    
    dataset = raw_df.copy()
    new_columns = [f"q{i}" for i in range(len(dataset.columns))]
    new_columns[0] = "timestamp"
    new_columns[1] = "email"
    new_columns[47] = "stress_level"
    dataset.columns = new_columns
    dataset.drop(columns=["timestamp", "email"], inplace=True)
    dataset.dropna(subset=["stress_level"], inplace=True)

    for i in dataset.columns:
        if i != "stress_level":
            dataset[i] = dataset[i].fillna("Not Answered")

    frequency_map = {"Rarely or never": 0, "Occasionally": 1, "Frequently": 2, "Not Answered": -1}
    time_map = {"Less than 6 hours": 0, "6-7 hours": 1, "7-8 hours": 2, "More than 8 hours": 3, "Not Answered": -1}
    involvement_map = {"Not involved at all": 0, "Not very involved": 1, "Somewhat involved": 2, "Very involved": 3, "Not Answered": -1}
    yes_no_map = {"No": 0, "Yes": 1, "Not Answered": -1}

    col_maps = {}
    for col in dataset.columns:
        if col != "stress_level":
            unique_vals = set(dataset[col].dropna().unique())
            if unique_vals.issubset(set(frequency_map.keys())):
                dataset[col] = dataset[col].map(frequency_map)
                col_maps[col] = frequency_map
            elif unique_vals.issubset(set(time_map.keys())):
                dataset[col] = dataset[col].map(time_map)
                col_maps[col] = time_map
            elif unique_vals.issubset(set(involvement_map.keys())):
                dataset[col] = dataset[col].map(involvement_map)
                col_maps[col] = involvement_map
            elif unique_vals.issubset(set(yes_no_map.keys())):
                dataset[col] = dataset[col].map(yes_no_map)
                col_maps[col] = yes_no_map

    non_mapped_cols = [c for c in dataset.columns if not pd.api.types.is_numeric_dtype(dataset[c]) and c != "stress_level"]
    safety_encoders = {}
    for col in non_mapped_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(dataset[[col]])
        safety_encoders[col] = enc
        
    return raw_df, dataset, col_maps, non_mapped_cols, safety_encoders

raw_df, dataset, col_maps, non_mapped_cols, safety_encoders = load_and_fit_encoders()

@st.cache_resource
def load_models():
    hybrid_preprocessor = joblib.load('hybrid_preprocessor.pkl')
    tuned_model = joblib.load('tuned_stress_model.pkl')
    return hybrid_preprocessor, tuned_model

hybrid_preprocessor, tuned_model = load_models()

# ==========================================
#  DYNAMIC QUESTIONS SCHEMA (UI METADATA)
# ==========================================
import json
import os

questions = {}
if os.path.exists('questions_schema.json'):
    with open('questions_schema.json', 'r') as f:
        questions = json.load(f)
else:
    st.warning("⚠️ Training pipeline has not generated the UI schema yet. Please run `python train_pipeline.py`.")


# ==========================================
#  SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3362/3362002.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to:", ["📊 Dashboard", "🧠 Test Yourself", "🤝 Contribute"])
    st.markdown("---")
    st.caption("Machine Learning powered Student Well-Being Analyzer.")

# ==========================================
#  PAGE 1: DASHBOARD
# ==========================================
if page == "📊 Dashboard":
    st.markdown('<p class="hero-title">Student Well-Being Dashboard</p>', unsafe_allow_html=True)
    st.write("A global overview of stress levels and demographics from the collected dataset.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_students = len(dataset)
    frequent_stress = len(dataset[dataset['stress_level'] == 'Frequently'])
    occasional_stress = len(dataset[dataset['stress_level'] == 'Occasionally'])
    rare_stress = len(dataset[dataset['stress_level'] == 'Rarely or never'])
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Students</div>
                <div class="metric-value">{total_students}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Stress</div>
                <div class="metric-value" style="color:#ef4444;">{frequent_stress}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Moderate Stress</div>
                <div class="metric-value" style="color:#f59e0b;">{occasional_stress}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Low Stress</div>
                <div class="metric-value" style="color:#10b981;">{rare_stress}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Stress Distribution")
        stress_counts = dataset['stress_level'].value_counts().reset_index()
        stress_counts.columns = ['Stress Level', 'Count']
        fig_donut = px.pie(stress_counts, values='Count', names='Stress Level', hole=0.6,
                           color='Stress Level', 
                           color_discrete_map={'Frequently':'#ef4444', 'Occasionally':'#f59e0b', 'Rarely or never':'#10b981'})
        fig_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with chart_col2:
        st.subheader("Stress by Background Area")
        # dataset['q2'] is Area Type
        area_stress = pd.crosstab(dataset['q2'], dataset['stress_level']).reset_index()
        # Melt for Plotly
        area_stress_melted = area_stress.melt(id_vars='q2', var_name='Stress Level', value_name='Count')
        fig_bar = px.bar(area_stress_melted, x='q2', y='Count', color='Stress Level', barmode='group',
                         labels={'q2': 'Background Area'},
                         color_discrete_map={'Frequently':'#ef4444', 'Occasionally':'#f59e0b', 'Rarely or never':'#10b981'})
        fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # AI BRAIN FEATURE IMPORTANCE GRAPH
    # ==========================================
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("🧠 What the AI Cares About")
    st.write("Our Shadow Model dynamically learns which questions are the strongest predictors of student stress. Here are the top factors it is currently tracking:")
    
    if questions:
        imp_data = []
        for q_id, q_data in questions.items():
            imp_data.append({"Question": q_data["text"], "Importance": q_data.get("importance", 0.0)})
        
        df_imp = pd.DataFrame(imp_data)
        # Filter out 0 importance if any, and sort
        df_imp = df_imp[df_imp["Importance"] > 0].sort_values(by="Importance", ascending=True)
        
        if not df_imp.empty:
            fig_imp = px.bar(df_imp, x="Importance", y="Question", orientation='h', 
                             color="Importance", color_continuous_scale="Purples")
            fig_imp.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance data is currently being calculated by the ML pipeline.")

# ==========================================
#  PAGE 2: TEST YOURSELF
# ==========================================
elif page == "🧠 Test Yourself":
    st.markdown('<p class="hero-title">Predict Your Stress Level</p>', unsafe_allow_html=True)
    st.write("Our machine learning model only needs 15 specific questions to evaluate your well-being. Fill out the form below!")
    
    st.markdown("---")
    
    user_answers = {}
    
    # Two columns layout for the form
    form_col1, form_col2 = st.columns(2)
    
    keys = list(questions.keys())
    for i, q_id in enumerate(keys):
        q_info = questions[q_id]
        if i % 2 == 0:
            with form_col1:
                user_answers[q_id] = st.selectbox(q_info["text"], options=q_info["options"], index=q_info["options"].index('Not Answered'))
        else:
            with form_col2:
                user_answers[q_id] = st.selectbox(q_info["text"], options=q_info["options"], index=q_info["options"].index('Not Answered'))

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze My Responses", type="primary", use_container_width=True):
        with st.spinner("Processing data through the neural network..."):
            df_user = pd.DataFrame([user_answers])
            for col in df_user.columns:
                if col in col_maps:
                    df_user[col] = df_user[col].map(col_maps[col])
            user_non_mapped = [c for c in df_user.columns if c in non_mapped_cols]
            if user_non_mapped:
                for col in user_non_mapped:
                    df_user[col] = safety_encoders[col].transform(df_user[[col]])[:, 0]
                    df_user[col] = df_user[col].astype(float)
                    
            X_processed = hybrid_preprocessor.transform(df_user)
            prediction = tuned_model.predict(X_processed)[0]
            
            if isinstance(prediction, str):
                result_text = prediction
            else:
                reverse_target_map = {0: "Rarely or never", 1: "Occasionally", 2: "Frequently"}
                result_text = reverse_target_map.get(prediction, "Unknown")
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if result_text == "Frequently":
                if lottie_danger: st_lottie(lottie_danger, height=200, key="danger")
            elif result_text == "Occasionally":
                if lottie_warning: st_lottie(lottie_warning, height=200, key="warning")
            else:
                if lottie_success: st_lottie(lottie_success, height=200, key="success")
                
        with res_col2:
            st.markdown(f"### Predicted Stress Level: **{result_text}**")
            if result_text == "Frequently":
                st.error("🚨 It looks like you might be experiencing a high level of stress. Please remember to take breaks, prioritize self-care, and consider reaching out to college support resources.")
            elif result_text == "Occasionally":
                st.warning("⚠️ You experience moderate stress. Try to balance your coursework with activities you enjoy to keep your stress manageable.")
            else:
                st.success("✅ You are managing your college experience well with minimal stress. Keep up the good work!")

# ==========================================
#  PAGE 3: CONTRIBUTE
# ==========================================
elif page == "🤝 Contribute":
    st.markdown('<p class="hero-title">Help Us Improve</p>', unsafe_allow_html=True)
    st.write("We are constantly gathering data to improve the accuracy of our machine learning model. If you are an engineering student, your anonymous input can help us map out the mental well-being of the student body.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.info("💡 **Why Contribute?** More data helps our model discover new patterns and provide more accurate predictions for everyone.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom stylized CTA button link
    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://docs.google.com/forms/d/e/1FAIpQLSfLy2U6OaVLeCeTThB0frMqVYWW5GrFLb_FUtws66PetalSIw/viewform" target="_blank" style="text-decoration: none;">
                <button style="background-color: #3b82f6; color: white; padding: 20px 40px; border: none; border-radius: 10px; font-size: 24px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); transition: transform 0.2s;">
                    📝 Take the Official Survey
                </button>
            </a>
            <p style="margin-top: 15px; color: #64748b;">Takes ~5 minutes. Completely anonymous.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Adding a subtle animation below
    if lottie_dashboard:
        st_lottie(lottie_dashboard, height=400, key="contrib_lottie")
