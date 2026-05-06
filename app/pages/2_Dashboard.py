"""
pages/2_Dashboard.py
Displays: Health Score + Category + Confidence
          SHAP bar chart + Expense pie chart
          Gemini AI personalised advice
Reads from: st.session_state (set by 1_Input_Form.py)
"""

import streamlit as st
import sys
import os

#path fix
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#page config
st.set_page_config(
    page_title="Dashboard — Finance Health Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#global CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');
    :root {
        --primary:       #1B4F8A;
        --primary-light: #2E6FBF;
        --primary-dark:  #0D2E52;
        --accent:        #00B4D8;
        --success:       #2ECC71;
        --warning:       #F39C12;
        --danger:        #E74C3C;
        --bg:            #F0F4F8;
        --card:          #FFFFFF;
        --text:          #1A202C;
        --text-light:    #4A5568;
        --border:        #CBD5E0;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-dark) 0%, var(--primary) 100%);
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    .main-header {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-light) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(27,79,138,0.3);
    }
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.92rem;
        margin: 0.4rem 0 0 0;
    }
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--primary);
        border-left: 4px solid var(--accent);
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .finance-card {
        background: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .score-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .advice-card {
        background: var(--card);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--primary-light);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .advice-card h4 {
        color: var(--primary);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .advice-card ul {
        margin: 0;
        padding-left: 1.2rem;
        color: var(--text-light);
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .disclaimer {
        background: #EBF4FF;
        border: 1px solid #BEE3F8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: var(--text-light);
        margin-top: 1rem;
    }
    [data-testid="stMetric"] {
        background: var(--card);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"]  { color: var(--text-light) !important; font-size: 0.85rem !important; }
    [data-testid="stMetricValue"]  { color: var(--primary) !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: var(--card);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--border);
        color: var(--text-light);
        font-weight: 500;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
        border-color: var(--primary) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600;
        box-shadow: 0 2px 8px rgba(27,79,138,0.3);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

#sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
            <div style='font-size:3rem;'>💰</div>
            <div style='font-family:Poppins,sans-serif; font-size:1.1rem;
                        font-weight:700; color:white; margin-top:0.5rem;'>
                Finance Analyzer
            </div>
            <div style='font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:0.25rem;'>
                ML + GenAI Powered
            </div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.15); margin:1rem 0;'/>
    """, unsafe_allow_html=True)
    st.page_link("main.py",               label="🏠  Home")
    st.page_link("pages/1_Input_Form.py",  label="📝  Analyze Finances")
    st.page_link("pages/2_Dashboard.py",   label="📊  Dashboard")
    st.page_link("pages/3_History.py",     label="📈  History")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:1rem 0;'/>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.5);
                    text-align:center; padding-bottom:1rem; line-height:1.6;'>
            Powered by<br/>
            <b style='color:rgba(255,255,255,0.8);'>XGBoost · SHAP · Gemini</b><br/>
            MySQL · Streamlit
        </div>
    """, unsafe_allow_html=True)

#header
st.markdown("""
    <div class='main-header'>
        <h1>📊 Your Financial Dashboard</h1>
        <p>ML score · SHAP explainability · Gemini AI advice</p>
    </div>
""", unsafe_allow_html=True)

#guard — no data yet
if "result" not in st.session_state:
    st.markdown("""
        <div style='text-align:center; padding:3rem 1rem;'>
            <div style='font-size:4rem;'>📋</div>
            <div style='font-family:Poppins,sans-serif; font-size:1.3rem;
                        font-weight:600; color:#1B4F8A; margin:1rem 0 0.5rem 0;'>
                No Analysis Yet
            </div>
            <div style='color:#4A5568; font-size:0.95rem;'>
                Go to <b>Analyze Finances</b> in the sidebar and submit your details first.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

#load data from session state
result     = st.session_state["result"]
advice     = st.session_state.get("advice", {})
input_data = st.session_state.get("input_data", {})
user_name  = st.session_state.get("user_name", "User")
month_year = st.session_state.get("month_year", "")

score      = result.get("score", 0)
category   = result.get("category", "Unknown")
confidence = result.get("confidence", 0)
probs      = result.get("probabilities", {})
narrative  = result.get("narrative", [])
hurting    = result.get("hurting_factors", [])
helping    = result.get("helping_factors", [])
shap_chart = result.get("shap_chart", None)
expense_pie= result.get("expense_pie", None)

#color mapping
color_map = {"Stable": "#2ECC71", "At Risk": "#F39C12", "Critical": "#E74C3C"}
emoji_map = {"Stable": "✅", "At Risk": "⚠️", "Critical": "🚨"}
color = color_map.get(category, "#1B4F8A")
emoji = emoji_map.get(category, "📊")

# SECTION 1 — Score + Metrics
st.markdown("<div class='section-header'>🎯 Financial Health Score</div>", unsafe_allow_html=True)

score_col, metrics_col = st.columns([1, 2])

with score_col:
    st.markdown(f"""
        <div class='score-card' style='background:linear-gradient(135deg,
             {color}15 0%, {color}30 100%); border: 2px solid {color};'>
            <div style='font-size:4rem; font-weight:700; color:{color};
                        font-family:Poppins,sans-serif; line-height:1;'>
                {score}
            </div>
            <div style='font-size:0.9rem; color:#4A5568; margin:0.25rem 0 0.75rem 0;'>
                out of 100
            </div>
            <div style='font-size:1.3rem; font-weight:600; color:{color};'>
                {emoji} {category}
            </div>
            <div style='font-size:0.82rem; color:#4A5568; margin-top:0.5rem;'>
                Confidence: {confidence:.0%}
            </div>
            <div style='margin-top:1rem; font-size:0.78rem; color:#4A5568;'>
                {user_name} · {month_year}
            </div>
        </div>
    """, unsafe_allow_html=True)

with metrics_col:
    # Class probabilities
    st.markdown("**Class Probabilities**")
    for cls, prob in probs.items():
        cls_color = color_map.get(cls, "#1B4F8A")
        st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:0.5rem;'>
                <div style='width:80px; font-size:0.85rem; color:#4A5568;'>{cls}</div>
                <div style='flex:1; background:#E2E8F0; border-radius:4px; height:20px; margin:0 0.75rem;'>
                    <div style='width:{prob*100:.1f}%; background:{cls_color};
                                height:100%; border-radius:4px; transition:width 0.5s;'></div>
                </div>
                <div style='width:45px; font-size:0.85rem; font-weight:600;
                            color:{cls_color}; text-align:right;'>{prob:.0%}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Key ratios
    income = input_data.get("monthly_income", 1)
    if income > 0:
        r1, r2, r3 = st.columns(3)
        with r1:
            savings_rate = input_data.get("savings", 0) / income * 100
            sr_color = "#2ECC71" if savings_rate >= 20 else "#F39C12" if savings_rate >= 10 else "#E74C3C"
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
        with r2:
            emi_ratio = input_data.get("emi", 0) / income * 100
            er_color = "#2ECC71" if emi_ratio <= 30 else "#F39C12" if emi_ratio <= 50 else "#E74C3C"
            st.metric("EMI Ratio", f"{emi_ratio:.1f}%")
        with r3:
            rent_ratio = input_data.get("rent", 0) / income * 100
            rr_color = "#2ECC71" if rent_ratio <= 30 else "#F39C12" if rent_ratio <= 40 else "#E74C3C"
            st.metric("Rent Ratio", f"{rent_ratio:.1f}%")

st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 2 — SHAP Charts
st.markdown("<div class='section-header'>🔍 SHAP Explainability — What's Driving Your Score</div>", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if shap_chart is not None:
        st.plotly_chart(shap_chart, use_container_width=True)
    else:
        st.info("SHAP chart not available.")

with chart_col2:
    if expense_pie is not None:
        st.plotly_chart(expense_pie, use_container_width=True)
    else:
        st.info("Expense chart not available.")

#SHAP narrative
if narrative:
    st.markdown("<div class='section-header'>📝 ML Model Explanation</div>", unsafe_allow_html=True)
    narr_col1, narr_col2 = st.columns(2)

    with narr_col1:
        st.markdown("**🔴 Hurting Your Score**")
        if hurting:
            for factor in hurting:
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    st.markdown(f"""
                        <div style='background:rgba(231,76,60,0.15); border-left:3px solid #E74C3C;
                                    padding:0.5rem 0.75rem; border-radius:4px; margin-bottom:0.4rem;
                                    font-size:0.88rem; color:#FFFFFF;'>
                            ❌ <b>{factor[0]}</b> — impact: {abs(float(factor[1])):.3f}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("*No major hurting factors identified*")

    with narr_col2:
        st.markdown("**✅ Helping Your Score**")
        if helping:
            for factor in helping:
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    st.markdown(f"""
                        <div style='background:rgba(46,204,113,0.15); border-left:3px solid #2ECC71;
                                    padding:0.5rem 0.75rem; border-radius:4px; margin-bottom:0.4rem;
                                    font-size:0.88rem; color:#FFFFFF;'>
                            ✅ <b>{factor[0]}</b> — impact: {abs(float(factor[1])):.3f}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("*No major helping factors identified*")

st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 3 — Gemini AI Advice
st.markdown("<div class='section-header'>✨ Gemini AI Financial Advice</div>", unsafe_allow_html=True)

#advice mode tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Personalised Advice",
    "📊 Monthly Summary",
    "📈 Goal Planning"
])

def render_advice(advice_dict: dict):
    """Render a structured advice dict as styled cards."""
    if not advice_dict:
        st.info("No advice available.")
        return

    # Summary banner
    summary = advice_dict.get("summary", "")
    if summary and "Invalid mode" not in summary and "No history" not in summary:
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1B4F8A15,#2E6FBF25);
                        border:1px solid #2E6FBF; border-radius:10px;
                        padding:1rem 1.25rem; margin-bottom:1rem;'>
                <div style='font-size:0.8rem; color:#1B4F8A; font-weight:600;
                            margin-bottom:0.25rem;'>KEY TAKEAWAY</div>
                <div style='color:#1A202C; font-size:0.95rem; font-weight:500;'>
                    💡 {summary}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Sections
    sections = advice_dict.get("sections", [])
    for section in sections:
        title  = section.get("title", "")
        points = section.get("points", [])
        if not points:
            continue

        # Pick border color based on section title emoji
        border = "#E74C3C" if "🔴" in title else \
                 "#2ECC71" if "✅" in title else \
                 "#F39C12" if "⚠️" in title else \
                 "#3498DB" if "📈" in title or "🗓️" in title else \
                 "#9B59B6" if "💡" in title else \
                 "#1B4F8A"

        points_html = "".join([f"<li>{p}</li>" for p in points])
        st.markdown(f"""
            <div class='advice-card' style='border-left-color:{border};'>
                <h4>{title}</h4>
                <ul>{points_html}</ul>
            </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    disclaimer = advice_dict.get("disclaimer", "")
    if disclaimer:
        st.markdown(f"<div class='disclaimer'>⚠️ {disclaimer}</div>", unsafe_allow_html=True)


with tab1:
    # Use advice already fetched in Input Form
    current_advice = st.session_state.get("advice", {})
    if current_advice.get("mode") == "personalised":
        render_advice(current_advice)
    else:
        # Fetch personalised advice now
        if st.button("🎯 Get Personalised Advice", key="btn_personalised"):
            with st.spinner("✨ Asking Gemini..."):
                try:
                    from gemini.advisor import get_advice
                    adv = get_advice(result, mode="personalised")
                    st.session_state["advice"] = adv
                    st.rerun()
                except Exception as e:
                    st.error(f"Gemini error: {e}")

with tab2:
    if st.button("📊 Get Monthly Summary", key="btn_monthly"):
        with st.spinner("✨ Asking Gemini..."):
            try:
                from gemini.advisor import get_advice
                from database.db_connect import fetch_user_history
                history = fetch_user_history(user_name)
                adv = get_advice(result, user_history=history, mode="monthly_summary")
                st.session_state["monthly_advice"] = adv
            except Exception as e:
                st.error(f"Error: {e}")

    if "monthly_advice" in st.session_state:
        render_advice(st.session_state["monthly_advice"])

with tab3:
    if st.button("📈 Get Goal Plan", key="btn_goal"):
        with st.spinner("✨ Asking Gemini..."):
            try:
                from gemini.advisor import get_advice
                from database.db_connect import fetch_user_history
                history = fetch_user_history(user_name)
                adv = get_advice(result, user_history=history, mode="goal_planning")
                st.session_state["goal_advice"] = adv
            except Exception as e:
                st.error(f"Error: {e}")

    if "goal_advice" in st.session_state:
        render_advice(st.session_state["goal_advice"])

st.markdown("<hr/>", unsafe_allow_html=True)

#re-analyze button
st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='color:#4A5568; font-size:0.9rem;'>
            Want to update your numbers?
        </div>
    </div>
""", unsafe_allow_html=True)

if st.button("📝 Re-Analyze Finances", use_container_width=True):
    st.switch_page("pages/1_Input_Form.py")