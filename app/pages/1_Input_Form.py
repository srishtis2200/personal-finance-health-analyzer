"""
pages/1_Input_Form.py
Collects 9 financial inputs → runs explainer → gets Gemini advice → saves to MySQL → redirects to Dashboard
  1. FinanceExplainer cached with st.cache_resource — not re-instantiated every click
  2. SHAP format conversion with isinstance guard — handles both dict and tuple formats
  3. Gemini errors logged to st.session_state for dashboard visibility
  4. Strong validation — expenses > income, negative disposable income, savings > income
"""

import streamlit as st
import sys
import os
import traceback
from datetime import datetime

#path fix — so imports work from app/pages/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#page config
st.set_page_config(
    page_title="Analyze Finances — Finance Health Analyzer",
    page_icon="📝",
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
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(27,79,138,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(27,79,138,0.4);
    }
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1.5px solid var(--border) !important;
    }
    .tip-box {
        background: #EBF4FF;
        border: 1px solid #BEE3F8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: var(--text-light);
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #92400E;
        margin-bottom: 0.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


#Cache explainer — load model ONCE, not on every button click
@st.cache_resource(show_spinner="Loading ML model...")
def load_explainer():
    """
    Cached with st.cache_resource — model loads once per session.
    Without this, FinanceExplainer() re-instantiates on every click,
    reloading the pkl file and rebuilding SHAP TreeExplainer each time.
    """
    from shap_explainer.explainer import FinanceExplainer
    return FinanceExplainer()


#Safe SHAP format converter
def _to_tuples(factors: list) -> list:
    """
    Convert SHAP factors to (feature, shap_value) tuples for Gemini advisor.
    Guards against both dict format AND already-tuple format — no crash either way.

    explainer returns : [{"feature": "emi", "shap_value": 0.4, ...}]
    gemini expects    : [("emi", 0.4)]
    """
    result = []
    for f in factors:
        if isinstance(f, dict):
            result.append((f.get("feature", "unknown"), f.get("shap_value", 0.0)))
        elif isinstance(f, (list, tuple)) and len(f) >= 2:
            result.append((str(f[0]), float(f[1])))
        # Skip malformed entries silently
    return result


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
        <h1>📝 Analyze Your Finances</h1>
        <p>Enter your monthly income and expenses — takes less than 2 minutes</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='tip-box'>
        💡 <b>Tip:</b> Use your actual monthly averages for the most accurate score.
        All amounts are in Indian Rupees (₹). Your data is saved privately by your name.
    </div>
""", unsafe_allow_html=True)

#user identity
st.markdown("<div class='section-header'>👤 Your Identity</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    user_name = st.text_input(
        "Your Name",
        placeholder="e.g. Rahul Sharma",
        help="Used to track your month-over-month progress"
    )
with col2:
    now = datetime.now()
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    selected_month = st.selectbox("Month", months, index=now.month - 1)
    selected_year  = st.number_input("Year", min_value=2020, max_value=2030, value=now.year)

month_num  = months.index(selected_month) + 1
month_year = f"{selected_year}-{month_num:02d}"

st.markdown("<hr/>", unsafe_allow_html=True)

#income
st.markdown("<div class='section-header'>💵 Monthly Income</div>", unsafe_allow_html=True)

monthly_income = st.number_input(
    "Total Monthly Income (₹)",
    min_value=0, max_value=1000000,
    value=45000, step=1000,
    help="Your total take-home income after tax"
)

st.markdown("<hr/>", unsafe_allow_html=True)

#expenses
st.markdown("<div class='section-header'>💸 Monthly Expenses</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    rent = st.number_input(
        "🏠 Rent / Home Loan EMI (₹)",
        min_value=0, max_value=500000,
        value=12000, step=500,
        help="Monthly rent or home loan EMI"
    )
    food = st.number_input(
        "🍱 Food & Groceries (₹)",
        min_value=0, max_value=100000,
        value=6000, step=500,
        help="Monthly spending on food, groceries, dining out"
    )
    emi = st.number_input(
        "💳 Other EMIs / Loan Payments (₹)",
        min_value=0, max_value=200000,
        value=5000, step=500,
        help="Personal loans, car loans, credit card EMIs (excluding home loan)"
    )
    transport = st.number_input(
        "🚗 Transport (₹)",
        min_value=0, max_value=50000,
        value=2000, step=500,
        help="Fuel, public transport, cab rides"
    )

with col2:
    subscriptions = st.number_input(
        "📱 Subscriptions & Utilities (₹)",
        min_value=0, max_value=50000,
        value=1500, step=100,
        help="Netflix, Spotify, electricity, internet, phone bills"
    )
    savings = st.number_input(
        "🏦 Monthly Savings (₹)",
        min_value=0, max_value=500000,
        value=5000, step=500,
        help="Amount you save/invest every month (SIP, FD, RD, etc.)"
    )
    emergency_fund_months = st.number_input(
        "🛡️ Emergency Fund (months of expenses)",
        min_value=0.0, max_value=24.0,
        value=2.0, step=0.5,
        help="How many months of expenses you have saved as emergency fund"
    )
    dependents = st.number_input(
        "👨‍👩‍👧 Number of Dependents",
        min_value=0, max_value=10,
        value=1, step=1,
        help="Family members financially dependent on you"
    )

st.markdown("<hr/>", unsafe_allow_html=True)

#Live expense summary 
total_expenses = rent + food + emi + transport + subscriptions
disposable     = monthly_income - total_expenses - savings
savings_rate   = (savings / monthly_income * 100) if monthly_income > 0 else 0

st.markdown("<div class='section-header'>📊 Quick Summary</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Expenses", f"₹{total_expenses:,.0f}")
with m2:
    st.metric("Monthly Savings", f"₹{savings:,.0f}")
with m3:
    st.metric("Savings Rate", f"{savings_rate:.1f}%")
with m4:
    disp_color = "normal" if disposable >= 0 else "inverse"
    st.metric("Disposable Income", f"₹{disposable:,.0f}",
              delta="Negative" if disposable < 0 else None,
              delta_color="inverse" if disposable < 0 else "normal")

#Real-time soft warnings before submission 
if monthly_income > 0:
    if total_expenses + savings > monthly_income:
        st.markdown(f"""
            <div class='warning-box'>
                ⚠️ <b>Expenses + Savings (₹{total_expenses + savings:,.0f})
                exceed your income (₹{monthly_income:,.0f}).</b>
                Please review your numbers before submitting.
            </div>
        """, unsafe_allow_html=True)
    if disposable < 0:
        st.markdown(f"""
            <div class='warning-box'>
                ⚠️ <b>Negative disposable income (₹{disposable:,.0f}).</b>
                This means your expenses and savings exceed your income.
            </div>
        """, unsafe_allow_html=True)
    if savings_rate < 5 and savings > 0:
        st.markdown("""
            <div class='warning-box'>
                ⚠️ <b>Savings rate below 5%.</b>
                Financial advisors recommend saving at least 20% of income.
            </div>
        """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

#analyze button
analyze_clicked = st.button("🔍 Analyze My Financial Health", use_container_width=True)

if analyze_clicked:

    #Strong validation — all edge cases
    errors = []

    if not user_name.strip():
        errors.append("Please enter your name.")

    if monthly_income == 0:
        errors.append("Monthly income cannot be zero.")

    if monthly_income > 0 and (total_expenses + savings) > monthly_income * 1.5:
        errors.append(
            f"Expenses + Savings (₹{total_expenses + savings:,.0f}) are more than "
            f"150% of your income (₹{monthly_income:,.0f}). Please check your numbers."
        )

    if savings > monthly_income:
        errors.append(
            f"Savings (₹{savings:,.0f}) cannot exceed monthly income (₹{monthly_income:,.0f})."
        )

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        st.stop()

    #soft warning for negative disposable — allow but warn
    if disposable < 0:
        st.warning(
            f"⚠️ Your disposable income is negative (₹{disposable:,.0f}). "
            "Proceeding with analysis — this will likely result in a Critical score."
        )

    #build input dict
    input_data = {
        "monthly_income":        float(monthly_income),
        "rent":                  float(rent),
        "food":                  float(food),
        "emi":                   float(emi),
        "transport":             float(transport),
        "subscriptions":         float(subscriptions),
        "savings":               float(savings),
        "emergency_fund_months": float(emergency_fund_months),
        "dependents":            int(dependents),
    }

    #Run ML explainer
    with st.spinner("🤖 Running ML analysis..."):
        try:
            explainer = load_explainer()          # ← FIX 1: cached, not re-created
            result    = explainer.explain(input_data)
            result["user_input"] = input_data

            #Safe format conversion with isinstance guard
            result["hurting_factors"] = _to_tuples(result.get("hurting_factors", []))
            result["helping_factors"] = _to_tuples(result.get("helping_factors", []))

        except Exception as e:
            st.error(f"❌ ML Analysis failed: {e}")
            st.expander("🔍 Full error details").write(traceback.format_exc())
            st.stop()

    st.success("✅ ML analysis complete!")

    #Get Gemini advice
    with st.spinner("✨ Getting AI advice from Gemini..."):
        gemini_error = None
        try:
            from gemini.advisor import get_advice
            advice = get_advice(result, mode="personalised")

            # FIX 3: Log Gemini JSON errors to session state for dashboard visibility
            if "error" in advice:
                gemini_error = advice["error"]

        except Exception as e:
            gemini_error = traceback.format_exc()
            advice = {
                "mode": "personalised",
                "summary": "AI advice temporarily unavailable — your score and charts are accurate.",
                "sections": [{
                    "title": "⚠️ Gemini Unavailable",
                    "points": [
                        "The AI advisor could not be reached right now.",
                        "Your ML score and SHAP analysis below are fully accurate.",
                        "Try refreshing the Dashboard to retry advice."
                    ]
                }],
                "disclaimer": "This advice is AI-generated for educational purposes only."
            }

        #Store error in session state — Dashboard can show it if needed
        if gemini_error:
            st.session_state["gemini_error"] = gemini_error
            st.warning("⚠️ Gemini advice unavailable — score and charts are still accurate.")
        elif "gemini_error" in st.session_state:
            del st.session_state["gemini_error"]   # Clear old errors on success

    #Save to MySQL
    with st.spinner("💾 Saving to database..."):
        try:
            from database.db_connect import insert_record
            saved = insert_record(user_name.strip(), month_year, input_data, result)
            if saved:
                st.success("✅ Data saved to database!")
            else:
                st.warning("⚠️ Could not save to database — results still shown below.")
        except Exception as e:
            st.warning(f"⚠️ Database save failed: {e}")
            st.session_state["db_error"] = traceback.format_exc()

    #Store in session state 
    st.session_state["result"]     = result
    st.session_state["advice"]     = advice
    st.session_state["input_data"] = input_data
    st.session_state["user_name"]  = user_name.strip()
    st.session_state["month_year"] = month_year

    #Score preview 
    score    = result.get("score", 0)
    category = result.get("category", "Unknown")
    color    = "#2ECC71" if category == "Stable" else "#F39C12" if category == "At Risk" else "#E74C3C"
    emoji    = "✅" if category == "Stable" else "⚠️" if category == "At Risk" else "🚨"

    st.markdown(f"""
        <div style='background:white; border-radius:12px; padding:1.5rem;
                    border-left:6px solid {color}; box-shadow:0 2px 12px rgba(0,0,0,0.08);
                    text-align:center; margin-top:1rem;'>
            <div style='font-size:3rem; font-weight:700; color:{color};'>{score}</div>
            <div style='font-size:1rem; color:#4A5568; margin-top:0.25rem;'>
                Financial Health Score / 100
            </div>
            <div style='font-size:1.2rem; font-weight:600; color:{color}; margin-top:0.5rem;'>
                {emoji} {category}
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.info("📊 Click **Dashboard** in the sidebar to see your full analysis, SHAP charts, and AI advice!")