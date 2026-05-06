"""
pages/3_History.py
Displays month-over-month score trends for any user.
Reads from MySQL via fetch_user_history().
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px

#path fix
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#page configuration
st.set_page_config(
    page_title="History — Finance Health Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

#global css
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
    .history-row {
        background: var(--card);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
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
        <h1>📈 Financial History & Trends</h1>
        <p>Track your month-over-month financial health score progress</p>
    </div>
""", unsafe_allow_html=True)

#load users from db
try:
    from database.db_connect import fetch_all_users, fetch_user_history
    all_users = fetch_all_users()
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

if not all_users:
    st.markdown("""
        <div style='text-align:center; padding:3rem 1rem;'>
            <div style='font-size:4rem;'>📭</div>
            <div style='font-family:Poppins,sans-serif; font-size:1.3rem;
                        font-weight:600; color:#1B4F8A; margin:1rem 0 0.5rem 0;'>
                No History Yet
            </div>
            <div style='color:#4A5568; font-size:0.95rem;'>
                Submit your first analysis via <b>Analyze Finances</b>
                and your history will appear here.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

#user selector
st.markdown("<div class='section-header'>👤 Select User</div>", unsafe_allow_html=True)

# Pre-select logged in user if available
default_user = st.session_state.get("user_name", all_users[0])
default_idx  = all_users.index(default_user) if default_user in all_users else 0

selected_user = st.selectbox(
    "Choose a user to view history",
    all_users,
    index=default_idx,
    label_visibility="collapsed"
)

#fetch history
history = fetch_user_history(selected_user)

if not history:
    st.info(f"No history found for **{selected_user}**. Submit at least one analysis first.")
    st.stop()

#parse history data
months     = [r["month_year"]    for r in history]
scores     = [r["health_score"]  for r in history]
categories = [r["risk_category"] for r in history]
incomes    = [float(r.get("monthly_income", 0)) for r in history]
savings_l  = [float(r.get("savings", 0))        for r in history]
emi_l      = [float(r.get("emi", 0))            for r in history]

color_map  = {"Stable": "#2ECC71", "At Risk": "#F39C12", "Critical": "#E74C3C"}
point_colors = [color_map.get(c, "#1B4F8A") for c in categories]

st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 1 — Summary Metrics
st.markdown("<div class='section-header'>📊 Summary Statistics</div>", unsafe_allow_html=True)

latest_score  = scores[-1]
best_score    = max(scores)
avg_score     = sum(scores) / len(scores)
total_months  = len(scores)
score_change  = scores[-1] - scores[-2] if len(scores) >= 2 else 0
trend_str     = f"+{score_change}" if score_change > 0 else str(score_change)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Latest Score",   f"{latest_score}/100",
              delta=f"{trend_str} vs last month" if len(scores) >= 2 else None)
with m2:
    st.metric("Best Score",     f"{best_score}/100")
with m3:
    st.metric("Average Score",  f"{avg_score:.1f}/100")
with m4:
    st.metric("Months Tracked", f"{total_months}")
with m5:
    latest_cat = categories[-1]
    cat_emoji  = {"Stable": "✅", "At Risk": "⚠️", "Critical": "🚨"}.get(latest_cat, "📊")
    st.metric("Current Status", f"{cat_emoji} {latest_cat}")

st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 2 — Score Trend Chart
st.markdown("<div class='section-header'>📈 Score Trend Over Time</div>", unsafe_allow_html=True)

fig_trend = go.Figure()

#shaded zone bands
fig_trend.add_hrect(y0=70,  y1=100, fillcolor="#2ECC71", opacity=0.06, line_width=0, annotation_text="Stable Zone",   annotation_position="top right")
fig_trend.add_hrect(y0=40,  y1=70,  fillcolor="#F39C12", opacity=0.06, line_width=0, annotation_text="At Risk Zone",  annotation_position="top right")
fig_trend.add_hrect(y0=0,   y1=40,  fillcolor="#E74C3C", opacity=0.06, line_width=0, annotation_text="Critical Zone", annotation_position="top right")

#score lines
fig_trend.add_trace(go.Scatter(
    x=months,
    y=scores,
    mode="lines+markers+text",
    name="Health Score",
    line=dict(color="#1B4F8A", width=3),
    marker=dict(
        size=12,
        color=point_colors,
        line=dict(color="white", width=2)
    ),
    text=[str(s) for s in scores],
    textposition="top center",
    textfont=dict(size=11, color="#1B4F8A", family="Inter"),
    hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>"
))

fig_trend.update_layout(
    height=380,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter", size=12, color="#1A202C"),
    xaxis=dict(
        title="Month",
        showgrid=True,
        gridcolor="#F0F4F8",
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        title="Health Score",
        range=[0, 105],
        showgrid=True,
        gridcolor="#F0F4F8",
        tickfont=dict(size=11)
    ),
    margin=dict(l=40, r=40, t=30, b=40),
    showlegend=False,
    hovermode="x unified"
)

st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 3 — Expense Trends
if len(history) >= 2:
    st.markdown("<div class='section-header'>💸 Expense Trends</div>", unsafe_allow_html=True)

    fig_exp = go.Figure()

    expense_keys = [
        ("monthly_income", "Income",        "#1B4F8A", "solid"),
        ("savings",        "Savings",        "#2ECC71", "solid"),
        ("emi",            "EMI",            "#E74C3C", "dash"),
        ("rent",           "Rent",           "#F39C12", "dash"),
        ("food",           "Food",           "#9B59B6", "dot"),
    ]

    for key, label, clr, dash in expense_keys:
        vals = [float(r.get(key, 0)) for r in history]
        fig_exp.add_trace(go.Scatter(
            x=months, y=vals,
            mode="lines+markers",
            name=label,
            line=dict(color=clr, width=2, dash=dash),
            marker=dict(size=7, color=clr),
            hovertemplate=f"<b>{label}</b>: ₹%{{y:,.0f}}<extra></extra>"
        ))

    fig_exp.update_layout(
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#1A202C"),
        xaxis=dict(title="Month", showgrid=True, gridcolor="#F0F4F8"),
        yaxis=dict(title="Amount (₹)", showgrid=True, gridcolor="#F0F4F8",
                   tickformat=",.0f"),
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig_exp, use_container_width=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

# SECTION 4 — Monthly Records Table
st.markdown("<div class='section-header'>🗃️ Monthly Records</div>", unsafe_allow_html=True)

for i, record in enumerate(reversed(history)):   # newest first
    month      = record.get("month_year", "?")
    score_val  = record.get("health_score", 0)
    cat        = record.get("risk_category", "Unknown")
    income_val = float(record.get("monthly_income", 0))
    savings_v  = float(record.get("savings", 0))
    conf       = float(record.get("confidence", 0))

    cat_color  = color_map.get(cat, "#1B4F8A")
    cat_emoji  = {"Stable": "✅", "At Risk": "⚠️", "Critical": "🚨"}.get(cat, "📊")

    # Score change indicator
    idx = len(history) - 1 - i
    if idx > 0:
        prev_score = history[idx - 1].get("health_score", score_val)
        delta      = score_val - prev_score
        delta_str  = f"▲ +{delta}" if delta > 0 else f"▼ {delta}" if delta < 0 else "— 0"
        delta_color= "#2ECC71" if delta > 0 else "#E74C3C" if delta < 0 else "#4A5568"
    else:
        delta_str  = "—"
        delta_color= "#4A5568"

    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1.5, 1.5, 1.5, 1])
    with c1:
        st.markdown(f"**{month}**")
    with c2:
        st.markdown(f"<span style='font-size:1.1rem; font-weight:700; color:{cat_color};'>{score_val}</span>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<span style='color:{cat_color}; font-weight:600;'>{cat_emoji} {cat}</span>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"₹{income_val:,.0f}")
    with c5:
        st.markdown(f"₹{savings_v:,.0f} saved")
    with c6:
        st.markdown(f"<span style='color:{delta_color}; font-weight:600;'>{delta_str}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.4rem 0;'/>", unsafe_allow_html=True)

# SECTION 5 — Gemini Goal Planning (from history)
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🎯 AI Goal Planning (Based on Your History)</div>", unsafe_allow_html=True)

if st.button("📈 Generate Goal Plan from History", use_container_width=True):
    with st.spinner("✨ Gemini is analyzing your trend..."):
        try:
            from gemini.advisor import get_advice
            result_stub = {
                "score":    scores[-1],
                "category": categories[-1],
                "confidence": float(history[-1].get("confidence", 0.7)),
                "probabilities": {},
                "hurting_factors": [],
                "helping_factors": [],
                "narrative": [],
                "user_input": {
                    "monthly_income": float(history[-1].get("monthly_income", 0))
                }
            }
            goal_advice = get_advice(result_stub, user_history=history, mode="goal_planning")
            st.session_state["history_goal_advice"] = goal_advice
        except Exception as e:
            st.error(f"Gemini error: {e}")

if "history_goal_advice" in st.session_state:
    adv = st.session_state["history_goal_advice"]
    summary = adv.get("summary", "")
    if summary:
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1B4F8A15,#2E6FBF25);
                        border:1px solid #2E6FBF; border-radius:10px;
                        padding:1rem 1.25rem; margin-bottom:1rem;'>
                <div style='font-size:0.8rem; color:#1B4F8A; font-weight:600;
                            margin-bottom:0.25rem;'>GOAL SUMMARY</div>
                <div style='color:#1A202C; font-size:0.95rem; font-weight:500;'>
                    🎯 {summary}
                </div>
            </div>
        """, unsafe_allow_html=True)

    for section in adv.get("sections", []):
        title  = section.get("title", "")
        points = section.get("points", [])
        if not points:
            continue
        border = "#3498DB" if "🗓️" in title else "#2ECC71" if "🔑" in title else \
                 "#F39C12" if "⚡" in title else "#9B59B6"
        points_html = "".join([f"<li>{p}</li>" for p in points])
        st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:1.2rem 1.5rem;
                        margin-bottom:0.75rem; border-left:4px solid {border};
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <h4 style='color:#1B4F8A; margin:0 0 0.5rem 0; font-size:1rem;
                           font-weight:600;'>{title}</h4>
                <ul style='margin:0; padding-left:1.2rem; color:#4A5568;
                           font-size:0.92rem; line-height:1.7;'>{points_html}</ul>
            </div>
        """, unsafe_allow_html=True)

    disclaimer = adv.get("disclaimer", "")
    if disclaimer:
        st.markdown(f"""
            <div style='background:#EBF4FF; border:1px solid #BEE3F8; border-radius:8px;
                        padding:0.75rem 1rem; font-size:0.82rem; color:#4A5568;
                        margin-top:1rem;'>
                ⚠️ {disclaimer}
            </div>
        """, unsafe_allow_html=True)