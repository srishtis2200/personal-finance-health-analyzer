"""
gemini/advisor.py
Phase 5 — Google Gemini AI Financial Advisor
Uses the new google-genai SDK (v1.75.0+)

One function, three modes:
  - personalised    : SHAP-grounded specific advice for current submission
  - monthly_summary : What changed vs last month (needs history)
  - goal_planning   : Timeline to reach Stable (needs full trend)

Usage:
    from gemini.advisor import get_advice
    advice = get_advice(result_dict, user_history=history_df, mode="personalised")
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

#configure Gemini client once at import time 
_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found in .env — "
        "get a free key at https://aistudio.google.com/app/apikey"
    )

try:
    from google import genai
    from google.genai import types
    _CLIENT = genai.Client(api_key=_API_KEY)
    _MODEL = "gemini-2.0-flash"
except ImportError:
    raise ImportError("google-genai package not found. Run: pip install google-genai")


#internal helpers

def _income_bracket(monthly_income: float) -> str:
    if monthly_income < 20000:
        return "low income (under ₹20,000/month)"
    elif monthly_income < 45000:
        return "lower-middle income (₹20,000–₹45,000/month)"
    elif monthly_income < 80000:
        return "middle income (₹45,000–₹80,000/month)"
    else:
        return "upper income (above ₹80,000/month)"


def _format_factors(factors: list) -> str:
    if not factors:
        return "None identified"
    lines = []
    for f in factors:
        if isinstance(f, (list, tuple)) and len(f) == 2:
            lines.append(f"  • {f[0]}: impact score {f[1]:.3f}")
        else:
            lines.append(f"  • {f}")
    return "\n".join(lines)


def _safe_get(result_dict: dict, key: str, default=None):
    return result_dict.get(key, default)


def _call_gemini(prompt: str) -> str:
    """Call Gemini API using new google-genai SDK."""
    response = _CLIENT.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=1024,
            top_p=0.9,
        )
    )
    return response.text


def _parse_gemini_response(raw_text: str, mode: str) -> dict:
    """
    Parse Gemini response into structured dict.
    Handles markdown fences, extracts JSON robustly.
    Falls back to text parsing if JSON fails.
    """
    try:
        # Step 1 — Extract content from ```json ... ``` fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        if fence_match:
            clean = fence_match.group(1).strip()
        else:
            clean = raw_text.strip()
        # Step 2 — Find outermost { } JSON block
        brace_match = re.search(r"\{[\s\S]*\}", clean)
        if brace_match:
            clean = brace_match.group(0)
        parsed = json.loads(clean)
        if "sections" in parsed and "summary" in parsed:
            parsed.setdefault("mode", mode)
            parsed.setdefault("disclaimer",
                "This advice is AI-generated for educational purposes only. "
                "Consult a certified financial advisor for professional guidance.")
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback text parsing
    sections, current_title, current_points = [], None, []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (line.endswith(":") or re.match(r"^[#*\-]?\s*[A-Z🔴🟡✅📈💡⚠️]", line)) \
                and len(line) < 80 and not line.startswith("•"):
            if current_title and current_points:
                sections.append({"title": current_title, "points": current_points})
            current_title, current_points = line.lstrip("#*- ").rstrip(":"), []
        elif line.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5.")):
            point = re.sub(r"^[•\-*\d.]\s*", "", line).strip()
            if point:
                current_points.append(point)
        elif current_title:
            current_points.append(line)

    if current_title and current_points:
        sections.append({"title": current_title, "points": current_points})
    if not sections:
        sections = [{"title": "💡 Advice", "points": [raw_text]}]

    summary = sections[0]["points"][0] if sections and sections[0]["points"] else "See advice below."
    return {
        "mode": mode,
        "summary": summary,
        "sections": sections,
        "disclaimer": (
            "This advice is AI-generated for educational purposes only. "
            "Consult a certified financial advisor for professional guidance."
        )
    }


#prompt builders

def _build_personalised_prompt(result_dict: dict) -> str:
    score      = _safe_get(result_dict, "score", "N/A")
    category   = _safe_get(result_dict, "category", "Unknown")
    confidence = _safe_get(result_dict, "confidence", 0)
    probs      = _safe_get(result_dict, "probabilities", {})
    hurting    = _safe_get(result_dict, "hurting_factors", [])
    helping    = _safe_get(result_dict, "helping_factors", [])
    narrative  = _safe_get(result_dict, "narrative", [])
    user_input = _safe_get(result_dict, "user_input", {})
    income     = user_input.get("monthly_income", 0) if user_input else 0
    bracket    = _income_bracket(income) if income else "income level unknown"
    prob_str   = ", ".join([f"{k}: {v:.0%}" for k, v in probs.items()]) if probs else "N/A"

    return f"""
You are a practical Indian personal finance advisor.
A user has submitted their monthly financial details and received an ML-based analysis.
Give specific, actionable advice grounded strictly in their data.

== USER FINANCIAL PROFILE ==
Health Score       : {score}/100
Risk Category      : {category} (Confidence: {confidence:.0%})
Class Probabilities: {prob_str}
Income Bracket     : {bracket}

== SHAP-VERIFIED ROOT CAUSES ==
Factors HURTING the score:
{_format_factors(hurting)}

Factors HELPING the score:
{_format_factors(helping)}

== ML MODEL NARRATIVE ==
{chr(10).join(f"  - {n}" for n in narrative) if narrative else "  - Not available"}

== YOUR TASK ==
Return a single raw JSON object. Do NOT wrap in markdown. Do NOT add any text before or after the JSON.
{{
  "summary": "One sentence: the single most important thing this person must do",
  "sections": [
    {{"title": "🔴 What's Hurting Your Score", "points": ["point 1", "point 2", "point 3"]}},
    {{"title": "✅ Immediate Actions (This Month)", "points": ["step 1", "step 2", "step 3"]}},
    {{"title": "📈 30-Day Financial Target", "points": ["target 1", "target 2"]}},
    {{"title": "💡 Long-Term Strategy", "points": ["strategy 1", "strategy 2"]}}
  ],
  "disclaimer": "This advice is AI-generated for educational purposes only. Consult a certified financial advisor for professional guidance."
}}

Rules: Address SHAP hurting factors directly. Use Indian context (₹, PPF, SIP, FD). Keep each point under 2 sentences. Be direct.
""".strip()


def _build_monthly_summary_prompt(result_dict: dict, user_history) -> str:
    score, category = _safe_get(result_dict, "score", "N/A"), _safe_get(result_dict, "category", "Unknown")
    history_str, prev_score, prev_category, score_change = "No previous history.", None, None, ""

    if user_history is not None:
        try:
            records = user_history.to_dict(orient="records") if hasattr(user_history, "to_dict") else list(user_history)
            if len(records) >= 2:
                prev = records[-2]
                prev_score, prev_category = prev.get("health_score", "N/A"), prev.get("risk_category", "Unknown")
                try:
                    delta = int(score) - int(prev_score)
                    direction = "improved" if delta > 0 else "declined" if delta < 0 else "unchanged"
                    score_change = f"Score {direction} by {abs(delta)} points ({prev_score} → {score})"
                except (ValueError, TypeError):
                    score_change = "Score change could not be calculated."
            history_str = "\n".join([
                f"  {r.get('month_year','?')}: Score={r.get('health_score','?')}, Category={r.get('risk_category','?')}"
                for r in records[-6:]
            ])
        except Exception:
            history_str = "History parsing error."

    return f"""
You are a practical Indian personal finance advisor reviewing monthly progress.

CURRENT: Score={score}/100, Category={category}
PREVIOUS: Score={prev_score or 'N/A'}, Category={prev_category or 'N/A'}, Change={score_change or 'N/A'}
LAST 6 MONTHS:
{history_str}

Respond ONLY with valid JSON (no markdown):
{{
  "summary": "One sentence on what changed and why it matters",
  "sections": [
    {{"title": "📊 What Changed This Month", "points": ["change 1", "change 2", "change 3"]}},
    {{"title": "✅ What You Did Right", "points": ["positive 1", "positive 2"]}},
    {{"title": "⚠️ What Needs Attention", "points": ["concern 1", "concern 2"]}},
    {{"title": "🎯 Focus for Next Month", "points": ["focus 1", "focus 2"]}}
  ],
  "disclaimer": "This advice is AI-generated for educational purposes only. Consult a certified financial advisor for professional guidance."
}}
""".strip()


def _build_goal_planning_prompt(result_dict: dict, user_history) -> str:
    score, category = _safe_get(result_dict, "score", "N/A"), _safe_get(result_dict, "category", "Unknown")
    history_str, months_of_data, trend_direction = "No history.", 0, "unknown"

    if user_history is not None:
        try:
            records = user_history.to_dict(orient="records") if hasattr(user_history, "to_dict") else list(user_history)
            months_of_data = len(records)
            if months_of_data >= 2:
                scores = [r.get("health_score", 0) for r in records]
                trend_direction = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "flat"
            history_str = "\n".join([
                f"  {r.get('month_year','?')}: Score={r.get('health_score','?')}, Category={r.get('risk_category','?')}"
                for r in records
            ])
        except Exception:
            history_str = "History parsing error."

    return f"""
You are a practical Indian personal finance advisor creating a goal plan.

CURRENT: Score={score}/100, Category={category}, Months of data={months_of_data}, Trend={trend_direction}
HISTORY:
{history_str}

Create a realistic plan to reach Stable (score ≥ 70). Respond ONLY with valid JSON (no markdown):
{{
  "summary": "Realistic timeline to reach Stable based on current trend",
  "sections": [
    {{"title": "🗓️ Your Timeline to Stable", "points": ["month 1-2: ...", "month 3-4: ...", "month 5-6: ..."]}},
    {{"title": "🔑 Key Milestones", "points": ["milestone 1", "milestone 2", "milestone 3"]}},
    {{"title": "⚡ Quick Wins (Do This Week)", "points": ["quick win 1", "quick win 2", "quick win 3"]}},
    {{"title": "🛡️ Risk Checkpoints", "points": ["if score drops below X do Y", "checkpoint 2"]}}
  ],
  "disclaimer": "This advice is AI-generated for educational purposes only. Consult a certified financial advisor for professional guidance."
}}

Rules: Base timeline on actual trend. Use Indian instruments (SIP, PPF, FD). Milestones must be measurable.
""".strip()


#main public function 

def get_advice(
    result_dict: dict,
    user_history=None,
    mode: str = "personalised"
) -> dict:
    """
    Get AI-powered financial advice from Google Gemini.

    Parameters
    ----------
    result_dict  : dict   Full output from shap_explainer/explainer.py
    user_history : list   Output of db_connect.fetch_user_history() — optional
    mode         : str    'personalised' | 'monthly_summary' | 'goal_planning'

    Returns
    -------
    dict with keys: mode, summary, sections, disclaimer, error (if failed)
    """
    VALID_MODES = {"personalised", "monthly_summary", "goal_planning"}

    if mode not in VALID_MODES:
        return {"mode": mode, "summary": "Invalid mode.", "sections": [], "disclaimer": "",
                "error": f"mode must be one of {VALID_MODES}"}

    if mode in {"monthly_summary", "goal_planning"} and user_history is None:
        label  = "Monthly Summary" if mode == "monthly_summary" else "Goal Planning"
        months = "2 months" if mode == "monthly_summary" else "1 month"
        return {
            "mode": mode,
            "summary": f"No history available for {label}.",
            "sections": [{"title": "⚠️ No History Found", "points": [
                f"Submit at least {months} of data to unlock {label}.",
                "Your data is saved automatically each time you submit the form."
            ]}],
            "disclaimer": (
                "This advice is AI-generated for educational purposes only. "
                "Consult a certified financial advisor for professional guidance."
            )
        }

    if mode == "personalised":
        prompt = _build_personalised_prompt(result_dict)
    elif mode == "monthly_summary":
        prompt = _build_monthly_summary_prompt(result_dict, user_history)
    else:
        prompt = _build_goal_planning_prompt(result_dict, user_history)

    try:
        raw_text = _call_gemini(prompt)
    except Exception as e:
        return {
            "mode": mode,
            "summary": "Could not fetch AI advice at this time.",
            "sections": [{"title": "⚠️ Gemini API Error", "points": [
                "The AI advisor is temporarily unavailable.",
                "Your financial score and SHAP analysis are still accurate.",
                f"Error: {str(e)}"
            ]}],
            "disclaimer": (
                "This advice is AI-generated for educational purposes only. "
                "Consult a certified financial advisor for professional guidance."
            ),
            "error": str(e)
        }

    return _parse_gemini_response(raw_text, mode)