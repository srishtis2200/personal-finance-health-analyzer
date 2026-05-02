import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, 'model', 'finance_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'model', 'label_encoder.pkl')

RAW_FEATURES = [
    'monthly_income', 'rent', 'food', 'emi', 'transport',
    'subscriptions', 'savings', 'emergency_fund_months', 'dependents'
]

FEATURE_LABELS = {
    'monthly_income':        'Monthly income',
    'rent':                  'Rent expense',
    'food':                  'Food expense',
    'emi':                   'EMI / loan repayment',
    'transport':             'Transport expense',
    'subscriptions':         'Subscriptions / entertainment',
    'savings':               'Monthly savings',
    'emergency_fund_months': 'Emergency fund (months)',
    'dependents':            'Number of dependents',
}

SCORE_WEIGHTS  = {'Stable': 1.0, 'At Risk': 0.5, 'Critical': 0.0}
POSITIVE_COLOR = '#10B981'
NEGATIVE_COLOR = '#EF4444'


class FinanceExplainer:

    def __init__(self):
        self.model     = joblib.load(MODEL_PATH)
        self.encoder   = joblib.load(ENCODER_PATH)
        self.explainer = shap.TreeExplainer(self.model)
        print("[Explainer] Model loaded")
        print("[Explainer] Classes:", list(self.encoder.classes_))


    def _validate_input(self, user_input):
        for f in RAW_FEATURES:
            if f not in user_input:
                raise ValueError(f"Missing: {f}")
        row = {f: float(user_input[f]) for f in RAW_FEATURES}
        return pd.DataFrame([row], columns=RAW_FEATURES)


    def _compute_score(self, proba):
        score = 0.0
        for i, cls in enumerate(self.encoder.classes_):
            score += proba[i] * SCORE_WEIGHTS.get(cls, 0.0)
        return int(round(score * 100))


    def _get_shap_values(self, X):
        raw = self.explainer.shap_values(X)
        arr = np.array(raw)

        print("DEBUG shap shape:", arr.shape)

        # fixed: was doing shap_values[pred][0] which crashed
        # now normalise to (9, 3) first
        if arr.shape == (9, 3):
            return arr
        elif arr.shape == (1, 9, 3):
            return arr[0]
        elif arr.shape == (3, 1, 9):
            return arr[:, 0, :].T
        else:
            raise ValueError(f"Unknown shape: {arr.shape}")


    def _extract_factors(self, shap_matrix, pred, pred_label, feature_names):
        sv        = shap_matrix[:, pred].flatten()
        total_abs = np.sum(np.abs(sv)) + 1e-9

        factors = []
        for i, feat in enumerate(feature_names):
            factors.append({
                'feature':    feat,
                'label':      FEATURE_LABELS.get(feat, feat),
                'shap_value': float(sv[i]),
                'impact_pct': float(abs(sv[i]) / total_abs * 100),
            })

        factors = sorted(factors, key=lambda x: abs(x['shap_value']), reverse=True)

        hurting, helping = [], []

        # fixed sign convention for Stable vs At Risk/Critical
        if pred_label == 'Stable':
            for f in factors:
                (helping if f['shap_value'] > 0 else hurting).append(f)
        else:
            for f in factors:
                (hurting if f['shap_value'] > 0 else helping).append(f)

        return hurting[:3], helping[:2]


    def _generate_narrative(self, hurting, helping, user_input):
        income = float(user_input['monthly_income'])
        lines  = []

        for f in hurting:
            feat = f['feature']
            val  = float(user_input.get(feat, 0))
            pct  = val / income * 100

            if feat == 'savings':
                lines.append(
                    f"⚠️  Monthly savings are only {pct:.1f}% of income "
                    f"— recommended is at least 20%. "
                    f"This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'emi':
                lines.append(
                    f"⚠️  EMI is {pct:.1f}% of income "
                    f"— RBI limit is 30%. "
                    f"This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'rent':
                lines.append(
                    f"⚠️  Rent is {pct:.1f}% of income "
                    f"— recommended cap is 35%. "
                    f"This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'emergency_fund_months':
                lines.append(
                    f"⚠️  Emergency fund is only {val:.0f} months "
                    f"— target is 3-6 months. "
                    f"This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'dependents':
                lines.append(
                    f"⚠️  Supporting {int(val)} dependent(s) increases financial risk. "
                    f"(impact ~{f['impact_pct']:.0f}%)"
                )
            else:
                lines.append(
                    f"⚠️  {f['label']} ({pct:.1f}% of income) is affecting your score. "
                    f"(impact ~{f['impact_pct']:.0f}%)"
                )

        for f in helping:
            feat = f['feature']
            val  = float(user_input.get(feat, 0))
            pct  = val / income * 100
            lines.append(f"✅  {f['label']} is working in your favour")

        return lines


    def _build_shap_chart(self, shap_vals):
        sv     = np.array(shap_vals).flatten()
        order  = np.argsort(sv)
        sv_ord = sv[order]
        labels = [FEATURE_LABELS.get(RAW_FEATURES[i], RAW_FEATURES[i]) for i in order]
        colors = [NEGATIVE_COLOR if v > 0 else POSITIVE_COLOR for v in sv_ord]

        fig = go.Figure(go.Bar(
            x=sv_ord.tolist(), y=labels, orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in sv_ord],
            textposition='outside',
        ))
        fig.update_layout(
            title=dict(text="SHAP Feature Contributions", font=dict(size=16)),
            xaxis_title="SHAP value",
            height=420,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_vline(x=0, line_width=1, line_color='#888')
        return fig


    def explain(self, user_input):
        X     = self._validate_input(user_input)
        pred  = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        label = self.encoder.inverse_transform([pred])[0]

        score      = self._compute_score(proba)
        confidence = float(np.max(proba))

        shap_matrix = self._get_shap_values(X)
        shap_vals   = shap_matrix[:, pred].flatten()

        hurting, helping = self._extract_factors(
            shap_matrix, pred, label, RAW_FEATURES
        )

        narrative  = self._generate_narrative(hurting, helping, user_input)
        shap_chart = self._build_shap_chart(shap_vals)

        return {
            'score':           score,
            'category':        label,
            'confidence':      confidence,
            'probabilities':   {
                cls: float(p) for cls, p in zip(self.encoder.classes_, proba)
            },
            'hurting_factors': hurting,
            'helping_factors': helping,
            'narrative':       narrative,
            'shap_chart':      shap_chart,
            'shap_values_raw': shap_vals,
        }


if __name__ == "__main__":

    test_user = {
        'monthly_income':        55000,
        'rent':                  12000,
        'food':                   8000,
        'emi':                   10000,
        'transport':              3000,
        'subscriptions':          1000,
        'savings':               12000,
        'emergency_fund_months':  4,
        'dependents':             1,
    }

    explainer = FinanceExplainer()
    result    = explainer.explain(test_user)

    print("Score      :", result['score'])
    print("Category   :", result['category'])
    print("Confidence :", f"{result['confidence']*100:.1f}%")
    print("\nNarrative:")
    for line in result['narrative']:
        print(" ", line)
    print("\nChart:", type(result['shap_chart']).__name__)