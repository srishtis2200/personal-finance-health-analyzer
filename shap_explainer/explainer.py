##shap explainer

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap

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

SCORE_WEIGHTS = {'Stable': 1.0, 'At Risk': 0.5, 'Critical': 0.0}


class FinanceExplainer:

    def __init__(self):
        self.model     = joblib.load(MODEL_PATH)
        self.encoder   = joblib.load(ENCODER_PATH)
        self.explainer = shap.TreeExplainer(self.model)
        print("[Explainer] Loaded successfully")


    def _validate_input(self, user_input):
        for f in RAW_FEATURES:
            if f not in user_input:
                raise ValueError(f"Missing: {f}")
        return pd.DataFrame([user_input], columns=RAW_FEATURES)


    def _compute_score(self, proba):
        score = 0
        for i, cls in enumerate(self.encoder.classes_):
            score += proba[i] * SCORE_WEIGHTS.get(cls, 0)
        return int(score * 100)


    def _get_shap_values(self, X):
        # get shap values for all classes
        shap_values = self.explainer.shap_values(X)
        return shap_values


    def _extract_factors(self, shap_values, pred_class, feature_names):
        # pick shap values for predicted class
        sv = shap_values[pred_class][0]  # assumes list of arrays

        total = np.sum(np.abs(sv)) + 1e-9

        factors = []
        for i, feat in enumerate(feature_names):
            factors.append({
                'feature':    feat,
                'label':      FEATURE_LABELS.get(feat, feat),
                'shap_value': float(sv[i]),
                'impact_pct': float(abs(sv[i]) / total * 100)
            })

        factors = sorted(factors, key=lambda x: abs(x['shap_value']), reverse=True)

        hurting = [f for f in factors if f['shap_value'] > 0][:3]
        helping = [f for f in factors if f['shap_value'] < 0][:2]

        return hurting, helping


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
                    f"— recommended is 20%. This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'emi':
                lines.append(
                    f"⚠️  EMI is {pct:.1f}% of income "
                    f"— RBI limit is 30%. This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'rent':
                lines.append(
                    f"⚠️  Rent is {pct:.1f}% of income "
                    f"— cap is 35%. This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            elif feat == 'emergency_fund_months':
                lines.append(
                    f"⚠️  Emergency fund is only {val:.0f} months "
                    f"— target is 3-6 months. This reduced your score by ~{f['impact_pct']:.0f}%"
                )
            else:
                lines.append(
                    f"⚠️  {f['label']} is affecting your score "
                    f"(impact ~{f['impact_pct']:.0f}%)"
                )

        for f in helping:
            lines.append(f"✅  {f['label']} is healthy — working in your favour")

        return lines


    def explain(self, user_input):
        X     = self._validate_input(user_input)
        pred  = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        label = self.encoder.inverse_transform([pred])[0]

        score = self._compute_score(proba)

        shap_values = self._get_shap_values(X)

        hurting, helping = self._extract_factors(shap_values, pred, RAW_FEATURES)

        narrative = self._generate_narrative(hurting, helping, user_input)

        return {
            'score':           score,
            'category':        label,
            'probabilities':   {
                cls: float(p) for cls, p in zip(self.encoder.classes_, proba)
            },
            'hurting_factors': hurting,
            'helping_factors': helping,
            'narrative':       narrative,
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

    print("Score    :", result['score'])
    print("Category :", result['category'])
    print("\nNarrative:")
    for line in result['narrative']:
        print(" ", line)