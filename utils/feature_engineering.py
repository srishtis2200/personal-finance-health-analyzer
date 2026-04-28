def engineer_features(income, rent, food, emi, transport,
                      subscriptions, savings, emergency_fund, dependents):

    features = {}

    features["monthly_income"]        = income
    features["savings_rate"]          = savings / income
    features["emi_ratio"]             = emi / income
    features["rent_ratio"]            = rent / income
    features["food_ratio"]            = food / income
    features["need_ratio"]            = (rent + food + transport) / income
    features["want_ratio"]            = subscriptions / income
    features["total_expense_ratio"]   = (rent + food + emi + transport + subscriptions) / income
    features["disposable_ratio"]      = 1 - features["total_expense_ratio"]
    features["emergency_fund_months"] = emergency_fund
    features["dependents"]            = dependents

    return features
