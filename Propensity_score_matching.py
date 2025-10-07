import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# 1️⃣ Helper: Standardized Mean Difference & Balance Summary
# ================================================================
def smd(x_treat, x_ctrl):
    return (x_treat.mean() - x_ctrl.mean()) / np.sqrt((x_treat.var() + x_ctrl.var()) / 2)

def balance_table(df, covariates):
    treat = df[df['group'] == 1]
    ctrl = df[df['group'] == 0]
    results = []
    for cov in covariates:
        x_treat, x_ctrl = treat[cov].dropna(), ctrl[cov].dropna()
        smd_val = smd(x_treat, x_ctrl)
        if df[cov].nunique() > 2:
            t, p = stats.ttest_ind(x_treat, x_ctrl, equal_var=False)
        else:
            contingency = pd.crosstab(df['group'], df[cov])
            _, p, _, _ = stats.chi2_contingency(contingency)
        results.append({
            'Variable': cov,
            'SMD': round(smd_val, 3),
            'P-value': round(p, 4)
        })
    return pd.DataFrame(results)

# ================================================================
# 2️⃣ Estimate Propensity Scores
# ================================================================
def estimate_propensity(df, covariates):
    X = df[covariates]
    y = df['group']
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, y)
    df['pscore'] = ps_model.predict_proba(X)[:, 1]
    return df

# ================================================================
# 3️⃣ Matching Function (1-to-2, Caliper 0.05)
# ================================================================
def match_members(df, caliper=0.05, ratio=2):
    treated = df[df['group'] == 1].copy()
    control = df[df['group'] == 0].copy()
    matched_rows = []

    for idx, row in treated.iterrows():
        ctrl_pool = control[np.abs(control['pscore'] - row['pscore']) <= caliper]
        if len(ctrl_pool) < ratio:
            continue
        ctrl_pool = ctrl_pool.assign(dist=np.abs(ctrl_pool['pscore'] - row['pscore']))
        top_matches = ctrl_pool.nsmallest(ratio, 'dist')
        top_matches['match_id'] = idx
        matched_rows.append(top_matches)

    if matched_rows:
        matched_ctrl = pd.concat(matched_rows)
        matched_treat = treated.loc[matched_ctrl['match_id'].unique()]
        matched_df = pd.concat([matched_treat, matched_ctrl])
        return matched_df
    else:
        raise ValueError("No matches found. Try increasing caliper or using 1:1 matching.")

# ================================================================
# 4️⃣ Visualization Functions
# ================================================================
def plot_pscore_distribution(df, matched_df):
    plt.figure(figsize=(8,4))
    sns.kdeplot(data=df, x='pscore', hue='group', fill=True, alpha=0.3, common_norm=False)
    plt.title("Pre-Match Propensity Score Overlap")
    plt.show()

    plt.figure(figsize=(8,4))
    sns.kdeplot(data=matched_df, x='pscore', hue='group', fill=True, alpha=0.3, common_norm=False)
    plt.title("Post-Match Propensity Score Overlap")
    plt.show()

def plot_love_chart(pre, post):
    merged = pre.merge(post, on='Variable', suffixes=('_pre', '_post'))
    plt.figure(figsize=(6,4))
    plt.scatter(merged['SMD_pre'], merged['Variable'], label='Pre-Match', color='red')
    plt.scatter(merged['SMD_post'], merged['Variable'], label='Post-Match', color='blue')
    plt.axvline(x=0.1, color='grey', linestyle='--')
    plt.axvline(x=-0.1, color='grey', linestyle='--')
    plt.title('Standardized Mean Differences (Love Plot)')
    plt.xlabel('SMD')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ================================================================
# 5️⃣ Run Entire Pipeline
# ================================================================
def run_matching_pipeline(df, covariates, caliper=0.05, ratio=2):
    print("=== Step 1: Estimate Propensity Scores ===")
    df = estimate_propensity(df, covariates)

    print("=== Step 2: Pre-Match Balance ===")
    pre_balance = balance_table(df, covariates)
    print(pre_balance)

    print("=== Step 3: Perform Matching ===")
    matched_df = match_members(df, caliper=caliper, ratio=ratio)

    print("=== Step 4: Post-Match Balance ===")
    post_balance = balance_table(matched_df, covariates)
    print(post_balance)

    print("=== Step 5: Balance Improvement ===")
    balance_compare = pre_balance.merge(post_balance, on='Variable', suffixes=('_pre', '_post'))
    balance_compare['SMD_reduction'] = np.abs(balance_compare['SMD_pre']) - np.abs(balance_compare['SMD_post'])
    print(balance_compare[['Variable', 'SMD_pre', 'SMD_post', 'SMD_reduction']])

    print("=== Step 6: Plots ===")
    plot_pscore_distribution(df, matched_df)
    plot_love_chart(pre_balance, post_balance)

    return matched_df, balance_compare

# ================================================================
# 6️⃣ Example Run
# ================================================================
# covariates = ['age', 'baseline_pdc', 'new_vs_existing', 'diabetic_fills']
# matched_df, balance_compare = run_matching_pipeline(df, covariates)
