import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================
# STEP 1: Estimate Propensity Scores
# ======================================

def estimate_propensity_scores(df, covariates, group_col='group'):
    X = df[covariates].copy()
    y = df[group_col]

    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    df['pscore'] = model.predict_proba(X_scaled)[:, 1]

    return df, model


# ======================================
# STEP 2: Restrict to Common Support
# ======================================

def restrict_common_support(df, group_col='group'):
    ps_treat = df[df[group_col] == 1]['pscore']
    ps_ctrl = df[df[group_col] == 0]['pscore']

    lower_bound = max(ps_treat.min(), ps_ctrl.min())
    upper_bound = min(ps_treat.max(), ps_ctrl.max())

    df_overlap = df[(df['pscore'] >= lower_bound) & (df['pscore'] <= upper_bound)].copy()
    print(f"Restricted to common support region: {df_overlap.shape[0]} members retained")
    return df_overlap


# ======================================
# STEP 3: Match Treated to Control (No Replacement)
# ======================================

def match_treated_to_controls(df, group_col='group', caliper=0.05, ratio=1):
    treated = df[df[group_col] == 1].copy()
    control = df[df[group_col] == 0].copy()

    treated_scores = treated['pscore'].values.reshape(-1, 1)
    control_scores = control['pscore'].values.reshape(-1, 1)

    distances = pairwise_distances(treated_scores, control_scores)
    matched_pairs = []
    used_ctrl = set()

    for i, dist_row in enumerate(distances):
        sorted_idx = np.argsort(dist_row)
        selected = []

        for idx in sorted_idx:
            if dist_row[idx] <= caliper and idx not in used_ctrl:
                selected.append(idx)
                used_ctrl.add(idx)
                if len(selected) >= ratio:
                    break
        for ctrl_idx in selected:
            matched_pairs.append((treated.index[i], control.index[ctrl_idx]))

    # Construct matched dataset
    matched_rows = []
    for t_idx, c_idx in matched_pairs:
        matched_rows.append(df.loc[t_idx])
        matched_rows.append(df.loc[c_idx])

    matched_df = pd.DataFrame(matched_rows).drop_duplicates(subset=['member_id']).reset_index(drop=True)

    return matched_df


# ======================================
# STEP 4: Evaluate Covariate Balance
# ======================================

def plot_balance(df_before, df_after, covariates):
    def smd(df, cov):
        x1 = df[df['group'] == 1][cov]
        x0 = df[df['group'] == 0][cov]
        return (x1.mean() - x0.mean()) / np.sqrt(0.5 * (x1.var() + x0.var()))

    before_smd = [smd(df_before, cov) for cov in covariates]
    after_smd = [smd(df_after, cov) for cov in covariates]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=before_smd, y=after_smd)
    plt.axvline(0.1, color='red', linestyle='--', label='|SMD|=0.1')
    plt.axhline(0.1, color='red', linestyle='--')
    plt.xlabel('Before Matching SMD')
    plt.ylabel('After Matching SMD')
    plt.title('Covariate Balance: Before vs After Matching')
    plt.legend()
    plt.show()

    balance_df = pd.DataFrame({'Covariate': covariates,
                               'Before_SMD': before_smd,
                               'After_SMD': after_smd})
    print(balance_df.round(3))


# ======================================
# STEP 5: Run the Pipeline
# ======================================

# Example covariates — replace these with yours
covariates = ['new_vs_existing', 'pdc', 'age', 'diabetic_fills']

# 1️⃣ Estimate propensity scores
df, ps_model = estimate_propensity_scores(df, covariates)

# 2️⃣ Restrict to overlapping region
df_overlap = restrict_common_support(df)

# 3️⃣ Match treated to a subset of controls (1:1)
df_matched = match_treated_to_controls(df_overlap, caliper=0.05, ratio=1)

# 4️⃣ Check results
print("\nUnique member counts BEFORE matching:")
print(df.groupby('group')['member_id'].nunique())

print("\nUnique member counts AFTER matching:")
print(df_matched.groupby('group')['member_id'].nunique())

# 5️⃣ Check balance
plot_balance(df, df_matched, covariates)

print(f"\nFinal matched dataset: {df_matched.shape[0]} rows, {df_matched['member_id'].nunique()} unique members")
