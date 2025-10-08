import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# STEP 1: Estimate Propensity Scores
# ==============================

def estimate_propensity_scores(df, covariates, group_col='group'):
    X = df[covariates].copy()
    y = df[group_col]

    # Handle missing values
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    df['pscore'] = model.predict_proba(X_scaled)[:, 1]

    return df, model


# ==============================
# STEP 2: Propensity Score Matching (No Replacement)
# ==============================

def match_members(df, group_col='group', caliper=0.05, ratio=1):
    treated = df[df[group_col] == 1].copy()
    control = df[df[group_col] == 0].copy()

    # Convert to arrays for faster distance computation
    treated_scores = treated['pscore'].values.reshape(-1, 1)
    control_scores = control['pscore'].values.reshape(-1, 1)

    # Compute absolute distance matrix
    distances = pairwise_distances(treated_scores, control_scores, metric='euclidean')

    matched_indices = []
    used_controls = set()

    for i, row_dists in enumerate(distances):
        # Sort control indices by closeness in propensity score
        sorted_idx = np.argsort(row_dists)
        selected = []

        for idx in sorted_idx:
            if row_dists[idx] <= caliper and idx not in used_controls:
                selected.append(idx)
                used_controls.add(idx)
                if len(selected) >= ratio:
                    break
        if selected:
            for ctrl_idx in selected:
                matched_indices.append((treated.index[i], control.index[ctrl_idx]))

    # Create matched DataFrame
    matched_pairs = []
    for t_idx, c_idx in matched_indices:
        matched_pairs.append(df.loc[t_idx])
        matched_pairs.append(df.loc[c_idx])

    matched_df = pd.DataFrame(matched_pairs).reset_index(drop=True)

    # Remove any duplicates that slipped through
    matched_df = matched_df.drop_duplicates(subset=['member_id']).reset_index(drop=True)

    return matched_df


# ==============================
# STEP 3: Evaluate Balance (Before/After)
# ==============================

def plot_balance(df_before, df_after, covariates):
    def smd(df, covariate):
        x1 = df[df['group'] == 1][covariate]
        x0 = df[df['group'] == 0][covariate]
        return (x1.mean() - x0.mean()) / np.sqrt(0.5 * (x1.var() + x0.var()))

    before_smd = [smd(df_before, cov) for cov in covariates]
    after_smd = [smd(df_after, cov) for cov in covariates]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=before_smd, y=after_smd)
    plt.axvline(0.1, color='red', linestyle='--', label='|SMD|=0.1 threshold')
    plt.axhline(0.1, color='red', linestyle='--')
    plt.xlabel('Before Matching SMD')
    plt.ylabel('After Matching SMD')
    plt.title('Covariate Balance Before vs After Matching')
    plt.legend()
    plt.show()

    balance_df = pd.DataFrame({
        'Covariate': covariates,
        'Before_SMD': before_smd,
        'After_SMD': after_smd
    })
    print(balance_df)


# ==============================
# STEP 4: Run the Matching Workflow
# ==============================

# Example covariates (replace with your actual ones)
covariates = ['new_vs_existing', 'pdc', 'age', 'diabetic_fills']

# 1. Estimate Propensity Scores
df, ps_model = estimate_propensity_scores(df, covariates)

# 2. Perform Matching (1:1 example, caliper 0.05)
df_matched = match_members(df, caliper=0.05, ratio=1)

# 3. Compare counts
print("Unique member counts BEFORE matching:")
print(df.groupby('group')['member_id'].nunique())

print("\nUnique member counts AFTER matching:")
print(df_matched.groupby('group')['member_id'].nunique())

# 4. Plot balance improvement
plot_balance(df, df_matched, covariates)

# 5. Quick sanity check
print("\nTotal rows after matching:", df_matched.shape[0])
print("Unique members after matching:", df_matched['member_id'].nunique())
