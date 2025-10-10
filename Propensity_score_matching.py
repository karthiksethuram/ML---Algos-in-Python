import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# =========================================
# STEP 1️⃣: Prepare and clean data
# =========================================
df = df.copy()

# prescriber_group: 'target_control_group' (treated) vs 'control_only_group' (control)
df['treatment'] = (df['prescriber_group'] == 'target_control_group').astype(int)

# Handle missing categorical variable (MEMBER_STATUS)
df['MEMBER_STATUS'] = df['MEMBER_STATUS'].fillna(df['MEMBER_STATUS'].mode()[0])

# One-hot encode MEMBER_STATUS (New/Returning)
df_encoded = pd.get_dummies(df, columns=['MEMBER_STATUS'], drop_first=True)

# Covariates for matching
covariates = ['MBR_AGE_FILLD', 'diab_fills_PY', 'statin_fills_PY', 'MEMBER_STATUS_Returning']

# Handle missing numeric values (impute with median)
for var in ['MBR_AGE_FILLD', 'diab_fills_PY', 'statin_fills_PY']:
    df_encoded[var] = df_encoded[var].fillna(df_encoded[var].median())

# =========================================
# STEP 2️⃣: Estimate propensity scores
# =========================================
model = LogisticRegression(max_iter=1000)
model.fit(df_encoded[covariates], df_encoded['treatment'])
df['propensity_score'] = model.predict_proba(df_encoded[covariates])[:, 1]
from sklearn.metrics import pairwise_distances

# =========================================
# STEP 3️⃣: True 1:1 Nearest Neighbor Matching (without replacement)
# =========================================
treated = df[df['treatment'] == 1].copy()
control = df[df['treatment'] == 0].copy()

treated = treated.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle to avoid bias
control_available = control.copy()

matches = []
caliper = 0.05  # optional: max allowable propensity score distance

for i, row in treated.iterrows():
    # compute absolute distance between this treated and all available controls
    control_available['distance'] = abs(control_available['propensity_score'] - row['propensity_score'])
    nearest = control_available.loc[control_available['distance'].idxmin()]

    # apply caliper constraint (optional but recommended)
    if nearest['distance'] <= caliper:
        matches.append((row.name, nearest.name))
        control_available = control_available.drop(nearest.name)
    
    # stop if controls exhausted
    if control_available.empty:
        break

# get matched data
treated_matched = treated.loc[[m[0] for m in matches]].copy()
control_matched = control.loc[[m[1] for m in matches]].copy()

treated_matched['match_id'] = range(len(matches))
control_matched['match_id'] = range(len(matches))

df_matched = pd.concat([treated_matched, control_matched], axis=0)
print(f"Matched {len(matches)} treated-control pairs.")

# =========================================
# STEP 4️⃣: Define balance checking function
# =========================================
def balance_table(df, group_col, covariates):
    rows = []
    for var in covariates:
        g1 = df[df[group_col] == 1][var].dropna()
        g0 = df[df[group_col] == 0][var].dropna()

        # Continuous vars: t-test + SMD
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        smd = (g1.mean() - g0.mean()) / np.sqrt((g1.var() + g0.var()) / 2)
        rows.append([
            var, 
            round(g1.mean(), 3), 
            round(g0.mean(), 3), 
            round(smd, 3), 
            round(p_val, 4)
        ])

    table = pd.DataFrame(rows, columns=['Variable', 'Mean (Treated)', 'Mean (Control)', 'SMD', 'p-value'])
    return table

# =========================================
# STEP 5️⃣: Pre- and Post-Matching Balance Checks
# =========================================
print("=== ⚖️ Pre-Matching Balance Check ===")
pre_table = balance_table(df_encoded.assign(treatment=df['treatment']), 'treatment', covariates)
print(pre_table, "\n")

print("=== 🔄 Post-Matching Balance Check ===")
df_matched_encoded = pd.get_dummies(df_matched, columns=['MEMBER_STATUS'], drop_first=True)
for var in ['MBR_AGE_FILLD', 'diab_fills_PY', 'statin_fills_PY']:
    df_matched_encoded[var] = df_matched_encoded[var].fillna(df_matched_encoded[var].median())

post_table = balance_table(df_matched_encoded.assign(treatment=df_matched['treatment']), 'treatment', covariates)
print(post_table, "\n")

# =========================================
# STEP 6️⃣: Balance improvement summary
# =========================================
compare = pre_table[['Variable', 'SMD']].merge(
    post_table[['Variable', 'SMD']], on='Variable', suffixes=('_Pre', '_Post')
)
compare['|SMD| Reduction'] = abs(compare['SMD_Pre']) - abs(compare['SMD_Post'])
print("=== ✅ Balance Improvement Summary ===")
print(compare, "\n")

# =========================================
# STEP 7️⃣: Optional Paired t-tests (Post-Matching)
# =========================================
print("=== 🧪 Paired t-tests after matching (for continuous vars) ===")
for var in ['MBR_AGE_FILLD', 'diab_fills_PY', 'statin_fills_PY']:
    matched_pairs = df_matched.sort_values('match_id')
    treated_vals = matched_pairs[matched_pairs['treatment'] == 1][var].values
    control_vals = matched_pairs[matched_pairs['treatment'] == 0][var].values
    t_stat, p_val = stats.ttest_rel(treated_vals, control_vals)
    print(f"{var}: paired t = {t_stat:.3f}, p = {p_val:.4f}")
