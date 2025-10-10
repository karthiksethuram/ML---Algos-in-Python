import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def match_control_to_target(df, covariates, group_col='group', member_id_col='member_id', 
                            k_neighbors=1, caliper=None):
    """
    Matches control members (group=0) to nearest target members (group=1) 
    based on propensity scores, including categorical covariates.

    Parameters:
        df: Input DataFrame
        covariates: List of covariate column names (can include categorical variables)
        group_col: Column indicating treatment (1=target, 0=control)
        member_id_col: Unique member ID column
        k_neighbors: Number of target matches per control
        caliper: Optional max allowable PS distance (e.g., 0.05)
    """
    # Drop missing values
    df = df.dropna(subset=covariates + [group_col]).copy()
    
    # --- Encode categorical covariates ---
    df_encoded = pd.get_dummies(df[covariates], drop_first=True)
    
    # --- Standardize continuous covariates ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # --- Fit logistic regression to compute propensity scores ---
    y = df[group_col]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    df['pscore'] = model.predict_proba(X_scaled)[:, 1]
    
    # --- Split into groups ---
    target = df[df[group_col] == 1].copy()
    control = df[df[group_col] == 0].copy()
    
    # --- Match control → target ---
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(target[['pscore']])
    distances, indices = nbrs.kneighbors(control[['pscore']])
    
    matched_pairs = []
    for i, control_idx in enumerate(control.index):
        for k in range(k_neighbors):
            if caliper and abs(control.loc[control_idx, 'pscore'] - target.iloc[indices[i, k]]['pscore']) > caliper:
                continue
            matched_pairs.append((control_idx, target.index[indices[i, k]]))
    
    # --- Create matched dataset ---
    matched_df = []
    for c_idx, t_idx in matched_pairs:
        matched_df.append(df.loc[c_idx])
        matched_df.append(df.loc[t_idx])
    
    df_matched = pd.DataFrame(matched_df).drop_duplicates(subset=member_id_col)
    
    print(f"\nMatched sample size: {df_matched.shape[0]} (unique members: {df_matched[member_id_col].nunique()})")
    print(df_matched[group_col].value_counts())
    
    # --- Plot pre/post matching propensity overlap ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(data=df, x='pscore', hue=group_col, common_norm=False, ax=ax[0])
    ax[0].set_title('Before Matching')
    
    sns.kdeplot(data=df_matched, x='pscore', hue=group_col, common_norm=False, ax=ax[1])
    ax[1].set_title('After Matching')
    plt.show()
    
    return df_matched
covariates = ['age', 'pdc', 'diabetic_fills', 'member_status']  # includes categorical variable
df_matched = match_control_to_target(
    df, covariates, group_col='group', member_id_col='member_id', 
    k_neighbors=1, caliper=0.05  # optional caliper
)
