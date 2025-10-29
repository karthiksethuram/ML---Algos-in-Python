# --- 1. Imports ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr

# --- 2. Basic cleaning ---
df = df.copy()

# Convert variables
df['is_target_member'] = (df['target_or_control'] == 'Target').astype(int)
df['is_new_member'] = (df['member_status'] == 'New').astype(int)
df['star_pdc_py_imputed'] = df['star_pdc_py'].fillna(df['star_pdc_py'].mean())

# Make sure your outcome is binary numeric
df['EDS_conversion_flag_150d'] = df['EDS_conversion_flag_150d'].astype(int)

# --- 3. Pairwise plot (only for continuous variables) ---
sns.pairplot(
    df,
    vars=['presc_target_share', 'star_pdc_py_imputed'],
    hue='EDS_conversion_flag_150d',
    palette='coolwarm',
    plot_kws={'alpha': 0.5}
)
plt.suptitle('Pairwise Plot: Continuous Predictors vs. EDS Conversion', y=1.02)
plt.show()

# --- 4. Boxplots / violin plots for categorical vs. binary outcome ---
categorical_vars = ['is_target_member', 'is_new_member', 'star_drug_cls']

for var in categorical_vars:
    plt.figure(figsize=(6,4))
    sns.barplot(x=var, y='EDS_conversion_flag_150d', data=df, ci=95)
    plt.title(f"Mean EDS Conversion Rate by {var}")
    plt.ylabel("EDS Conversion Probability")
    plt.grid(alpha=0.3)
    plt.show()

# --- 5. Check linearity of continuous vars in logit scale ---
def plot_logit_linearity(var):
    """
    For each continuous variable, bin it and plot mean outcome vs. mean predictor.
    A straight line in logit space means linearity holds for logistic regression.
    """
    temp = df[[var, 'EDS_conversion_flag_150d']].copy()
    temp['bin'] = pd.qcut(temp[var], 10, duplicates='drop')  # 10 quantile bins
    grouped = temp.groupby('bin').agg(
        mean_x=(var, 'mean'),
        mean_y=('EDS_conversion_flag_150d', 'mean'),
        count=('EDS_conversion_flag_150d', 'size')
    ).reset_index()

    grouped['logit_y'] = np.log(grouped['mean_y'] / (1 - grouped['mean_y'] + 1e-5))  # avoid divide by zero

    fig, ax1 = plt.subplots(figsize=(7,4))
    sns.scatterplot(x='mean_x', y='logit_y', data=grouped, ax=ax1, s=70)
    sns.lineplot(x='mean_x', y='logit_y', data=grouped, ax=ax1)
    ax1.set_title(f'Linearity Check (Logit of EDS Conversion) vs {var}')
    ax1.set_xlabel(var)
    ax1.set_ylabel('Logit of Mean Conversion Probability')
    plt.grid(alpha=0.3)
    plt.show()
    
    corr, _ = spearmanr(grouped['mean_x'], grouped['logit_y'])
    print(f"Spearman correlation (logit linearity check) for {var}: {corr:.3f}")

for var in ['presc_target_share', 'star_pdc_py_imputed']:
    plot_logit_linearity(var)

# --- 6. Interaction exploration (optional) ---
plt.figure(figsize=(7,5))
sns.lineplot(
    x='presc_target_share',
    y='EDS_conversion_flag_150d',
    hue='is_target_member',
    data=df,
    estimator='mean'
)
plt.title("Conditional Mean of EDS Conversion vs Prescriber Target Share (by Target/Control)")
plt.xlabel("Prescriber Target Share")
plt.ylabel("EDS Conversion Rate")
plt.grid(alpha=0.3)
plt.show()

# --- 7. Quick correlation summary ---
corrs = df[['EDS_conversion_flag_150d', 'presc_target_share', 'star_pdc_py_imputed', 'is_target_member', 'is_new_member']].corr(method='pearson')
print("\nCorrelation matrix:\n", corrs.round(3))

