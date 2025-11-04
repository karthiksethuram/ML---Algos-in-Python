import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Example: create prescriber_target_share
df['prescriber_target_share'] = df.groupby('prescriber_id')['is_target'].transform('mean')

# 1A: Linear mixed model (LMM) - if eds_conv is continuous or when using approximations
# Formula: eds_conv ~ is_target + prescriber_target_share + is_target:prescriber_target_share + is_new_member + ...
formula = "eds_conv ~ is_target + prescriber_target_share + is_target:prescriber_target_share + is_new_member + age + baseline_adherence"
# MixedLM needs exog and groups
md = smf.mixedlm(formula, data=df, groups=df["prescriber_id"])
mdf = md.fit(reml=False)
print(mdf.summary())

# 1B: GEE with logit link for binary outcome with exchangeable covariance within prescriber clusters
import statsmodels.genmod.generalized_estimating_equations as gee_mod
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

family = Binomial()
cov_struct = Exchangeable()  # assumes exchangeable correlation within prescriber
gee = smf.gee("eds_conv ~ is_target + prescriber_target_share + is_target:prescriber_target_share + is_new_member + age + baseline_adherence",
              groups="prescriber_id",
              data=df,
              cov_struct=cov_struct,
              family=family)
gee_res = gee.fit()
print(gee_res.summary())






import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

# Create prescriber-level summarised table
pres = df.groupby('prescriber_id').agg(
    n_patients=('member_id','nunique'),
    n_conv=('eds_conv','sum'),
    prescriber_target_share=('prescriber_target_share','first'),  # same within group
    avg_age=('age','mean'),
    pct_new=('is_new_member','mean')
).reset_index()

pres['conv_rate'] = pres['n_conv'] / pres['n_patients']

# Simple OLS
X = pres[['prescriber_target_share','avg_age','pct_new','n_patients']]
X = sm.add_constant(X)
y = pres['conv_rate']

ols = sm.OLS(y, X).fit(cov_type='HC3')  # robust HC3
print(ols.summary())

# If you want cluster-robust SEs (e.g., clustering on region), use cov_cluster:
# clustered = ols.get_robustcov_results(cov_type='cluster', groups=pres['region'])




