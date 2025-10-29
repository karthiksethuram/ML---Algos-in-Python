# --- Install packages (if not already) ---
# !pip install rpy2
# In R: install.packages("interference")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Import R packages
ro.r('library(interference)')

# Pass your pandas DataFrame to R
r_df = pandas2ri.py2rpy(df)

# Define the R command
ro.r.assign("r_df", r_df)
ro.r('''
results <- interference(
  formula = EDS_conversion_flag_150d ~ star_pdc_py_imputed + is_new_member + factor(star_drug_cls),
  treat = is_target_member,
  group = npi_id,
  allocation = "observed",
  model_method = "logit",
  effect = "both",
  data = r_df
)
print(summary(results))
''')

# 1) Install & load
install.packages("inferference")   # from CRAN
library(inferference)

# 2) Prepare your data
# Assume your dataset is named df with columns:
#   npi_id (cluster), is_target_member (0/1), EDS_conversion_flag_150d (0/1),
#   star_pdc_py_imputed, is_new_member (0/1), star_drug_cls (categorical)
df$is_target_member <- as.integer(df$target_or_control == "Target")
df$is_new_member <- as.integer(df$member_status == "New")
df$star_drug_cls <- factor(df$star_drug_cls)

# 3) Run interference model
# The formula syntax for inferference is: outcome | exposure ~ propensity covariates | group
res <- interference(
  formula = EDS_conversion_flag_150d | is_target_member ~
              star_pdc_py_imputed + is_new_member + star_drug_cls | npi_id,
  allocations = c(0, 0.5, 1.0),
  data = df,
  model_method = "glm",
  model_options = list(family = binomial(link = "logit")),
  causal_estimation_method = "ipw",
  causal_estimation_options = list(variance_estimation = "robust")
)

# 4) View results
print(res)
summary(res)

# 5) Retrieve effects
direct_eff  <- direct_effect(res, allocation = c(0,0.5,1.0))
indirect_eff <- indirect_effect(res, allocation1 = c(0,0.5,1.0))
total_eff   <- total_effect(res, allocation1 = c(0,0.5,1.0))
overall_eff <- overall_effect(res, allocation1 = c(0,0.5,1.0))

# 6) Display effect tables
print(direct_eff)
print(indirect_eff)
print(total_eff)
print(overall_eff)

