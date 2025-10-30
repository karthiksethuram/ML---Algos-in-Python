-- BigQuery: Recursive CTE must be first
RECURSIVE excess_calc AS (
  -- Base case: first fill per (lvl11_acct_gid, member_id, drug_cls, fill_year)
  SELECT
    CAST(lvl11_acct_gid AS STRING) AS lvl11_acct_gid,
    CAST(member_id AS STRING) AS member_id,
    CAST(drug_cls AS STRING) AS drug_cls,
    EXTRACT(YEAR FROM DATE(fill_date)) AS fill_year,
    DATE(fill_date) AS fill_date,
    CAST(day_sply_qty AS INT64) AS day_sply_qty,
    DATE_ADD(DATE(fill_date), INTERVAL CAST(day_sply_qty AS INT64) DAY) AS med_end_date,
    1 AS rn,
    CAST(0 AS INT64) AS prev_excess,
    DATE_ADD(DATE(fill_date), INTERVAL CAST(day_sply_qty AS INT64) DAY) AS current_end_date
  FROM your_table t
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY lvl11_acct_gid, member_id, drug_cls, EXTRACT(YEAR FROM DATE
