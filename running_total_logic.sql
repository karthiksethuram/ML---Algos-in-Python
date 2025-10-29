WITH RECURSIVE recursive_calc AS (

  -- ANCHOR: earliest fill per lv1_acct_id, eph_id, drug_cls, fill_year
  SELECT
    a.lv1_acct_id,
    a.eph_id,
    a.drug_cls,
    ANY_VALUE(a.drug_name) AS drug_name,
    a.fill_year,
    a.fill_dt,
    CAST(MAX(a.day_sply_qty) AS INT64) AS day_sply_qty,
    CAST(1 AS INT64) AS rn,
    DATE_ADD(a.fill_dt, INTERVAL CAST(MAX(a.day_sply_qty) AS INT64) DAY) AS rolling_end,
    CAST(0 AS INT64) AS excess_supply,
    CAST(0 AS INT64) AS cumulative_excess
  FROM `your_dataset.your_table_name` a
  GROUP BY a.lv1_acct_id, a.eph_id, a.drug_cls, a.fill_year, a.fill_dt
  QUALIFY rn = 1

  UNION ALL

  -- RECURSIVE STEP
  SELECT
    b.lv1_acct_id,
    b.eph_id,
    b.drug_cls,
    b.drug_name,
    b.fill_year,
    b.fill_dt,
    CAST(b.day_sply_qty AS INT64) AS day_sply_qty,
    CAST(b.rn AS INT64) AS rn,

    -- excess_supply
    CAST(
      CASE
        WHEN b.fill_dt > p.rolling_end THEN 0
        ELSE GREATEST(
          DATE_DIFF_
