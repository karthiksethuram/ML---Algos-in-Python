WITH RECURSIVE recursive_calc AS (

  -- ANCHOR
  SELECT
    a.lv1_acct_id,
    a.eph_id,
    a.drug_cls,
    ANY_VALUE(a.drug_name) AS drug_name,
    a.fill_year,
    a.fill_dt,
    CAST(MAX(a.day_sply_qty) AS INT64) AS day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY a.lv1_acct_id, a.eph_id, a.drug_cls, a.fill_year
      ORDER BY a.fill_dt
    ) AS rn,
    -- rolling_end must be DATE
    DATE_ADD(a.fill_dt, INTERVAL CAST(MAX(a.day_sply_qty) AS INT64) DAY) AS rolling_end,
    0 AS excess_supply,
    0 AS cumulative_excess
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
    b.rn,

    -- excess = days beyond previous rolling_end
    CASE
      WHEN b.fill_dt > p.rolling_end THEN 0
      ELSE GREATEST(
        DATE_DIFF(DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY), p.rolling_end, DAY),
        0
      )
    END AS excess_supply,

    -- rolling_end: max of previous rolling_end vs current fill end, always DATE
    CASE
      WHEN b.fill_dt > p.rolling_end THEN DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY)
      ELSE (
        SELECT MAX(d) FROM UNNEST([p.rolling_end, DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY)]) AS d
      )
    END AS rolling_end,

    -- cumulative_excess: add current excess to previous cumulative, reset if gap
    CASE
      WHEN b.fill_dt > p.rolling_end THEN 0
      ELSE p.cumulative_excess + GREATEST(
             DATE_DIFF(DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY), p.rolling_end, DAY),
             0
           )
    END AS cumulative_excess

  FROM recursive_calc p
  JOIN (
    SELECT
      lv1_acct_id,
      eph_id,
      drug_cls,
      ANY_VALUE(drug_name) AS drug_name,
      fill_year,
      fill_dt,
      CAST(MAX(day_sply_qty) AS INT64) AS day_sply_qty,
      ROW_NUMBER() OVER (
        PARTITION BY lv1_acct_id, eph_id, drug_cls, fill_year
        ORDER BY fill_dt
      ) AS rn
    FROM `your_dataset.your_table_name`
    GROUP BY lv1_acct_id, eph_id, drug_cls, fill_year, fill_dt
  ) AS b
    ON b.lv1_acct_id = p.lv1_acct_id
   AND b.eph_id = p.eph_id
   AND b.drug_cls = p.drug_cls
   AND b.fill_year = p.fill_year
   AND b.rn = p.rn + 1
)

SELECT
  lv1_acct_id,
  eph_id,
  drug_cls,
  drug_name,
  fill_year,
  fill_dt,
  day_sply_qty,
  excess_supply,
  cumulative_excess,
  rolling_end AS coverage_end_after_fill
FROM recursive_calc
ORDER BY lv1_acct_id, eph_id, drug_cls, fill_year, fill_dt;
