WITH RECURSIVE recursive_calc AS (

  ----------------------------------------------------------------
  -- ANCHOR: earliest fill per (lv1_acct_id, eph_id, drug_cls, fill_year)
  -- Aggregate same-day fills by taking MAX(day_sply_qty)
  ----------------------------------------------------------------
  SELECT
    a.lv1_acct_id,
    a.eph_id,
    a.drug_cls,
    ANY_VALUE(a.drug_name) AS drug_name,
    a.fill_year,
    a.fill_dt,
    MAX(a.day_sply_qty) AS day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY a.lv1_acct_id, a.eph_id, a.drug_cls, a.fill_year
      ORDER BY a.fill_dt
    ) AS rn,
    -- rolling coverage end after this fill
    DATE_ADD(a.fill_dt, INTERVAL MAX(a.day_sply_qty) DAY) AS rolling_end,
    0 AS excess_supply,           -- first fill → no excess
    0 AS cumulative_excess        -- first fill → cumulative starts at 0
  FROM `your_dataset.your_table_name` a
  GROUP BY a.lv1_acct_id, a.eph_id, a.drug_cls, a.fill_year, a.fill_dt
  QUALIFY rn = 1

  UNION ALL

  ----------------------------------------------------------------
  -- RECURSIVE STEP: process next fill per drug class
  ----------------------------------------------------------------
  SELECT
    b.lv1_acct_id,
    b.eph_id,
    b.drug_cls,
    b.drug_name,
    b.fill_year,
    b.fill_dt,
    b.day_sply_qty,
    b.rn,

    -- excess = new coverage beyond previous rolling_end
    CASE
      WHEN b.fill_dt > p.rolling_end THEN 0
      ELSE GREATEST(
        DATE_DIFF(DATE_ADD(b.fill_dt, INTERVAL b.day_sply_qty DAY), p.rolling_end, DAY),
        0
      )
    END AS excess_supply,

    -- update rolling_end: max of previous rolling_end vs current fill end
    CASE
      WHEN b.fill_dt > p.rolling_end THEN DATE_ADD(b.fill_dt, INTERVAL b.day_sply_qty DAY)
      ELSE GREATEST(p.rolling_end, DATE_ADD(b.fill_dt, INTERVAL b.day_sply_qty DAY))
    END AS rolling_end,

    -- cumulative_excess: add current excess to previous cumulative; reset if gap
    CASE
      WHEN b.fill_dt > p.rolling_end THEN 0
      ELSE p.cumulative_excess + GREATEST(
             DATE_DIFF(DATE_ADD(b.fill_dt, INTERVAL b.day_sply_qty DAY), p.rolling_end, DAY),
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
      MAX(day_sply_qty) AS day_sply_qty,
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

----------------------------------------------------------------
-- FINAL OUTPUT
----------------------------------------------------------------
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
