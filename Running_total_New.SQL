WITH RECURSIVE recursive_calc AS (

  ----------------------------------------------------------------
  -- Anchor part:
  -- For each (lvl1_acct_gid, eph_id, drug_cls, fill_year),
  -- find the earliest fill date.
  -- Aggregate same-day same-drug-class fills using MAX(day_sply_qty)
  ----------------------------------------------------------------
  SELECT
    a.lvl1_acct_gid,
    a.eph_id,
    a.drug_cls,
    ANY_VALUE(a.drug_name) AS drug_name,
    a.fill_year,
    a.fill_dt,
    MAX(a.day_sply_qty) AS day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY a.lvl1_acct_gid, a.eph_id, a.drug_cls, a.fill_year
      ORDER BY a.fill_dt
    ) AS rn,
    0 AS days_between,
    0 AS running_excess  -- start with zero (no carry before first fill)
  FROM `your_dataset.your_table_name` a
  GROUP BY a.lvl1_acct_gid, a.eph_id, a.drug_cls, a.fill_year, a.fill_dt
  QUALIFY rn = 1


  UNION ALL

  ----------------------------------------------------------------
  -- Recursive part:
  -- For the next fill (rn = prev + 1):
  --   days_between = gap between fills
  --   running_excess = GREATEST(0, prev.running_excess + prev.day_sply_qty - days_between)
  -- This formula correctly resets to zero when negative.
  ----------------------------------------------------------------
  SELECT
    b.lvl1_acct_gid,
    b.eph_id,
    b.drug_cls,
    b.drug_name,
    b.fill_year,
    b.fill_dt,
    b.day_sply_qty,
    b.rn,
    DATE_DIFF(b.fill_dt, p.fill_dt, DAY) AS days_between,
    GREATEST(
      0,
      p.running_excess + p.day_sply_qty - DATE_DIFF(b.fill_dt, p.fill_dt, DAY)
    ) AS running_excess
  FROM recursive_calc p
  JOIN (
    SELECT
      lvl1_acct_gid,
      eph_id,
      drug_cls,
      ANY_VALUE(drug_name) AS drug_name,
      fill_year,
      fill_dt,
      MAX(day_sply_qty) AS day_sply_qty,
      ROW_NUMBER() OVER (
        PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
        ORDER BY fill_dt
      ) AS rn
    FROM `your_dataset.your_table_name`
    GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt
  ) AS b
  ON b.lvl1_acct_gid = p.lvl1_acct_gid
  AND b.eph_id = p.eph_id
  AND b.drug_cls = p.drug_cls
  AND b.fill_year = p.fill_year
  AND b.rn = p.rn + 1
)

----------------------------------------------------------------
-- Final output: running_excess after each fill
----------------------------------------------------------------
SELECT
  lvl1_acct_gid,
  eph_id,
  drug_cls,
  drug_name,
  fill_year,
  fill_dt,
  day_sply_qty,
  days_between,
  running_excess
FROM recursive_calc
ORDER BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt;
