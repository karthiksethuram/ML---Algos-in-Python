WITH aggregated AS (
  -- Pre-aggregate same-day fills at drug class level:
  -- use MAX(day_sply_qty) so same-day different drug_names in same class don't stack
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
),

WITH_RECURSIVE recursive_calc AS (
  ----------------------------------------------------------------
  -- Anchor: the first chronological aggregated fill per partition.
  -- running_excess is set to 0 (no prior excess before we observe fills).
  ----------------------------------------------------------------
  SELECT
    a.lvl1_acct_gid,
    a.eph_id,
    a.drug_cls,
    a.drug_name,
    a.fill_year,
    a.fill_dt,
    a.day_sply_qty,
    a.rn,
    0 AS days_between,
    0 AS running_excess   -- start at 0 (no carry from before)
  FROM aggregated a
  WHERE a.rn = 1

  UNION ALL

  ----------------------------------------------------------------
  -- Recursive step: consider next fill (b) after previous (p)
  -- days_between = days between fills
  -- new_running_excess = GREATEST(0, p.running_excess + p.day_sply_qty - days_between)
  -- (previous running_excess + previous fill supply - elapsed days), clamped to 0
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
  JOIN aggregated b
    ON b.lvl1_acct_gid = p.lvl1_acct_gid
   AND b.eph_id = p.eph_id
   AND b.drug_cls = p.drug_cls
   AND b.fill_year = p.fill_year
   AND b.rn = p.rn + 1
)

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
