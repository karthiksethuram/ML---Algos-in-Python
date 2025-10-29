-- Make sure RECURSIVE CTE is first (BigQuery requirement)
WITH RECURSIVE recursive_calc AS (

  ----------------------------------------------------------------
  -- Anchor: first fill per member + drug class + year (rn = 1)
  -- NOTE: This references the 'ordered' CTE which is defined later.
  ----------------------------------------------------------------
  SELECT
    o.lvl1_acct_gid,
    o.eph_id,
    o.drug_cls,
    o.fill_year,
    o.fill_dt,
    o.med_end_dt,
    o.day_sply_qty,                 -- INT64
    o.drug_names,
    CAST(0 AS INT64) AS days_between,
    CAST(0 AS INT64) AS excess_days, -- INT64 to avoid casting issues
    o.med_end_dt AS running_end_dt,
    o.rn
  FROM ordered o
  WHERE o.rn = 1

  UNION ALL

  ----------------------------------------------------------------
  -- Recursive step: take the next rn = p.rn + 1 for same (acct, eph, class, year)
  ----------------------------------------------------------------
  SELECT
    b.lvl1_acct_gid,
    b.eph_id,
    b.drug_cls,
    b.fill_year,
    b.fill_dt,
    b.med_end_dt,
    b.day_sply_qty,
    b.drug_names,
    CAST(DATE_DIFF(b.fill_dt, p.fill_dt, DAY) AS INT64) AS days_between,
    -- excess_days: only the true overlap (previous running_end_dt - new fill date)
    CASE
      WHEN b.fill_dt <= p.running_end_dt
      THEN GREATEST(CAST(0 AS INT64), CAST(DATE_DIFF(p.running_end_dt, b.fill_dt, DAY) AS INT64))
      ELSE CAST(0 AS INT64)
    END AS excess_days,
    -- extend running_end_dt to the later of prior or current med end
    CASE
      WHEN p.running_end_dt > b.med_end_dt THEN p.running_end_dt
      ELSE b.med_end_dt
    END AS running_end_dt,
    b.rn
  FROM recursive_calc p
  JOIN ordered b
    ON b.lvl1_acct_gid = p.lvl1_acct_gid
   AND b.eph_id = p.eph_id
   AND b.drug_cls = p.drug_cls
   AND b.fill_year = p.fill_year
   AND b.rn = p.rn + 1

),

/*--------------------------------------------------------------------
  Non-recursive helper CTEs (defined after recursive_calc per BigQuery
  requirement that RECURSIVE be first). These prepare class-level fills
  by taking the MAX end date among same-day same-class fills.
--------------------------------------------------------------------*/
base AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    -- ensure day_sply_qty is INT64
    CAST(MAX(CAST(day_sply_qty AS INT64)) AS INT64) AS day_sply_qty,
    -- med_end_dt: class-level latest end date for that fill_dt
    DATE_ADD(
      fill_dt,
      INTERVAL CAST(MAX(CAST(day_sply_qty AS INT64)) - 1 AS INT64) DAY
    ) AS med_end_dt,
    ARRAY_AGG(DISTINCT drug_name IGNORE NULLS) AS drug_names
  FROM `your_dataset.your_table_name`
  GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt
),

ordered AS (
  SELECT
    b.*,
    ROW_NUMBER() OVER (
      PARTITION BY b.lvl1_acct_gid, b.eph_id, b.drug_cls, b.fill_year
      ORDER BY b.fill_dt
    ) AS rn
  FROM base b
)

----------------------------------------------------------------
-- Final output: running excess / overlap per class-level fill
----------------------------------------------------------------
SELECT
  rc.lvl1_acct_gid,
  rc.eph_id,
  rc.drug_cls,
  rc.fill_year,
  rc.fill_dt,
  rc.day_sply_qty,
  FORMAT_DATE('%Y-%m-%d', rc.med_end_dt)        AS med_end_dt,
  rc.days_between,
  rc.excess_days,
  FORMAT_DATE('%Y-%m-%d', rc.running_end_dt)    AS running_end_dt,
  ARRAY_TO_STRING(rc.drug_names, ', ')          AS drugs_in_class,
  rc.rn
FROM recursive_calc rc
ORDER BY rc.lvl1_acct_gid, rc.eph_id, rc.drug_cls, rc.fill_year, rc.fill_dt;
