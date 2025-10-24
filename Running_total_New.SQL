-- Replace `your_table` with your table name
WITH RECURSIVE
base AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    DATE(fill_dt) AS fill_dt,
    day_sply_qty,
    LAG(DATE(fill_dt)) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY DATE(fill_dt)
    ) AS prev_fill_dt
  FROM `your_project.your_dataset.your_table`
),

gap_calc AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    day_sply_qty,
    prev_fill_dt,
    CASE
      WHEN prev_fill_dt IS NULL THEN NULL
      ELSE DATE_DIFF(fill_dt, prev_fill_dt, DAY)
    END AS gap_days
  FROM base
),

-- Aggregate multiple drug_name fills occurring on the same date (same cls)
agg_by_fill AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    SUM(day_sply_qty) AS total_day_sply_qty,
    MIN(gap_days) AS gap_days  -- gap_days is the same for rows with same fill_dt, MIN is safe
  FROM gap_calc
  GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt
),

ordered_fills AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt
    ) AS rn,
    COUNT(*) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
    ) AS cnt_fills
  FROM agg_by_fill
),

-- Recursive CTE to compute running_excess
running_excess AS (
  -- anchor: first fill (rn = 1)
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    total_day_sply_qty,
    gap_days,
    rn,
    cnt_fills,
    total_day_sply_qty AS running_excess
  FROM ordered_fills
  WHERE rn = 1

  UNION ALL

  -- recursive step: next fill (rn = prev.rn + 1)
  SELECT
    nxt.lvl1_acct_gid,
    nxt.eph_id,
    nxt.drug_cls,
    nxt.fill_year,
    nxt.fill_dt,
    nxt.total_day_sply_qty,
    nxt.gap_days,
    nxt.rn,
    nxt.cnt_fills,
    -- compute new running_excess and clamp at 0
    GREATEST(0,
      prev.running_excess
      - COALESCE(nxt.gap_days, 0)   -- days lost to gap since previous fill
      + nxt.total_day_sply_qty      -- days from this fill
    ) AS running_excess
  FROM running_excess AS prev
  JOIN ordered_fills AS nxt
    ON prev.lvl1_acct_gid = nxt.lvl1_acct_gid
   AND prev.eph_id = nxt.eph_id
   AND prev.drug_cls = nxt.drug_cls
   AND prev.fill_year = nxt.fill_year
   AND nxt.rn = prev.rn + 1
)

-- Final outputs:
-- 1) detailed running_excess per fill row
-- 2) current_excess per lvl1_acct_gid,eph_id,drug_cls,fill_year (latest fill)
SELECT
  re.lvl1_acct_gid,
  re.eph_id,
  re.drug_cls,
  re.fill_year,
  re.fill_dt,
  re.total_day_sply_qty,
  re.gap_days,
  re.running_excess
FROM running_excess AS re
ORDER BY re.lvl1_acct_gid, re.eph_id, re.drug_cls, re.fill_dt;

-- If you want the current (latest) excess per group, run:
-- SELECT lvl1_acct_gid, eph_id, drug_cls, fill_year, ARRAY_REVERSE_AGG(STRUCT(fill_dt, running_excess))[OFFSET(0)].*
-- FROM running_excess
-- GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year;
