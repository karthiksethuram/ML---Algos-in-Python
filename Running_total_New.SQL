-- Replace project.dataset.fills with your actual table
WITH RECURSIVE fill_base AS (
  -- Step 1: Add sequence number and lags per drug_name
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, drug_name, fill_year
      ORDER BY fill_dt
    ) AS rn,
    LAG(fill_dt) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, drug_name, fill_year
      ORDER BY fill_dt
    ) AS prev_fill_dt,
    LAG(day_sply_qty) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, drug_name, fill_year
      ORDER BY fill_dt
    ) AS prev_day_sply_qty
  FROM `project.dataset.fills`
  WHERE fill_dt >= '2024-01-01'
),

calc AS (
  -- Step 2: Compute overlap/delay in days
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    day_sply_qty,
    rn,
    CASE
      WHEN prev_fill_dt IS NULL THEN 0
      ELSE (prev_day_sply_qty - DATE_DIFF(fill_dt, prev_fill_dt, DAY))
    END AS net_excess_change
  FROM fill_base
),

recursive_running AS (
  -- Step 3: Recursive accumulation (true reset logic per drug_name)
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    rn,
    day_sply_qty,
    net_excess_change,
    GREATEST(net_excess_change, 0) AS running_excess
  FROM calc
  WHERE rn = 1

  UNION ALL

  SELECT
    c.lvl1_acct_gid,
    c.eph_id,
    c.drug_cls,
    c.drug_name,
    c.fill_year,
    c.fill_dt,
    c.rn,
    c.day_sply_qty,
    c.net_excess_change,
    GREATEST(r.running_excess + c.net_excess_change, 0) AS running_excess
  FROM calc c
  JOIN recursive_running r
    ON c.lvl1_acct_gid = r.lvl1_acct_gid
    AND c.eph_id = r.eph_id
    AND c.drug_cls = r.drug_cls
    AND c.drug_name = r.drug_name
    AND c.fill_year = r.fill_year
    AND c.rn = r.rn + 1
),

fill_intervals AS (
  -- Step 4: Convert each fill to start–end date ranges per drug_name
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt AS start_dt,
    DATE_ADD(fill_dt, INTERVAL day_sply_qty - 1 DAY) AS end_dt,
    day_sply_qty,
    running_excess
  FROM recursive_running
),

ordered_intervals AS (
  -- Step 5: Combine all drugs within a class and sort by start date
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    start_dt,
    end_dt,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY start_dt
    ) AS rn
  FROM fill_intervals
),

merged_class AS (
  -- Step 6: Recursive merge of overlapping intervals (PDC logic)
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    rn,
    start_dt,
    end_dt
  FROM ordered_intervals
  WHERE rn = 1

  UNION ALL

  SELECT
    o.lvl1_acct_gid,
    o.eph_id,
    o.drug_cls,
    o.fill_year,
    o.rn,
    CASE
      WHEN o.start_dt <= DATE_ADD(m.end_dt, INTERVAL 1 DAY)
      THEN m.start_dt
      ELSE o.start_dt
    END AS start_dt,
    CASE
      WHEN o.start_dt <= DATE_ADD(m.end_dt, INTERVAL 1 DAY)
      THEN GREATEST(m.end_dt, o.end_dt)
      ELSE o.end_dt
    END AS end_dt
  FROM merged_class m
  JOIN ordered_intervals o
    ON o.lvl1_acct_gid = m.lvl1_acct_gid
    AND o.eph_id = m.eph_id
    AND o.drug_cls = m.drug_cls
    AND o.fill_year = m.fill_year
    AND o.rn = m.rn + 1
    AND o.start_dt <= DATE_ADD(m.end_dt, INTERVAL 1 DAY)
)

-- Step 7: Compute class-level totals
SELECT
  f.lvl1_acct_gid,
  f.eph_id,
  f.drug_cls,
  f.fill_year,
  SUM(f.day_sply_qty) AS total_days_supplied,
  SUM(DATE_DIFF(m.end_dt, m.start_dt, DAY) + 1) AS unique_covered_days,
  SUM(f.day_sply_qty) - SUM(DATE_DIFF(m.end_dt, m.start_dt, DAY) + 1) AS excess_med_in_hand_class_level
FROM fill_intervals f
JOIN (
  SELECT DISTINCT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    start_dt,
    end_dt
  FROM merged_class
) m
USING (lvl1_acct_gid, eph_id, drug_cls, fill_year)
GROUP BY f.lvl1_acct_gid, f.eph_id, f.drug_cls, f.fill_year
ORDER BY f.lvl1_acct_gid, f.eph_id, f.drug_cls, f.fill_year;
