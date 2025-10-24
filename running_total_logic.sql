-- Replace this with your actual dataset/table name
WITH RECURSIVE fills_sorted AS (
  -- Step 1: Sort fills and compute lag values
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt
    ) AS rn,
    LAG(fill_dt) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt
    ) AS prev_fill_dt,
    LAG(day_sply_qty) OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt
    ) AS prev_day_sply_qty
  FROM `project.dataset.fills`
  WHERE fill_dt >= '2024-01-01'
),

calc AS (
  -- Step 2: Compute net change in supply between fills
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    day_sply_qty,
    rn,
    CASE
      WHEN prev_fill_dt IS NULL THEN 0
      ELSE (prev_day_sply_qty - DATE_DIFF(fill_dt, prev_fill_dt, DAY))
    END AS net_excess_change
  FROM fills_sorted
),

-- Step 3: Recursive accumulation that resets to 0 when cumulative goes negative
recursive_excess AS (
  -- Base case: first fill in each group
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    rn,
    day_sply_qty,
    net_excess_change,
    GREATEST(net_excess_change, 0) AS running_excess
  FROM calc
  WHERE rn = 1

  UNION ALL

  -- Recursive case: accumulate from prior fill
  SELECT
    c.lvl1_acct_gid,
    c.eph_id,
    c.drug_cls,
    c.fill_year,
    c.fill_dt,
    c.rn,
    c.day_sply_qty,
    c.net_excess_change,
    GREATEST(
      r.running_excess + c.net_excess_change,
      0
    ) AS running_excess
  FROM calc c
  JOIN recursive_excess r
    ON c.lvl1_acct_gid = r.lvl1_acct_gid
    AND c.eph_id = r.eph_id
    AND c.drug_cls = r.drug_cls
    AND c.fill_year = r.fill_year
    AND c.rn = r.rn + 1
)

SELECT
  lvl1_acct_gid,
  eph_id,
  drug_cls,
  fill_year,
  fill_dt,
  day_sply_qty,
  running_excess AS excess_med_in_hand
FROM recursive_excess
ORDER BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt;
