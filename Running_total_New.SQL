-- 1) base intervals per fill
WITH fills AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    day_sply_qty,
    DATE_ADD(fill_dt, INTERVAL day_sply_qty - 1 DAY) AS end_dt
  FROM `project.dataset.fills`
  WHERE fill_dt >= '2024-01-01'
),

-- 2) aggregated total supply per class (no duplication)
class_total AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    SUM(day_sply_qty) AS total_days_supplied
  FROM fills
  GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year
),

-- 3) order intervals across all drugs in same class
ordered_intervals AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt AS start_dt,
    end_dt,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt, end_dt
    ) AS rn
  FROM fills
),

-- 4) recursively merge overlapping/contiguous intervals per class
--    produces non-overlapping merged segments
RECURSIVE merged AS (
  -- base: first interval in each partition
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
      WHEN o.start_dt <= DATE_ADD(m.end_dt, INTERVAL 1 DAY) THEN m.start_dt
      ELSE o.start_dt
    END AS start_dt,
    CASE
      WHEN o.start_dt <= DATE_ADD(m.end_dt, INTERVAL 1 DAY) THEN GREATEST(m.end_dt, o.end_dt)
      ELSE o.end_dt
    END AS end_dt
  FROM merged m
  JOIN ordered_intervals o
    ON o.lvl1_acct_gid = m.lvl1_acct_gid
    AND o.eph_id = m.eph_id
    AND o.drug_cls = m.drug_cls
    AND o.fill_year = m.fill_year
    AND o.rn = m.rn + 1
  -- note: join condition intentionally does not filter o.start_dt > m.end_dt
  -- because the CASE above handles overlap vs non-overlap
),

-- 5) deduplicate merged segments: keep only rows that represent final merged segments
--    (the recursive merge may produce intermediate rows; selecting distinct start/end per rn)
final_segments AS (
  SELECT DISTINCT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    start_dt,
    end_dt
  FROM merged
)

-- 6) compute unique covered days per class and join with total supply
SELECT
  t.lvl1_acct_gid,
  t.eph_id,
  t.drug_cls,
  t.fill_year,
  t.total_days_supplied,
  COALESCE(SUM(DATE_DIFF(s.end_dt, s.start_dt, DAY) + 1), 0) AS unique_covered_days,
  t.total_days_supplied - COALESCE(SUM(DATE_DIFF(s.end_dt, s.start_dt, DAY) + 1), 0) AS excess_med_in_hand_class_level
FROM class_total t
LEFT JOIN final_segments s
  USING (lvl1_acct_gid, eph_id, drug_cls, fill_year)
GROUP BY t.lvl1_acct_gid, t.eph_id, t.drug_cls, t.fill_year, t.total_days_supplied
ORDER BY t.lvl1_acct_gid, t.eph_id, t.drug_cls, t.fill_year;
