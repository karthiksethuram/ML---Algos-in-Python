WITH RECURSIVE recursive_calc AS (

  ----------------------------------------------------------------
  -- Anchor: the first fill per lvl1_acct_gid, eph_id, drug_cls, fill_year
  --   - combine same-day same-class fills using MAX(day_sply_qty)
  --   - set running_excess = day_sply_qty (supply on hand immediately after the fill)
  ----------------------------------------------------------------
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
    ) AS rn,
    0 AS days_between,
    MAX(day_sply_qty) AS running_excess
  FROM `your_dataset.your_table_name`
  GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt
  QUALIFY rn = 1

  UNION ALL

  ----------------------------------------------------------------
  -- Recursive step: for each next ordered fill row (rn = prev.rn + 1)
  -- Compute:
  --   days_between = DATE_DIFF(b.fill_dt, p.fill_dt, DAY)
  --   remaining_before_next = GREATEST(0, p.running_excess - days_between)
  --   new_running_excess = remaining_before_next + b.day_sply_qty
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
    -- correct recursion: remaining before b = max(0, p.running_excess - days_between)
    -- then add b.day_sply_qty
    (GREATEST(0, p.running_excess - DATE_DIFF(b.fill_dt, p.fill_dt, DAY)) + b.day_sply_qty) AS running_excess
  FROM recursive_calc p
  JOIN (
    -- pre-aggregated base for same-day handling and row-numbering
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

-- Final result: running_excess is the days of extra medication in hand immediately after each fill
SELECT
  lvl1_acct_gid,
  eph_id,
  drug_cls,
  drug_name,
  fill_year,
  fill_dt,
  day_sply_qty,
  running_excess
FROM recursive_calc
ORDER BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt;
