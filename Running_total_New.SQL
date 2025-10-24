WITH RECURSIVE recursive_calc AS (
  -- 🔹 Anchor: first fill per member–drug_cls–year
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt,
    SUM(day_sply_qty) AS day_sply_qty,
    ROW_NUMBER() OVER (
      PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
      ORDER BY fill_dt
    ) AS rn,
    0 AS days_between,
    SUM(day_sply_qty) AS running_excess
  FROM `your_dataset.your_table_name`
  GROUP BY
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    drug_name,
    fill_year,
    fill_dt
  QUALIFY rn = 1

  UNION ALL

  -- 🔹 Recursive: compute running excess for next fills
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
    CASE
      WHEN (p.running_excess + p.day_sply_qty - DATE_DIFF(b.fill_dt, p.fill_dt, DAY)) < 0 THEN 0
      ELSE (p.running_excess + p.day_sply_qty - DATE_DIFF(b.fill_dt, p.fill_dt, DAY))
    END AS running_excess
  FROM recursive_calc p
  JOIN (
    -- pre-aggregated base table (same-day fills combined)
    SELECT
      lvl1_acct_gid,
      eph_id,
      drug_cls,
      drug_name,
      fill_year,
      fill_dt,
      SUM(day_sply_qty) AS day_sply_qty,
      ROW_NUMBER() OVER (
        PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year
        ORDER BY fill_dt
      ) AS rn
    FROM `your_dataset.your_table_name`
    GROUP BY
      lvl1_acct_gid,
      eph_id,
      drug_cls,
      drug_name,
      fill_year,
      fill_dt
  ) b
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
  running_excess
FROM recursive_calc
ORDER BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt;
