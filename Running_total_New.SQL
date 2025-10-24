WITH RECURSIVE recursive_calc AS (
  -- Step 1: Aggregate same-day fills and assign order
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

  UNION ALL

  -- Step 2: Recursive logic to compute running excess
  SELECT
    b.lvl1_acct_gid,
    b.eph_id,
    b.drug_cls,
    b.drug_name,
    b.fill_year,
    b.fill_dt,
    b.day_sply_qty,
    b.rn,
    CASE
      WHEN (p.running_excess + p.day_sply_qty - DATE_DIFF(b.fill_dt, p.fill_dt, DAY)) < 0 THEN 0
      ELSE (p.running_excess + p.day_sply_qty - DATE_DIFF(b.fill_dt, p.fill_dt, DAY))
    END AS running_excess
  FROM recursive_calc p
  JOIN (
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
