WITH base AS (
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    MAX(DATE_ADD(fill_dt, INTERVAL day_sply_qty - 1 DAY)) AS med_end_dt,  -- class-level end date
    MAX(day_sply_qty) AS day_sply_qty,
    ARRAY_AGG(DISTINCT drug_name IGNORE NULLS) AS drug_names
  FROM `your_dataset.your_table_name`
  GROUP BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt
),

ordered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY lvl1_acct_gid, eph_id, drug_cls, fill_year ORDER BY fill_dt) AS rn
  FROM base
),

RECURSIVE recursive_calc AS (

  -- Anchor: first fill
  SELECT
    lvl1_acct_gid,
    eph_id,
    drug_cls,
    fill_year,
    fill_dt,
    med_end_dt,
    day_sply_qty,
    drug_names,
    0 AS days_between,
    0 AS excess_days,
    med_end_dt AS running_end_dt
  FROM ordered
  WHERE rn = 1

  UNION ALL

  -- Recursive part
  SELECT
    b.lvl1_acct_gid,
    b.eph_id,
    b.drug_cls,
    b.fill_year,
    b.fill_dt,
    b.med_end_dt,
    b.day_sply_qty,
    b.drug_names,
    DATE_DIFF(b.fill_dt, p.fill_dt, DAY) AS days_between,

    CASE 
      WHEN b.fill_dt <= p.running_end_dt 
        THEN GREATEST(0, DATE_DIFF(p.running_end_dt, b.fill_dt, DAY))  -- only true overlap
      ELSE 0
    END AS excess_days,

    GREATEST(p.running_end_dt, b.med_end_dt) AS running_end_dt

  FROM recursive_calc p
  JOIN ordered b
    ON b.lvl1_acct_gid = p.lvl1_acct_gid
    AND b.eph_id = p.eph_id
    AND b.drug_cls = p.drug_cls
    AND b.fill_year = p.fill_year
    AND b.rn = (
      SELECT MIN(rn)
      FROM ordered
      WHERE lvl1_acct_gid = p.lvl1_acct_gid
        AND eph_id = p.eph_id
        AND drug_cls = p.drug_cls
        AND fill_year = p.fill_year
        AND rn > (
          SELECT rn FROM ordered
          WHERE lvl1_acct_gid = p.lvl1_acct_gid
            AND eph_id = p.eph_id
            AND drug_cls = p.drug_cls
            AND fill_year = p.fill_year
            AND fill_dt = p.fill_dt
        )
    )
)

SELECT
  lvl1_acct_gid,
  eph_id,
  drug_cls,
  fill_year,
  fill_dt,
  day_sply_qty,
  FORMAT_DATE('%Y-%m-%d', med_end_dt) AS med_end_dt,
  excess_days,
  FORMAT_DATE('%Y-%m-%d', running_end_dt) AS running_end_dt,
  ARRAY_TO_STRING(drug_names, ', ') AS drugs_in_class
FROM recursive_calc
ORDER BY lvl1_acct_gid, eph_id, drug_cls, fill_year, fill_dt;
