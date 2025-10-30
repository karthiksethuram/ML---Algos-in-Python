-- BigQuery: Recursive CTE must be first
RECURSIVE excess_calc AS (
  -- Base case: first fill per (lvl11_acct_gid, member_id, drug_cls, fill_year)
  SELECT
    CAST(lvl11_acct_gid AS STRING) AS lvl11_acct_gid,
    CAST(member_id AS STRING) AS member_id,
    CAST(drug_cls AS STRING) AS drug_cls,
    EXTRACT(YEAR FROM DATE(fill_date)) AS fill_year,
    DATE(fill_date) AS fill_date,
    CAST(day_sply_qty AS INT64) AS day_sply_qty,
    DATE_ADD(DATE(fill_date), INTERVAL CAST(day_sply_qty AS INT64) DAY) AS med_end_date,
    1 AS rn,
    CAST(0 AS INT64) AS prev_excess,
    DATE_ADD(DATE(fill_date), INTERVAL CAST(day_sply_qty AS INT64) DAY) AS current_end_date
  FROM your_table t
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY lvl11_acct_gid, member_id, drug_cls, EXTRACT(YEAR FROM DATE(fill_date))
    ORDER BY DATE(fill_date)
  ) = 1

  UNION ALL

  -- Recursive case: next fill event
  SELECT
    o.lvl11_acct_gid,
    o.member_id,
    o.drug_cls,
    o.fill_year,
    o.fill_date,
    o.day_sply_qty,
    o.med_end_date,
    o.rn,
    CASE 
      WHEN DATE_DIFF(o.fill_date, e.current_end_date, DAY) < 0 THEN 
        ABS(DATE_DIFF(o.fill_date, e.current_end_date, DAY))
      ELSE 0
    END AS prev_excess,
    CASE 
      WHEN o.med_end_date > e.current_end_date THEN o.med_end_date
      ELSE e.current_end_date
    END AS current_end_date
  FROM excess_calc e
  JOIN (
    SELECT
      CAST(lvl11_acct_gid AS STRING) AS lvl11_acct_gid,
      CAST(member_id AS STRING) AS member_id,
      CAST(drug_cls AS STRING) AS drug_cls,
      EXTRACT(YEAR FROM DATE(fill_date)) AS fill_year,
      DATE(fill_date) AS fill_date,
      CAST(day_sply_qty AS INT64) AS day_sply_qty,
      DATE_ADD(DATE(fill_date), INTERVAL CAST(day_sply_qty AS INT64) DAY) AS med_end_date,
      ROW_NUMBER() OVER (
        PARTITION BY lvl11_acct_gid, member_id, drug_cls, EXTRACT(YEAR FROM DATE(fill_date))
        ORDER BY DATE(fill_date)
      ) AS rn
    FROM your_table
  ) o
  ON o.lvl11_acct_gid = e.lvl11_acct_gid
  AND o.member_id = e.member_id
  AND o.drug_cls = e.drug_cls
  AND o.fill_year = e.fill_year
  AND o.rn = e.rn + 1
),

-- Summarize cumulative excess by year
final_excess AS (
  SELECT
    lvl11_acct_gid,
    member_id,
    drug_cls,
    fill_year,
    SUM(prev_excess) AS cumulative_excess
  FROM excess_calc
  GROUP BY 1,2,3,4
)

SELECT * 
FROM final_excess
ORDER BY lvl11_acct_gid, member_id, drug_cls, fill_year;
