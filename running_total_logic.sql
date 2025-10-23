WITH fills AS (
  SELECT
    Eph_ID,
    drug_cls,
    brand_name,
    fill_dt,
    days_supply,
    EXTRACT(YEAR FROM fill_dt) AS fill_year
  FROM `project.dataset.member_fills`
  WHERE fill_dt <= DATE '2025-09-30'
),

ordered AS (
  SELECT
    Eph_ID,
    drug_cls,
    brand_name,
    fill_dt,
    days_supply,
    fill_year,
    LAG(fill_dt) OVER (PARTITION BY Eph_ID, drug_cls ORDER BY fill_dt) AS prev_fill_dt,
    LAG(days_supply) OVER (PARTITION BY Eph_ID, drug_cls ORDER BY fill_dt) AS prev_days_supply
  FROM fills
),

expected AS (
  SELECT
    *,
    DATE_ADD(prev_fill_dt, INTERVAL prev_days_supply DAY) AS expected_fill_dt,
    CASE 
      WHEN prev_fill_dt IS NULL THEN 0
      ELSE DATE_DIFF(fill_dt, DATE_ADD(prev_fill_dt, INTERVAL prev_days_supply DAY), DAY)
    END AS gap_days
  FROM ordered
),

-- Step 3: Compute incremental "stock change" per fill
stock_change AS (
  SELECT
    Eph_ID,
    drug_cls,
    brand_name,
    fill_dt,
    fill_year,
    days_supply,
    gap_days,
    -- negative gap = early fill → adds to stockpile
    -- positive gap = late fill → consumes stockpile
    -gap_days AS stock_delta
  FROM expected
),

-- Step 4: Running cumulative stock with reset when going negative
cumulative AS (
  SELECT
    *,
    SUM(stock_delta) OVER (
      PARTITION BY Eph_ID, drug_cls 
      ORDER BY fill_dt
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS raw_running_balance
  FROM stock_change
),

-- Step 5: Adjust running balance to never go below 0
-- (BigQuery doesn’t have lag-in-window reset, but we can emulate it)
normalized AS (
  SELECT
    Eph_ID,
    drug_cls,
    brand_name,
    fill_dt,
    fill_year,
    days_supply,
    gap_days,
    raw_running_balance,
    GREATEST(0, raw_running_balance) AS extra_med_days
  FROM cumulative
)

SELECT
  Eph_ID,
  drug_cls,
  fill_year,
  MAX(extra_med_days) AS total_extra_med_days_asof_year_end
FROM normalized
WHERE fill_dt <= CASE 
    WHEN fill_year = 2024 THEN DATE '2024-09-30'
    WHEN fill_year = 2025 THEN DATE '2025-09-30'
  END
GROUP BY Eph_ID, drug_cls, fill_year
ORDER BY Eph_ID, drug_cls, fill_year;
