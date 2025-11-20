-- Replace dataset.table with your actual input table
WITH base AS (
    SELECT
        eph_id,
        fill_date,
        day_supply,
        units_dispensed,
        strength_mg,
        (units_dispensed * strength_mg) AS total_qty_dispensed,
        DATE_ADD(fill_date, INTERVAL day_supply DAY) AS exhaust_date
    FROM `project.dataset.claims_fill_level`
),

-- Generate month-end reporting dates from the min to max fill range
month_calendar AS (
    SELECT
        LAST_DAY(month_start) AS reporting_date
    FROM (
        SELECT
            DATE_TRUNC(MIN(fill_date), MONTH) AS min_month,
            DATE_TRUNC(MAX(fill_date), MONTH) AS max_month
        FROM base
    ), UNNEST(
        GENERATE_DATE_ARRAY(min_month, max_month, INTERVAL 1 MONTH)
    ) AS month_start
),

-- Cross join to get ReportingMonth × FillDate
expanded AS (
    SELECT
        m.reporting_date,
        b.*
    FROM base b
    CROSS JOIN month_calendar m
),

-- Compute metrics using your 3 formulas
final AS (
    SELECT
        reporting_date,
        eph_id,
        fill_date,
        day_supply,
        exhaust_date,
        units_dispensed,
        strength_mg,
        total_qty_dispensed,

        -- Formula 2: Day Supply Used = min(ds, date_diff)
        CASE
            WHEN reporting_date < fill_date THEN 0
            ELSE LEAST(day_supply, DATE_DIFF(reporting_date, fill_date))
        END AS day_supply_used,

        -- Total Qty Used = prorated by day supply used
        CASE
            WHEN reporting_date < fill_date THEN 0
            ELSE LEAST(day_supply, DATE_DIFF(reporting_date, fill_date)) 
                 * (total_qty_dispensed / day_supply)
        END AS total_qty_used,

        -- Formula 3: Last 30 days usage
        (
            SELECT
                CASE
                    WHEN end_date < start_date THEN 0
                    ELSE DATE_DIFF(end_date, start_date) + 1
                END
            FROM (
                SELECT
                    GREATEST(fill_date, DATE_SUB(reporting_date, INTERVAL 29 DAY)) AS start_date,
                    LEAST(exhaust_date, reporting_date) AS end_date
            )
        ) AS last_30_days_used

    FROM expanded
)

SELECT * 
FROM final
ORDER BY reporting_date, fill_date;
