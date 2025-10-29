-- Anchor
DATE_ADD(a.fill_dt, INTERVAL CAST(MAX(a.day_sply_qty) AS INT64) DAY) AS rolling_end

-- Recursive
CASE
  WHEN b.fill_dt > p.rolling_end THEN DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY)
  ELSE GREATEST(p.rolling_end, DATE_ADD(b.fill_dt, INTERVAL CAST(b.day_sply_qty AS INT64) DAY))
END AS rolling_end
