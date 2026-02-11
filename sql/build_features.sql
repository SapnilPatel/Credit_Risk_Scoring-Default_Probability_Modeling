-- Creates a features table with clean numeric/categorical fields and label
CREATE OR REPLACE TABLE loans_features AS
WITH base AS (
  SELECT
    *,
    -- label: 1 = bad (default), 0 = good
    CASE
      WHEN loan_status IN ('Charged Off', 'Default') THEN 1
      WHEN loan_status IN ('Fully Paid') THEN 0
      ELSE NULL
    END AS y,

    -- clean fields
    CAST(REPLACE(int_rate, '%', '') AS DOUBLE) / 100.0 AS int_rate_num,
    CAST(REPLACE(revol_util, '%', '') AS DOUBLE) / 100.0 AS revol_util_num,

    -- term like " 36 months"
    CAST(REGEXP_EXTRACT(term, '([0-9]+)') AS INTEGER) AS term_months,

    -- emp_length like "10+ years", "< 1 year"
    CASE
      WHEN emp_length IS NULL THEN NULL
      WHEN emp_length LIKE '10+%' THEN 10
      WHEN emp_length LIKE '< 1%' THEN 0
      ELSE CAST(REGEXP_EXTRACT(emp_length, '([0-9]+)') AS INTEGER)
    END AS emp_years,

    -- issue_d like "Dec-2016"
    TRY_STRPTIME(issue_d, '%b-%Y') AS issue_dt
  FROM loans_raw
)
SELECT
  -- label
  y,

  -- numeric features
  loan_amnt,
  term_months,
  int_rate_num,
  installment,
  annual_inc,
  dti,
  delinq_2yrs,
  revol_util_num,
  total_acc,
  emp_years,

  -- categorical features
  grade,
  sub_grade,
  home_ownership,
  verification_status,
  purpose,

  -- time column (for time-based split)
  issue_dt

FROM base
WHERE y IS NOT NULL;
