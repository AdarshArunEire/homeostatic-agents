# Prototype 2


## SIMULATION LENGTH SUMMARY: len_sweep_300k_500k_1m

### Configs tested:
  configs: 3
  total runs: 15
  seeds per config: 5

### Compact summary table:
 config  n_seeds  sim_len run_time_seconds_median  mean_comfort_median  mean_comfort_std  eval_deaths_median zero_death_rate  food_visit_pct_median  water_visit_pct_median  path_efficiency_median perfectish_trip_rate_median solved_rate
len500k        5   500000                  2m 21s                0.584             0.023                17.0              0%                    9.3                    61.4                     1.0                         97%          0%
  len1m        5  1000000                  4m 41s                0.572             0.053                15.0              0%                   10.1                    63.8                     1.0                         96%          0%
len300k        5   300000                  1m 24s                0.540             0.104                19.0              0%                    9.8                    60.2                     1.0                         98%          0%


### Winners / notable configs


**Best median comfort:**
  len500k
  median comfort = 0.584
  std comfort    = 0.023

**Most stable comfort:**
  len500k
  std comfort    = 0.023
  median comfort = 0.584

**Best death robustness:**
  len1m
  zero-death rate     = 0%
  median eval deaths  = 15.0

**Best path efficiency:**
  len300k
  path efficiency        = 1.000
  perfect-ish trip rate  = 98%

**Best overall solved rate:**
  len500k
  solved rate      = 0%
  median comfort   = 0.584
  path efficiency  = 1.000


 config  sim_len  mean_comfort_median  mean_comfort_std zero_death_rate  path_efficiency_median perfectish_trip_rate_median solved_rate
len300k   300000                0.540             0.104              0%                     1.0                         98%          0%
len500k   500000                0.584             0.023              0%                     1.0                         97%          0%
  len1m  1000000                0.572             0.053              0%                     1.0                         96%          0%