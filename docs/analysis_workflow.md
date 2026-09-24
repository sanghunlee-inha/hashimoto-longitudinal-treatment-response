# Analysis workflow

## 1. Cohort lock
The analysis cohort is frozen at 1,022 exact longitudinal transitions from 458 patients.

## 2. Target lock
Primary target: `outcome_log_excess_reduction`.

## 3. Validation
Patient-grouped OOF validation is used so that all records from one patient remain in the same fold.

## 4. Uncertainty
Uncertainty intervals are estimated using 20,000 patient-cluster bootstrap replicates.

## 5. Clinical benchmark
The locked ML prediction is compared with a TSH-only linear benchmark on matched transitions.

## 6. Longitudinal decoupling
Within the consecutive-transition subset (564 transitions, 329 patients), the analysis evaluates:

- direct predicted next response vs observed next response;
- raw Δ association as descriptive only;
- innovation correlation after removing the current-response effect; and
- cluster-robust cross-lag regression:
  `observed_next ~ predicted_next + current_response`.

## 7. Safeguards
No synthetic dates, automatic time transformation, first-observed-as-onset assumption, or use of model output as ground truth.
