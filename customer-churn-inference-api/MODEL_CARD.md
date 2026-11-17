# Model Card — Customer Churn Demonstration

## Intended use
Educational demonstration of a real-time ML inference API and FastAPI deployment pattern. Not for production or automated customer treatment.

## Model
Regularised scikit-learn Logistic Regression in a preprocessing pipeline. The pipeline is trained on the complete 12-row assignment dataset only to supply an executable model artifact for the API demonstration.

## Data
12 rows; 5 positive churn labels and 7 negative labels. Features are tenure, support-ticket count, monthly spend, last-login recency, and plan type. Customer ID is excluded.

## Known limitations
All five Basic records are labelled churned; all Standard and Pro records are labelled not churned. The sample is too small to establish generalisation, calibration, fairness, or a robust threshold. Data provenance, consent, churn definition, and prediction horizon are undocumented.

## Output
Binary class using a demonstration threshold of 0.5, plus estimated class-1 probability. Probabilities must not be interpreted as calibrated real-world risk.

## Safeguards
API responses explicitly identify the model as experimental and include a warning. Human review is required for any future pilot. No cancellation, pricing, eligibility, penalty, or service restriction decisions may be automated.

## Required work before deployment
Acquire larger permissioned time-stamped data; confirm labels and horizon; perform temporal validation, calibration and error analysis; assess appropriate group-level harms; define monitoring, rollback, and governance.
