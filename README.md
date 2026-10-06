---
title: Credit Risk Model Studio
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
short_description: Train, explain and govern a mortgage default model in-browser
---

# Credit Risk Model Studio

An end-to-end, in-browser workbench for a **probability-of-default model** on residential secured lending (mortgages and HELOCs).
Everything runs client-side: the data is generated, cleaned, modelled, explained and validated in your browser, and nothing is uploaded.

> **All data is synthetic.** It does not come from any lender and is not tied to any institution's portfolio, policy or models.

## The workflow

| Step | What you can do |
|---|---|
| **1 Data** | 30,000 accounts with realistic defects. Seven automated integrity controls find missing values, invalid codes and duplicates; fixes are logged and use training-set medians only (no leakage). Explore any feature against default rate. |
| **2 Train** | Train a **champion** regularised logistic scorecard and a **challenger** gradient-boosting model written from scratch. Tune hyperparameters, watch live learning curves, and use the *Overfit* preset to see a model memorise its training data and early stopping fix it. |
| **3 Evaluate** | ROC, AUC / Gini / KS, calibration, decile lift, score distribution and an interactive approve/decline operating point. |
| **4 Explain** | Exact **TreeSHAP** (verified against brute-force Shapley values), SHAP beeswarm, mean \|SHAP\| vs permutation importance, partial dependence / ICE with business-logic checks, and per-account waterfalls with adverse-action style **reason codes** and **counterfactual recourse**. |
| **5 Decide** | Turn PDs into auto-approve / manual-review / decline bands, expected vs realised loss, strategy curves and a **swap-set analysis** against a rules-based policy at the same approval rate. |
| **6 Govern** | Nine automated validation checks (discrimination, overfitting, out-of-time, calibration, PSI, monotonicity, segment consistency, explainability, data integrity), population drift, macro stress tests and a downloadable model card. |
| **7 Copilot** | An executive model-risk memo and Q&A. Optional AI (free Hugging Face Inference Providers) writes the text, but may only quote numbers from the computed fact sheet: anything else, or any claim that contradicts the validation results, is rejected and a template is used. |

## AI is optional

Click **Connect AI** and paste a Hugging Face token that can call Inference Providers (a free fine-grained token is enough).
It stays in your browser, is sent only to `router.huggingface.co`, and is stored locally only if you tick "Remember on this device".
Without a token every feature still works; only the memo and answers fall back to built-in templates.

## Run and test locally

```bash
python -m http.server 7860          # then open http://localhost:7860
node --test tests/ml.test.mjs tests/governance.test.mjs
```

The tests cover metrics, training, TreeSHAP additivity and equality with brute-force Shapley values, decision bands, swap-set maths, recourse, stress tests and the AI guardrails.

## Notes

- Models, SHAP, PSI and metrics are implemented in plain JavaScript (`js/ml.js`) so every number is traceable.
- Stress-test sensitivities are mechanical re-scoring, not an economic forecast. Segment checks cover business segments only; a real fairness review is broader.
