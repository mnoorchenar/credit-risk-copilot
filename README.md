---
title: Credit Risk Copilot
emoji: 🗺️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Credit Risk Copilot

An interactive map plus a plain-language assistant for residential secured lending risk across Ontario.
Click a region, or ask "Which areas have the worst credit risk?", "Why is Toronto scored this way?", "Compare Peel and Ottawa"
or "What if rates rise 200 bps and house prices fall 15%?". Answers appear as ranked bars, comparison tables and trend charts,
and the map recolors under stress. A **Guided demo** button plays a one-minute tour.

> **All data is synthetic.** It does not come from any lender and is not tied to any institution's policy, portfolio or models.

## How the AI is used (and kept honest)

1. A free open model on Hugging Face Inference Providers turns the question into a small structured JSON query.
2. A deterministic engine (`js/engine.js`) validates that query and computes every number from the data.
3. The model may add a two-sentence summary, shown only if every number in it appears in the computed facts.
4. Without a token, or if the API fails, a built-in rules parser answers instead, so the demo never breaks.

To turn the AI on, click **Connect AI** and paste a Hugging Face token that can call Inference Providers
(fine-grained token, free tier is enough). The token stays in your browser, is sent only to `router.huggingface.co`,
and is stored locally only if you tick "Remember on this device".

## Run locally

```bash
python -m http.server 7860      # then open http://localhost:7860
node --test tests/engine.test.mjs
python scripts/build_data.py    # regenerate the synthetic portfolio
```

## Notes

- Region shapes are Ontario public health unit boundaries (Open Government Licence - Ontario), used as a stand-in for credit regions.
- Stress-test elasticities are illustrative, not calibrated to any real portfolio.
