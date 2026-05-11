# AutoAD 15-Minute Presentation and Demo Guide

## Timing

- Slides 1-7: about 9 minutes
- Slides 8-9 and live demo: about 4-5 minutes
- Slides 10-11: about 1-2 minutes

## Demo setup

Run from the repository root:

```powershell
.venv\Scripts\activate
cd api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000/

## Recommended demo file

- `data/test_data.txt`

## Live demo flow

1. Open the dashboard and point out the three-part workflow: EDA, synthetic preview/export, full pipeline.
2. Upload `data/test_data.txt` in EDA and run EDA. Mention row/column counts, missingness, correlations, and outlier statistics.
3. In synthetic preview, choose `spike_single` with seed `42`. Explain that this creates controlled anomalies with known ground truth.
4. Export the corrupted CSV. If the browser downloads it, use that file for the full pipeline upload.
5. Run full analysis. Interpret anomaly count, anomaly score table, model list, and metrics if labels are available.

## Backup plan

- If the full pipeline is slow, show only EDA and synthetic preview.
- Then explain that the full pipeline can take longer because it uses Optuna and PyTorch model startup.
- Use the prepared CSV results under `results/` to discuss evaluation.

## Closing sentence

AutoAD is valuable because it makes anomaly detection more reproducible: instead of trusting one fixed model, it gives us a workflow for inspecting data, injecting controlled anomalies, comparing detectors, and interpreting results.

## Slide 1: Automated Anomaly Detection for CSV-like Data

Open with the practical goal: reduce manual work when a new CSV dataset arrives. The project is both an ML pipeline and a usable API/UI workflow.

## Slide 2: Problem Motivation

Frame the problem as workflow fragility. We usually do not know in advance whether a tree, boundary, density, or reconstruction method will work best.

## Slide 3: Project Objective

Emphasize that the contribution is not just one algorithm. The contribution is an integrated system for analysis, model comparison, and evaluation.

## Slide 4: System Workflow

Walk through the workflow left to right. The separation between EDA and the pipeline makes the demo clear and keeps the system practical.

## Slide 5: Core Pipeline Components

Keep this slide technical but short. Mention why diversity matters: different detectors see different forms of abnormality.

## Slide 6: Synthetic Anomaly Injection

Explain that synthetic results should not be oversold. They test controlled behavior and robustness, but real data remains harder.

## Slide 7: Evaluation Takeaways

Use this as the honesty slide. The project becomes stronger when it admits where default ensembling does not generalize.

## Slide 8: Live Demo Plan

Keep the demo around four to five minutes. If the full pipeline is slow, show EDA and synthetic preview, then use prepared results as backup.

## Slide 9: Demo Script

This slide can stay on screen before the live demo, or be skipped if time is tight.

## Slide 10: Limitations and Next Steps

Be balanced. The system works end to end, but the main research challenge is generalization across datasets.

## Slide 11: Conclusion

End with the project value: it makes anomaly-detection experiments more systematic and explainable.
