# Mini-project rubric review (before vs. after improvements)

This review is based on the files supplied in the project archive. Scores are an estimate against the supplied rubric, not a guarantee of the instructor's marks.

## Initial assessment

| Criterion | Initial estimate | Evidence and gap |
|---|---:|---|
| Data cleaning & splitting | 2.5 / 3 | Removes ID and the single `gender == Other` row, coerces BMI, uses stratified 80/20 split, and keeps imputation/outlier capping inside pipelines. Could document the cleaning summary and show EDA outputs more clearly. |
| Model comparison & analytics | 2.5 / 4 | Compares Dummy, Logistic Regression, KNN, Decision Tree and Random Forest; reports PR-AUC, ROC-AUC, recall, precision, F1 and accuracy; uses CV and threshold tuning. Critical inconsistency: the saved model was hard-coded to Random Forest although the supplied CV PR-AUC table ranks Decision Tree highest. |
| UI & cross-platform accessibility | 2 / 4 | A basic Streamlit form existed. An optional styled FastAPI page existed, but its normalized 0–100 risk index and clinical-sounding wording could mislead users. Deployment setup was not included. |
| Code quality, GitHub, report | 2 / 4 | Source is split into modules and a report exists, but `requirements.txt` omitted Streamlit and plotting dependencies, the archive bundled a large local `.venv`, README claims were inconsistent with the results, and deployment instructions were incomplete. |
| **Estimated initial total** | **9 / 15** | |

## Improvements made in this revision

- Rebuilt the Streamlit interface with a responsive design system, project navigation, patient-profile form, result cards, model-analytics page, feature-importance chart, and project overview.
- Added explicit non-clinical disclaimers and replaced misleading “low/high risk” claims with neutral above/below model-threshold language.
- Corrected training logic to select the final model family using the cross-validation PR-AUC criterion instead of hard-coding Random Forest.
- Re-ran training; the selected family is **Decision Tree** for this fixed split, with CV PR-AUC approximately **0.293** and test PR-AUC approximately **0.420**. These are dataset-specific metrics, not clinical validation.
- Updated the FastAPI result to show the raw model probability percentage rather than a fabricated normalized risk index.
- Updated dependencies, README, Git ignore rules, Dockerfile, Streamlit configuration, and this review.
- Excluded the bundled virtual environment from the clean submission archive.

## Revised estimated score after using this revision

| Criterion | Revised estimate | What still affects full marks |
|---|---:|---|
| Data cleaning & splitting | 2.5 / 3 | Add a clear EDA/cleaning summary figure or notebook if the instructor expects visual evidence. |
| Model comparison & analytics | 3.5 / 4 | Explain threshold trade-offs and the precision/recall limitations in the final report; avoid calling the probability calibrated. |
| UI & cross-platform accessibility | 3.5 / 4 | Run it on a phone-width browser and deploy to a public host before claiming live deployment. |
| Code quality, GitHub, report | 3 / 4 | Push the cleaned files to GitHub, include screenshots and a final conclusion, and verify setup from a fresh environment. |
| **Revised estimated total** | **12.5 / 15** | |

## Before submission checklist

- [ ] Test the app from a clean Python environment using `pip install -r requirements.txt`.
- [ ] Check the desktop and mobile browser layout.
- [ ] Include screenshots of the app, model comparison, confusion matrices and PR/ROC curves in the report.
- [ ] In the report, clearly explain why PR-AUC is emphasized for an imbalanced dataset and why precision remains low.
- [ ] Add the GitHub URL and deployment URL only after verifying they work.
- [ ] Keep the raw dataset and any personal/medical data out of the public repository.
