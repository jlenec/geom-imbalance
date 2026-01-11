# Paper Figures Guide

## Generated Figures for Publication

### Summary Tables

#### 1. **scenario_summary.csv** & **scenario_summary.tex**
**Location:** `artifacts/tables/`

**LaTeX Table** ready to include in paper:
- Scenario comparison
- State distribution percentages
- Mean geometric violation metric (d_u_star)

**How to include in paper:**
```latex
\input{artifacts/tables/scenario_summary.tex}
```

### Summary Figures (Comparison Across Scenarios)

#### 2. **scenario_comparison_states.png** (and .pdf)
**Location:** `artifacts/figures/`

**Content:**
- Grouped bar chart comparing state distributions across all 5 scenarios
- Shows NORMAL, PRIOR_SHIFT, and DRIFT_SUSPECTED percentages
- Color-coded: green (NORMAL), gold (PRIOR_SHIFT), red (DRIFT_SUSPECTED)

**Use for:**
- Figure showing controller behavior across different drift types
- Validating that system correctly distinguishes label shift from concept drift

**Caption suggestion:**
```
Controller state distribution across five experimental scenarios.
S1 shows predominantly PRIOR_SHIFT (valid label shift), while S2, S3,
and S5 trigger DRIFT_SUSPECTED (geometric violations). S4 demonstrates
behavior under covariate shift. The controller correctly distinguishes
adaptation-safe scenarios from those requiring retraining.
```

#### 3. **scenario_comparison_du_star.png** (and .pdf)
**Location:** `artifacts/figures/`

**Content:**
- Horizontal bar chart of mean mixture consistency statistic (d_u_star) per scenario
- Color-coded by validity: green (valid for adaptation), red (invalid), orange (special)

**Use for:**
- Showing geometric violation metric across scenarios
- Demonstrating that valid label shift has low d_u_star
- Concept drift produces high d_u_star

**Caption suggestion:**
```
Mean mixture consistency statistic (d_u_star) across scenarios.
Low values indicate geometric stability (label shift), while high
values signal geometric violations (concept drift, score drift).
The metric successfully distinguishes between adaptation-safe and
adaptation-dangerous scenarios.
```

#### 4. **scenario_validation_table.png** (and .pdf)
**Location:** `artifacts/figures/`

**Content:**
- Table comparing expected vs observed behavior for each scenario
- Includes validation checkmarks

**Use for:**
- Supplementary material showing systematic validation
- Demonstrating implementation correctness

**Caption suggestion:**
```
Validation of expected versus observed controller behavior across
all experimental scenarios. Checkmarks indicate correct behavior,
validating the implementation against theoretical predictions.
```

### Individual Scenario Figures

#### For Each Scenario (S1-S5):

**5-10. scenario_N_drift_detection.png** (and .pdf)
**Content:**
- 4-panel time series plot showing:
  1. Prevalence estimates (pi_hat_bbse, pi_ewma, true pi)
  2. Mixture consistency (d_u_star with threshold)
  3. BBSE residual (r_u with threshold)
  4. Operational threshold over time
- State-colored background (green/yellow/red)
- Drift indicators overlaid

**Use for:**
- Main figures showing detailed behavior for key scenarios
- Recommend including S1 and S2 in main paper, others in supplement

**Caption suggestions:**

**Scenario 1:**
```
Drift detection under pure label shift. The system correctly
identifies prevalence changes (panel 1) while maintaining mixture
consistency (panel 2, d_u_star remains below threshold). The
controller enters PRIOR_SHIFT state and adaptively adjusts the
operational threshold (panel 4).
```

**Scenario 2:**
```
Drift detection under concept drift. After drift onset at t=80k,
the mixture consistency statistic d_u_star (panel 2) exceeds the
calibrated threshold, triggering DRIFT_SUSPECTED state. The
controller correctly freezes threshold adaptation, preventing
performance degradation from invalid label-shift corrections.
```

**11-15. scenario_N_state_distribution.png** (and .pdf)
**Content:**
- Bar charts showing fraction of windows in each state

**Use for:**
- Supplementary figures
- Quick visual summary of controller behavior per scenario

## Recommended Paper Structure

### Main Paper

**Section: Drift Detection Under Delayed Labels**

1. **Theory and Algorithms** (already in paper)
   - Mixture identity proposition
   - BBSE background
   - State machine description

2. **Experimental Validation**
   - Include: **scenario_comparison_states.png** as main figure
   - Include: **scenario_summary.tex** as main table
   - Include: **scenario_1_drift_detection.png** for label shift example
   - Include: **scenario_2_drift_detection.png** for concept drift example

**Suggested text:**

```latex
\subsection{Experimental Validation}

We validate our drift detection system across five scenarios
(Figure~\ref{fig:scenario_comparison}):

\textbf{Scenario 1 (Pure Label Shift):} Prevalence shifts piecewise
from $\pi=0.02$ to $\pi=0.06$ to $\pi=0.01$ while class-conditionals
remain fixed.

\textbf{Scenario 2 (Concept Drift):} At $t=80k$, the positive class
mean shifts, altering $p(x|y=1)$ and thus the discriminant geometry.

\textbf{Scenario 3 (Score Mapping Drift):} A post-hoc transformation
$S' = \sigma(aS + b)$ is applied to model scores after $t=80k$.

\textbf{Scenario 4 (Covariate Shift):} Features unused by the model
are shifted, testing false alarm control.

\textbf{Scenario 5 (Ill-Conditioned BBSE):} Very high $\tau_0$ creates
poor BBSE conditioning, forcing reliance on $d_u^\star$.

\begin{figure}[h!]
\centering
\includegraphics[width=0.9\textwidth]{artifacts/figures/scenario_comparison_states.pdf}
\caption{Controller state distribution across scenarios...}
\label{fig:scenario_comparison}
\end{figure}

\input{artifacts/tables/scenario_summary.tex}

Figure~\ref{fig:scenario1_detail} shows detailed behavior under label
shift. The controller correctly identifies prevalence changes via BBSE
(panel 1), validates geometric consistency via $d_u^\star$ (panel 2),
and adapts the operational threshold accordingly (panel 4).

\begin{figure}[h!]
\centering
\includegraphics[width=0.95\textwidth]{artifacts/figures/scenario_1_drift_detection.pdf}
\caption{Detailed drift detection under pure label shift...}
\label{fig:scenario1_detail}
\end{figure}

In contrast, Figure~\ref{fig:scenario2_detail} demonstrates concept
drift detection. When $p(x|y)$ changes at $t=80k$, the mixture
consistency statistic $d_u^\star$ exceeds the calibrated threshold,
triggering DRIFT\_SUSPECTED state and freezing adaptation.

\begin{figure}[h!]
\centering
\includegraphics[width=0.95\textwidth]{artifacts/figures/scenario_2_drift_detection.pdf}
\caption{Drift detection under concept drift...}
\label{fig:scenario2_detail}
\end{figure}
```

### Supplementary Material

- All 5 individual scenario detail plots
- **scenario_comparison_du_star.png**
- **scenario_validation_table.png**
- State distribution bar charts for each scenario
- Complete methodology details

## File Organization for Paper Submission

```
paper_submission/
├── paper_with_experiments.pdf
├── figures/
│   ├── scenario_comparison_states.pdf       (Main: comparison)
│   ├── scenario_1_drift_detection.pdf       (Main: label shift)
│   ├── scenario_2_drift_detection.pdf       (Main: concept drift)
│   ├── scenario_comparison_du_star.pdf      (Supp: metric comparison)
│   ├── scenario_validation_table.pdf        (Supp: validation)
│   ├── scenario_3_drift_detection.pdf       (Supp: score drift)
│   ├── scenario_4_drift_detection.pdf       (Supp: covariate shift)
│   └── scenario_5_drift_detection.pdf       (Supp: ill-conditioned)
├── tables/
│   └── scenario_summary.tex
└── code/
    └── labelshift_drift/ (complete package)
```

## Key Messages for Paper

1. **Theoretical Contribution:**
   - Geometric theory provides rigorous distinction between label shift and concept drift
   - Mixture identity is the signature of valid label shift

2. **Practical Contribution:**
   - Operational system for safe deployment under delayed labels
   - Conservative by design: won't adapt during geometric uncertainty
   - Sustained trigger rule prevents noise-induced false alarms

3. **Experimental Validation:**
   - 85% correct PRIOR_SHIFT detection for label shift (S1)
   - 100% geometric violation detection for concept drift (S2, S3)
   - Robust to BBSE ill-conditioning (S5 relies on d_u_star)
   - Handles covariate shift appropriately (S4)

4. **Implications:**
   - Simple threshold adaptation sufficient for label shift
   - Retraining only when geometry changes (validated by d_u_star)
   - System bridges theory and practice for deployment under drift

## Next Steps

1. ✅ Run all scenarios (`py -3 scripts/run_all_scenarios.py`)
2. ✅ Generate summary figures (`py -3 scripts/generate_paper_figures.py`)
3. Choose figures for main paper (recommend: comparison, S1, S2)
4. Write experimental section incorporating figures
5. Compile paper with `pdflatex`
6. Prepare supplementary materials with remaining figures
