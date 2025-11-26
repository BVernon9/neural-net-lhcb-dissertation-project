# Using Neural Networks to Predict Pixel Thresholds in the LHCb VELO

This repository contains the code and report for my BSc Physics dissertation at the University of Liverpool.  
The project uses neural networks to predict per-pixel threshold (“trim”) values in the upgraded LHCb Vertex Locator (VELO) from a reduced set of calibration measurements.

> Assessment outcome: First-class project (72% average across code, presentation and written report).

---

## Project overview

The upgraded VELO contains ~40 million pixels. Each pixel needs its threshold calibrated via a 16-step noise scan. This is accurate but slow and keeps the sensors powered (and warm) for longer than ideal.

In this project, the full scan is replaced by a **16-class classification problem**:

- Inputs: noise mean and noise width at the two extreme threshold settings (0x0 and 0xF).
- Output: predicted trim value (0–15) for each pixel.

A small feed-forward neural network (Keras/TensorFlow) is trained on construction data and evaluated on both construction and later operational data from the VELO. A boosted decision tree (XGBoost) and a regression-style neural network are used as baselines and comparisons. :contentReference[oaicite:0]{index=0}  

On construction data the final model reaches **≈98.6% accuracy within ±1 trim unit**, while inference on a 256×256 pixel tile takes under 2 seconds on CPU.

---

## Repository contents

Main files:

- `NNData_Many.ipynb`  
  Final neural-network model: classification approach using multiple training datasets and tuned architecture / preprocessing.

- `XGBData_Many.ipynb`  
  Boosted decision-tree (XGBoost) comparison using the same inputs and train/test splits as the neural network.

- `Regression_NNData_Many.ipynb`  
  Regression variant of the network (continuous output, later rounded back to trim values) used as an alternative approach.

- `documents/Project_Report.pdf`  
  Full dissertation report, including physics context, data description, model design, results and discussion.

Other files in the repo are earlier tests or helper notebooks used on the way to the final models.

---

## Data and reproducibility

The original LHCb VELO datasets are **not** included in this repository:

- All data were accessed via supervisor access to CERN EOS/LHCb storage.
- The data are not publicly shareable, and a lot of the work relies on large internal `.tgz` archives and processed arrays.

Because of that:

- The notebooks and scripts are here mainly as **read-only reference**.
- They will not run “out of the box” without the internal data and paths.
- The full methodology, preprocessing steps and evaluation are documented in `Project_Report.pdf`.

If you are interested in adapting this approach to your own detector / dataset, the report and notebooks should still be useful as a template for workflow and model design.

---

## Methods and models (short summary)

- **Frameworks:** Python, Keras/TensorFlow for neural networks, XGBoost for boosted trees.
- **Inputs:**  
  - Mean noise at low trim (0x0)  
  - Mean noise at high trim (0xF)  
  - Noise width at 0x0 and 0xF  
  - Faulty pixels removed based on mask flags

- **Final NN architecture:**  
  - 4 input features → 2 hidden layers (4 neurons each, ReLU) → 16-node softmax output layer (one per trim value).
  - Manual sensor-specific normalisation to preserve the relationship between noise at 0x0 and 0xF.
  - Accuracy metric rewards predictions within ±1 trim unit.

- **Comparisons:**
  - **XGB model** (in `XGBData_Many.ipynb`) achieves slightly lower accuracy but similar runtime.
  - **Regression NN** (in `Regression_NNData_Many.ipynb`) underperforms the classification setup and adds complexity in post-processing.

For full details, including correlation studies, hyper-parameter scans and operational-data tests, see `Project_Report.pdf`.

---

## Results (headline numbers)

- **Construction data (60 datasets across 10 modules):**  
  - ~98.6% of pixels predicted within ±1 trim unit.  
  - >99% of predictions within ±2 units.  
  - Training time <20 s; prediction <2 s per 256×256 array on an Intel i7-11370H CPU.

- **Operational 2025 module:**  
  - Accuracy drops to ~83% within ±1 trim unit due to changing sensor conditions and irradiation effects.

The report discusses how periodic retraining on updated full scans could turn this into a dynamic calibration system that tracks detector ageing over time.

---

## Acknowledgements

This work was completed as part of my final-year P
