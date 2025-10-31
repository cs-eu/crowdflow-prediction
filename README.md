# Crowdflow Prediction

## Project Structure

```
├── data
│   ├── Bottleneck_Data
│   ├── Corridor_Data
│   └── Walking_Pedestrian_Data
├── README.md
├── requirements.txt
├── pedestrian_prediction.ipynb
├── pedestrian_prediction_extra_ds.ipynb
├── pedestrian_prediction_NN_comparison.ipynb
├── pedestrian_prediction_SFM.ipynb
└── src
    ├── models
    │   └── modeling.py
    └── utils
        ├── data_loader.py
        └── visualization.py
```

* `data/` – pedestrian trajectory datasets.
* `src/` – implementation of models and utility functions.
* `.ipynb` files – Jupyter notebooks for experimentation and analysis.
* `requirements.txt` – Python dependencies.

---

## Project Objectives

1. **Problem Analysis & Method Selection**
   Examine existing studies, identify challenges in reproducing experiments, and select appropriate modeling approaches.

2. **Neural Network Implementation**
   Implement a minimalistic neural network (single hidden layer, 3 neurons) to predict pedestrian speeds based on trajectory data.

3. **Initial Testing**
   Validate the network on simple scenarios to ensure functionality and accuracy.

4. **Model Evaluation**
   Compare neural network predictions with the Weidmann model and published results, highlighting similarities and discrepancies.

5. **Critical Analysis**
   Discuss the strengths, limitations, and potential improvements of both modeling approaches.

---

## Implementation Details

### Data Processing

* Pedestrian trajectory data from bottleneck and corridor experiments.
* Preprocessing steps:

  * Compute pedestrian velocity
  * Extract k-nearest neighbor distances
  * Split into training (50%) and testing (50%) sets

### Weidmann Model

* Physics-based reference model for pedestrian speed estimation.

### Neural Network Model

* **Architecture:** Single hidden layer with 3 neurons
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Training:** Bootstrapping with 5-fold cross-validation for better generalization

### Model Comparison

* Evaluation metric: **MSE**
* Neural network outperforms the Weidmann model, especially in complex environments.

---

## Results and Observations

* Neural networks provide more accurate speed predictions compared to the Weidmann model.
* Bottleneck scenarios exhibit higher variability; neural networks improve prediction accuracy.
* Future improvements include deeper architectures with regularization for robust performance.

