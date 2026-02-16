# YOLO-OBB Trolley Detection Project

This repository contains a set of Jupyter Notebooks for training, fine-tuning, and evaluating YOLO-OBB (Oriented Bounding Box) models for trolley detection. The project explores different training strategies, including training on synthetic data, real data, mixed datasets, and transfer learning (fine-tuning).

## Project Structure

The project is organized into five main notebooks, each serving a specific stage in the machine learning pipeline:

1.  **`trolley_yolo_obb_training_synthetized.ipynb`**: Training on 100% Synthetic Data.
2.  **`trolley_yolo_obb_training_real.ipynb`**: Training on 100% Real Data.
3.  **`yolo_obb_mixed_training.ipynb`**: Training on Mixed Datasets (Synthetic + Real).
4.  **`yolo_obb_finetuning.ipynb`**: Fine-tuning Synthetic Models on Real Data.
5.  **`testing.ipynb`**: Evaluation and Testing.

## Notebook Descriptions

### 1. Training on Synthetic Data
*   **Filename**: `trolley_yolo_obb_training_synthetized.ipynb`
*   **Purpose**: Trains a YOLO-OBB model purely on synthetic (rendered) data.
*   **Key Steps**:
    *   Installs dependencies (Ultralytics, Albumentations, etc.).
    *   Processes synthetic datasets from folders like `Synthetized_Data/Capture_as_our_dataset`, `Capture_scene_8`, etc.
    *   Applies data augmentation (geometric transformations, noise, etc.) to the training set.
    *   Splits data into Train (80%) and Validation (20%).
    *   Trains the model and saves the weights.

### 2. Training on Real Data
*   **Filename**: `trolley_yolo_obb_training_real.ipynb`
*   **Purpose**: Trains a YOLO-OBB model purely on real-world data to establish a baseline.
*   **Key Steps**:
    *   Loads real data from `Real_Data/`.
    *   Respects existing splits (e.g., `train(80_ REAL DATA)`, `test(20_ REAL DATA)`).
    *   Applies specific augmentations.
    *   Trains the model on the real dataset.

### 3. Mixed Training (Synthetic + Real)
*   **Filename**: `yolo_obb_mixed_training.ipynb`
*   **Purpose**: Investigates if adding synthetic data improves performance when real data is limited.
*   **Experiments**:
    *   **Mixed Data**: Trains from scratch on **100% Synthetic Data** combined with various percentages of **Real Data** (5%, 10%, 20%, 30%, 40%).
    *   **Real Data Baselines**: Trains from scratch on *only* the corresponding percentages of real data for comparison.
*   **Key Steps**:
    *   Dynamically creates proper dataset YAML files for each mixture.
    *   Runs multiple training experiments in sequence.
    *   Saves results to `YOLO_OBB_Mixed_Training_Results`.

### 4. Fine-Tuning
*   **Filename**: `yolo_obb_finetuning.ipynb`
*   **Purpose**: Takes a model pre-trained on synthetic data and fine-tunes it on small amounts of real data.
*   **Experiments**:
    *   Fine-tunes the synthetic model on 5%, 10%, 20%, 30%, 40%, and 50% of real data.
    *   Compares against training from scratch (baseline) on the same subsets.
*   **Key Features**:
    *   Uses **Fixed Splits**: Ensures that the specific 5% or 10% subset of data used is consistent across experiments for fair comparison.
    *   Optimized hyperparameters for fine-tuning (lower learning rate, freezing backbone layers initially).

### 5. Benchmark & Evaluation
*   **Filename**: `testing.ipynb`
*   **Purpose**: The final evaluation step. It takes all the trained models from the previous steps and evaluates them on a held-out test set.
*   **Key Steps**:
    *   Defines a list of model paths (e.g., `yolo26l_mixed_5pct.pt`, `yolo_obb_ft_10pct.pt`).
    *   Runs `model.val()` on the `last-test` dataset.
    *   Generates a comprehensive report (JSON and Text) containing mAP50, mAP50-95, Precision, and Recall for every model.

## Prerequisites & Usage

### Dependencies
The notebooks generally require the following Python libraries:
```bash
pip install ultralytics albumentations opencv-python numpy pyyaml tqdm scikit-learn wandb
```

### Data Directory Structure
The notebooks expect a specific directory structure. Ensure your data is organized as follows:
*   `Real_Data/`: Contains the real-world images and labels.
*   `Synthetized_Data/`: Contains the synthetic images and labels.
*   `last-test/`: The held-out test set for final evaluation.

### How to Run
1.  **Start with Synthetic Training**: Run `trolley_yolo_obb_training_synthetized.ipynb` to generate your pre-trained synthetic model.
2.  **Run Experiments**:
    *   For **Mixed Training**, run `yolo_obb_mixed_training.ipynb`.
    *   For **Fine-Tuning**, update the `SYNTHETIC_MODEL_PATH` in `yolo_obb_finetuning.ipynb` to point to your best synthetic model, then run the notebook.
3.  **Evaluate**: Run `testing.ipynb`. Ensure the paths in `models_to_test` point to your actual trained model weights.

## Output
*   **Weights**: Trained model weights are saved in their respective `runs/` directories or specifically defined output folders (e.g., `YOLO_OBB_FineTuning_Results`).
*   **Logs**: Training logs are saved to Weights & Biases (if enabled) and local `runs/` folders.
*   **Test Results**: Final evaluation metrics are saved in `last-test/test_results_last/` as JSON and TXT files.

## Data and Models Availability

The datasets used for training the models mentioned in the paper, as well as the trained models themselves, are available for download.

