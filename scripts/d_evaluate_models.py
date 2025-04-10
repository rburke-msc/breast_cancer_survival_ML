import os # Provides functions for interacting with the operating system (e.g., file paths, directories)
import pandas as pd # For tabular data manipulation and exporting to CSV
from c_survival_models import train_rsf_pipeline, train_coxph_model # For tabular data manipulation and exporting to CSV
from sksurv.metrics import concordance_index_censored # Metric to evaluate model performance on survival data

# --------------------------------------------------------
# Function: save_results_to_csv
# Description:
# Saves evaluation results to a CSV file under /results.
# Accepts a list of dictionaries and a filename.
# --------------------------------------------------------
def save_results_to_csv(results_list, filename="rsf_evaluation_results.csv"):
    # Create results/ folder if it doesn't exist
    results_dir = os.path.join("..", "results") # Define relative path to the results folder
    os.makedirs(results_dir, exist_ok=True) # Create directory if it doesn't already exist (prevents crash on first run)

    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(results_list) # Convert list of result dictionaries into a DataFrame for easier CSV export

    # Save to CSV
    output_path = os.path.join(results_dir, filename) # Full path to the output CSV file
    results_df.to_csv(output_path, index=False) # Save results without row indices (cleaner format)

    print(f"Results saved to {output_path}")


# --------------------------------------------------------
# Step 1: Train and Evaluate the Random Survival Forest
# --------------------------------------------------------
rsf_model, X_test_rsf, y_test_rsf, y_pred_rsf = train_rsf_pipeline()
# Train the RSF model and retrieve the model, test features, true labels, and predictions

# Calculate concordance index for RSF
rsf_c_index = concordance_index_censored(
    y_test_rsf["Status OS"], # Event indicator (1 if death occurred, 0 if censored)
    y_test_rsf["OS at baseline"], # Time to event (survival time)
    y_pred_rsf # Predicted risk or survival scores
)[0] # Extract only the concordance index value from the result tuple

# --------------------------------------------------------
# Step 2: Train and Evaluate the Cox Proportional Hazards Model
# --------------------------------------------------------
coxph_c_index, coxph_n_samples = train_coxph_model()
# Train Cox PH model and return C-index and number of test samples used

# --------------------------------------------------------
# Step 3: Compile and Save Results
# --------------------------------------------------------
results = [
    {
        "model": "Random Survival Forest", # Name of model
        "concordance_index": round(rsf_c_index, 4), # RSF model performance (rounded for readability)
        "num_test_samples": len(X_test_rsf) # Number of samples evaluated in RSF
    },
    {
        "model": "Cox Proportional Hazards", # Name of model
        "concordance_index": round(coxph_c_index, 4), # CoxPH performance
        "num_test_samples": coxph_n_samples # Sample count used for evaluation
    }
]

save_results_to_csv(results) # Call function to save final performance results to CSV file in results directory






