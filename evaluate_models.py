import os
import pandas as pd
from survival_models import train_rsf_pipeline, train_coxph_model
from sksurv.metrics import concordance_index_censored

def save_results_to_csv(results_list, filename="rsf_evaluation_results.csv"):
    # Create results/ folder if it doesn't exist
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_path = os.path.join(results_dir, filename)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

# === Step 1: Train RSF ===
rsf_model, X_test_rsf, y_test_rsf, y_pred_rsf = train_rsf_pipeline()
rsf_c_index = concordance_index_censored(y_test_rsf["Status OS"], y_test_rsf["OS at baseline"], y_pred_rsf)[0]

# === Step 2: Train Cox PH ===
coxph_c_index, coxph_n_samples = train_coxph_model()

# === Step 3: Save results ===
results = [
    {
        "model": "Random Survival Forest",
        "concordance_index": round(rsf_c_index, 4),
        "num_test_samples": len(X_test_rsf)
    },
    {
        "model": "Cox Proportional Hazards",
        "concordance_index": round(coxph_c_index, 4),
        "num_test_samples": coxph_n_samples
    }
]

save_results_to_csv(results)






