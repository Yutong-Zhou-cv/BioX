import os
import pandas as pd
import contextlib
import io
from sklearn.preprocessing import StandardScaler
from dowhy import CausalModel
from config import FILE_PATHS, NOTEARS_PARAMS, DOWHY_PARAMS
import logging
logging.getLogger("dowhy").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def standardize_species_name(species_raw, top_species_csv):

    species_raw = str(species_raw).strip().lower()
    df_top = pd.read_csv(top_species_csv)
    mapping = {}

    for full in df_top["Scientific Name"].dropna().unique():
        full_clean = str(full).strip()
        key = " ".join(full_clean.split()[:2]).lower() 
        mapping[key] = full_clean

    short_key = " ".join(species_raw.split()[:2]).lower()
    return mapping.get(short_key, species_raw)


def run_dowhy_batch_from_top_edges():

    # print(f"===Step 4: Causal Inference on Species Occurrence===")
    
    method = NOTEARS_PARAMS["method"]

    # Load top edges file
    top_edges_path = FILE_PATHS["notears_top_edges"].replace(".csv", f"_{method}.csv")
    save_dir =FILE_PATHS["DoWhy_output"]
    os.makedirs(save_dir, exist_ok=True)

    df_edges = pd.read_csv(top_edges_path)
    grouped = df_edges.groupby("Species")

    summary_results = []

    for species, group in grouped:
        
        if DOWHY_PARAMS["edges"] == "top5":
            top_bios = group["Source"].dropna().unique().tolist()
        else:
            top_bios = [f"BIO{i}" for i in range(1, 20)]
        
        if len(top_bios) == 0:
            continue

        print(f"Running DoWhy for: {species} | Top BIOs: {top_bios}")

        dfs = pd.read_csv(FILE_PATHS["top_species_csv"])
        dfs = dfs[dfs["Scientific Name"] == species].copy()

        df = pd.read_excel(FILE_PATHS["final_dataset_output"])
        df = df.dropna(subset=top_bios + ["Presence"])
        
        all_bios = [f"BIO{i}" for i in range(1, 20)]

        results = []

        for bio in top_bios:
            
            if DOWHY_PARAMS["confounder_mode"] == "dag":
                bio_edges = group[group["Target"] == bio]
                covariates = bio_edges["Source"].dropna().unique().tolist()
                covariates = [b for b in covariates if b != bio and b in df.columns]

                if len(covariates) == 0:
                    covariates = [b for b in all_bios if b != bio and b in df.columns]
            else:
                covariates = [b for b in all_bios if b != bio and b in df.columns]
                        
            df_temp = df[["Presence", bio] + covariates].dropna().copy()

            scaler = StandardScaler()
            df_temp[df_temp.columns] = scaler.fit_transform(df_temp[df_temp.columns])

            with contextlib.redirect_stdout(io.StringIO()):
                model = CausalModel(
                    data=df_temp,
                    treatment=bio,
                    outcome="Presence",
                    common_causes=covariates,
                    instruments=None
                )
                identified_estimand = model.identify_effect()
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=DOWHY_PARAMS["method"]
                )

            species = standardize_species_name(species, FILE_PATHS["top_species_csv"])

            results.append({
                "Species": species,
                "Treatment": bio,
                "ATE": estimate.value,
                "Method": "PSM",
                "Significance": estimate.test_stat_significance("bootstrap"),
                "CommonCauses": covariates
            })

        df_result = pd.DataFrame(results)

        out_path = os.path.join(save_dir, f"DoWhy_{species.replace(' ', '_')}_{method}.csv")
        df_result.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        summary_results.append(df_result)

    if summary_results:
        combined = pd.concat(summary_results, ignore_index=True)
        combined_path = os.path.join(save_dir, f"DoWhy_All_Results_{method}.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n⭐ All DoWhy results saved to: {combined_path}")
    else:
        print("[🤔Warning] No results were generated. Check data consistency.")


# === Example usage ===
if __name__ == "__main__":
    run_dowhy_batch_from_top_edges()
