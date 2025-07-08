import os
import pandas as pd
from ollama import chat
from config import FILE_PATHS, NOTEARS_PARAMS, EXPLAIN_PARMS

import logging

for name in logging.root.manager.loggerDict:
    if "http" in name or "ollama" in name:
        logging.getLogger(name).setLevel(logging.WARNING)

# === BIO Descriptions===
BIO_DESC = {
    "BIO1": "Annual Mean Temperature",
    "BIO2": "Mean Diurnal Range (Mean of monthly max temp - min temp)",
    "BIO3": "Isothermality (BIO2/BIO7)*100",
    "BIO4": "Temperature Seasonality (standard deviation *100)",
    "BIO5": "Max Temperature of Warmest Month",
    "BIO6": "Min Temperature of Coldest Month",
    "BIO7": "Temperature Annual Range (BIO5-BIO6)",
    "BIO8": "Mean Temperature of Wettest Quarter",
    "BIO9": "Mean Temperature of Driest Quarter",
    "BIO10": "Mean Temperature of Warmest Quarter",
    "BIO11": "Mean Temperature of Coldest Quarter",
    "BIO12": "Annual Precipitation",
    "BIO13": "Precipitation of Wettest Month",
    "BIO14": "Precipitation of Driest Month",
    "BIO15": "Precipitation Seasonality (Coefficient of Variation)",
    "BIO16": "Precipitation of Wettest Quarter",
    "BIO17": "Precipitation of Driest Quarter",
    "BIO18": "Precipitation of Warmest Quarter",
    "BIO19": "Precipitation of Coldest Quarter"
}

def is_significant(pval):
    try:
        if isinstance(pval, str):
            pval = eval(pval)
        if isinstance(pval, tuple):
            return pval[1] < 0.05
        elif isinstance(pval, dict):
            pv = pval.get("p_value", 1)
            if isinstance(pv, tuple):
                return pv[1] < 0.05
            else:
                return float(pv) < 0.05
        else:
            return float(pval) < 0.05
    except:
        return False

# === Rule-based Explanation ===
def rule_explanation(treatment, effect, controls, species_name):
    direction = "✅" if abs(effect) > 0.05 else "❌"
    effect_str = f"{effect:+.4f}"
    ctrl_text = ", ".join(controls) if isinstance(controls, list) else controls or "None"
    bio_name = BIO_DESC.get(treatment, treatment)
    strength = "strong" if abs(effect) > 0.1 else "moderate" if abs(effect) > 0.05 else "weak"

    if effect > 0:
        if strength == "strong":
            interp = f"High {bio_name} strongly promotes {species_name} presence."
        elif strength == "moderate":
            interp = f"Increasing {bio_name} moderately increases occurrence."
        else:
            interp = f"{bio_name} has a weak positive effect."
    else:
        if strength == "strong":
            interp = f"{bio_name} strongly limits {species_name} distribution."
        elif strength == "moderate":
            interp = f"Higher {bio_name} moderately suppresses presence."
        else:
            interp = f"{bio_name} has a minor negative effect."

    return f"{treatment} — {bio_name}\n{direction} Effect: {effect_str} \nExplanation: {interp}\n"

# === LLM-enhanced Explanation ===
def llm_prompt(species, treatment, bio_name, effect, controls):
    direction = "positive" if effect > 0 else "negative"
    effect_str = f"{effect:+.3f}"
    ctrl_text = ", ".join(controls) if controls else "None"

    prompt = f"""
    You are an ecological scientist with expertise in species distribution modeling and climate-driven habitat analysis.

    Given the following causal inference result, explain in detail why this specific bioclimatic variable may causally influence the presence of the species.
    Base your explanation on ecological principles, including physiological tolerances, climatic constraints, seasonal dependencies, or habitat specialization.
    Ensure the explanation is realistic, grounded in ecological reasoning, and free from vague generalizations.

    Species: {species}
    BIO Variable: {treatment} ({bio_name})
    Estimated Causal Effect: {effect_str} ({direction})
    Controlled Variables: {ctrl_text}

    Write 3-5 sentence explaining the most likely ecological mechanism behind this causal link. Use precise ecological terminology and avoid speculative or non-scientific language.
    Explanation:

    """
    return prompt.strip()

# === Main Execution ===
def explain_significant_bios():
    
    # print(f"===Step 5: Explanation Generation===")
    
    method = NOTEARS_PARAMS["method"]
    mode = EXPLAIN_PARMS.get("explanation_mode", "rule")

    dowhy_dir = FILE_PATHS["DoWhy_output"]

    output_txt_dir = os.path.join(FILE_PATHS["Exp_output"], f"Explanations_{mode}_{method}")
    os.makedirs(output_txt_dir, exist_ok=True)
    
    all_files = [
    f for f in os.listdir(dowhy_dir)
    if f.startswith("DoWhy_") and
       f.endswith(f"_{method}.csv") and
       not f.startswith("DoWhy_All_Results")
    ]

    for file in all_files:
        species = file.replace("DoWhy_", "").replace(f"_{method}.csv", "").replace("_", " ")
        input_path = os.path.join(dowhy_dir, file)

        df = pd.read_csv(input_path)
        if "All_Results" in file:
            continue
        df["Species"] = species
        df["Significant"] = df["Significance"].apply(is_significant)
        df_sig = df[df["Significant"]].copy()

        lines = [f"Species: {species}\n"]

        if df_sig.empty:
            print(f"No significant BIO found for species: {species}")
            lines.append("[🤔Warning] No statistically significant causal effects found.\n")
        else:
            print("Species:", df_sig["Species"].unique())

        for _, row in df_sig.iterrows():
            treatment = row["Treatment"]
            effect = row["ATE"] if "ATE" in row else row["Effect"]
            controls = eval(row["CommonCauses"]) if isinstance(row["CommonCauses"], str) else row["CommonCauses"]
            bio_name = BIO_DESC.get(treatment, treatment)

            if mode == "llm":
                prompt = llm_prompt(species, treatment, bio_name, effect, controls)
                response = chat(model=EXPLAIN_PARMS["llm_model"], messages=[{"role": "user", "content": prompt}])
                explanation = response['message']['content'].strip()
            else:
                explanation = rule_explanation(treatment, effect, controls, species)

            lines.append(f"{explanation}\n")

        output_file = os.path.join(output_txt_dir, f"{species.replace(' ', '_')}_explanation.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"⭐ Saved explanation for: {species}")

if __name__ == "__main__":
    explain_significant_bios()
