import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import FILE_PATHS, NOTEARS_PARAMS
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear_auto
import time

BIO_COLS = [f"BIO{i}" for i in range(1, 20)]
BIO_TYPE_COLOR = {
    "Temperature-related": "#EEA9A9",
    "Precipitation-related": "#A5DEE4",
    "Derived": "#B481BB"
}

def get_bio_type(bio_name):
    num = int(bio_name.replace("BIO", ""))
    if num in [1, 2, 5, 6, 8, 9, 10, 11]:
        return "Temperature-related"
    elif num in [12, 13, 14, 16, 17, 18, 19]:
        return "Precipitation-related"
    else:
        return "Derived"

def normalize_species(name):
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name



def notears_bio_dags(method=None):

    print(f"===Step 3: Environmental Causal Graph Learning===")
    
    if method is None:
        method = NOTEARS_PARAMS["method"]
    
    df_all = pd.read_excel(FILE_PATHS["final_dataset_output"])
    df_all = df_all[df_all["Presence"] == 1].copy()
    df_all = df_all[df_all["scientificName"].notna()]

    # Load species list and build normalized mapping
    top_species_df = pd.read_csv(FILE_PATHS["top_species_csv"])
    canonical_names = top_species_df["Scientific Name"].dropna().unique().tolist()
    name_map = {normalize_species(name): name for name in canonical_names}

    # Normalize and map scientific names to canonical form
    df_all["Normalized"] = df_all["scientificName"].apply(normalize_species)
    df_all["Canonical"] = df_all["Normalized"].map(name_map)

    grouped = df_all.groupby("Canonical")
    output_dir = FILE_PATHS["DAGs_output"]
    os.makedirs(output_dir, exist_ok=True)

    all_weights = {}
    all_top_edges = []
    all_explanations = []


    for canonical_name, group in grouped:
        
        start_time = time.perf_counter()

        if pd.isna(canonical_name):
            print("[❌Error] Unknown species mapping.")
            continue

        df_bio = group[BIO_COLS].dropna().copy()
        df_bio = df_bio.loc[:, df_bio.std() > 0]

        if df_bio.shape[0] < 30 or df_bio.shape[1] == 0:
            print(f"[❌Error] {canonical_name}: Insufficient data (rows={df_bio.shape[0]}, cols={df_bio.shape[1]})")
            continue

        print(f"Running NOTEARS for: {canonical_name} ({method})")

        X = df_bio.to_numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        var_names = df_bio.columns.tolist()

        try:
            if method == "linear":
                W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
            elif method == "nonlinear":
                W_est = notears_nonlinear_auto(X)
            else:
                raise ValueError("Please choose 'linear' or 'nonlinear'")
        except Exception as e:
            print(f"[❌Error] Failed to run NOTEARS on {canonical_name}: {e}")
            continue

        np.fill_diagonal(W_est, 0)
        G = nx.DiGraph()
        edges = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                w = W_est[i, j]
                if abs(w) > 0.04:
                    G.add_edge(var_names[j], var_names[i], weight=w)
                    edges.append((var_names[j], var_names[i], w))

        edges = sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:5]
        top_edges = [f"{a} → {b} (weight = {w:.2f})" for a, b, w in edges]
        explanations = [
            f"{a} {'positively' if w > 0 else 'negatively'} influences {b} with strength {abs(w):.2f}."
            for a, b, w in edges
        ]

        # pos = nx.shell_layout(G)
        var_names_sorted = sorted(G.nodes(), key=lambda x: int(x.replace("BIO", "")))
        pos = nx.shell_layout(G, nlist=[var_names_sorted])
        pos = nx.rescale_layout_dict(pos, scale=3)
        node_colors = [BIO_TYPE_COLOR[get_bio_type(n)] for n in G.nodes()]

        all_weights[canonical_name] = pd.DataFrame(W_est, columns=var_names, index=var_names)
        all_top_edges.extend([[canonical_name, src, tgt, w] for src, tgt, w in edges])
        all_explanations.extend([f"{canonical_name}: {line}" for line in explanations])

        plt.figure(figsize=(13, 11))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=10, arrowsize=20)
        edge_labels = { (a, b): f"{w:.2f}" for a, b, w in edges }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        legend_handles = [
            mpatches.Patch(color=color, label=label) for label, color in BIO_TYPE_COLOR.items()
        ]
        plt.legend(handles=legend_handles, title="BIO Variable Types", loc="lower right")

        plt.title(f"Causal DAG of BIO Variables for {canonical_name} ({method})")
        explanation_text = "\n".join(["Top 5 strongest causal edges:"] + [f"  - {line}" for line in top_edges])
        plt.gcf().text(0.01, 0.01, explanation_text, fontsize=9, va='bottom', ha='left', wrap=True)

        species_safe = canonical_name.replace(' ', '_').replace('/', '_')
        fig_path = os.path.join(output_dir, f"NOTEARS_DAG_{species_safe}_{method}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        elapsed = time.perf_counter() - start_time
        print(f"Time taken for {canonical_name}: {elapsed:.2f} seconds")

        print(f"⭐ The DAGs is/are saved at {output_dir}.")

    output_excel_path = FILE_PATHS["notears_output"].replace(".xlsx", f"_{method}.xlsx")
    with pd.ExcelWriter(output_excel_path) as writer:
        for name, df_w in all_weights.items():
            safe_name = name.replace(' ', '_').replace('/', '_')[:30]
            df_w.to_excel(writer, sheet_name=safe_name)

    top_edges_df = pd.DataFrame(all_top_edges, columns=["Species", "Source", "Target", "Weight"])
    top_edges_path = FILE_PATHS["notears_top_edges"].replace(".csv", f"_{method}.csv")
    top_edges_df.to_csv(top_edges_path, index=False)

    explanations_path = FILE_PATHS["notears_explanations"].replace(".txt", f"_{method}.txt")
    with open(explanations_path, "w", encoding="utf-8") as f:
        for line in all_explanations:
            f.write(line + "\n")
        
    return G, top_edges, explanations, fig_path

if __name__ == "__main__":
    G, top_edges, explanations, fig_path = notears_bio_dags()