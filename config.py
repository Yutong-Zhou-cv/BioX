"""Configuration settings for the species occurrence pipeline."""

# GBIF API parameters
GBIF_PARAMS = {
    "hasCoordinate": True,
    "basisOfRecord": "HUMAN_OBSERVATION",
    "limit": 300,
    "sample_per_year": 40,
    "start_year": 2001,
    "end_year": 2025
}

# Background points generation parameters
BACKGROUND_PARAMS = {
    "sampling_method": "buffer",  # "buffer" or "env_stratified"???
    "n_background_ratio": 2.0,    # Number of background points relative to presence points
    "buffer_degree": 2.0,         # Geographic expansion from bounding box (degrees)
    "min_distance_km": 5.0,       # Minimum distance from any presence point (km)
    "env_n_clusters": 10,         # Number of clusters for KMeans (stratified sampling)
    "env_points_per_cluster": 50  # Number of background points per cluster center
}

# Natural Earth
LAND_MASK_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
LAND_MASK_CACHE_PATH = "temp/ne_10m_land.shp"

# WorldClim data
WORLDCLIM_BASE_URL = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio_{}.tif"
WORLDCLIM_RESOLUTION = "2.5m"  # 2.5 arc-minutes (~5km)
WORLDCLIM_CACHE_DIR = "temp/wc2.1_2.5m_bio" 
WORLDCLIM_FILENAME_TEMPLATE = "wc2.1_{resolution}_bio_{index}.tif"

# File paths 
FILE_PATHS = {
    # Step 1: Species Recognition
    "input_image_dir": "test/",
    "species_recognition_dir": "Output/1_Recognition",
    "species_recognition_csv": "Output/1_Recognition/1.1_Species_Recognition.csv",
    "top_species_csv": "Output/1_Recognition/1.2_Top_Species_Per_Class.csv",

    # Step 2: Data Collection
    "data_dir": "Output/2_Data",
    "presence_output": "Output/2_Data/2.1_Species_Presence_Records.xlsx",
    "background_buffer_output": "Output/2_Data/2.2_Background_Point_buffer.xlsx",
    "background_env_stratified_output": "Output/2_Data/2.2_Background_Point_env_stratified.xlsx",
    "temp_presence_bio": "Output/2_Data/temp_presence_with_bio.xlsx",
    "final_dataset_output": "Output/2_Data/2.3_Final_Data_with_Bioclim.xlsx",

    # Step 3: Notears
    "DAGs_output": "Output/3_NOTEARS",
    "notears_output": "Output/3_NOTEARS/3.1_Notears_Output.xlsx",
    "notears_top_edges": "Output/3_NOTEARS/3.2_Notears_Top_Edges.csv",
    "notears_explanations": "Output/3_NOTEARS/3.3_Notears_BIOsExplanations.txt",  

    # Step 4: DoWhy
    "DoWhy_output": "Output/4_DoWhy",

    # Step 5: Explaination Generation
    "Exp_output": "Output/5_Explaination",

    # Others
    "local_backup_dir": "temp",
    "output_dir": "Output"
}

# NOTEARS causal discovery method: "linear" or "nonlinear"
NOTEARS_PARAMS = {
    "method": "nonlinear"  # "nonlinear" or "linear"
}

DOWHY_PARAMS = {
    "edges": "top5",  # "top5" or "all"
    "method": "backdoor.linear_regression",  # "backdoor.linear_regression" or "backdoor.propensity_score_stratification"
    "confounder_mode": "dag"  # "dag" is confounders from the DAGs, or "all"
}

EXPLAIN_PARMS = {
    "explanation_mode": "llm",  # "llm" or "rule"
    "llm_model": "llama3.3:70b"
}

