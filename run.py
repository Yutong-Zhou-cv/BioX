import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# === Step 0: Logger
from utils.logger_setup import setup_logging
logger = setup_logging()

# === Step 1: Species recognition
logger.info("\n===== STEP 1: Image-based Species Recognition =====")
from modules.species_recognition import batch_predict_species
from config import FILE_PATHS
batch_predict_species(FILE_PATHS["input_image_dir"])

# === Step 2: Retrive Presence data, generate Background and extract BIO
logger.info("\n===== STEP 2: GBIF + Background + BIO Extraction =====")
from modules.data_collection import data_processing_pipeline
data_processing_pipeline(input_file=FILE_PATHS["top_species_csv"], sampling_method="buffer")

# === Step 3: NOTEARS for learning DAGs within BIOs
logger.info("\n===== STEP 3: Causal Graph Learning (NOTEARS) =====")
from modules.notears_bios import notears_bio_dags
notears_bio_dags()

# === Step 4: DoWhy for causal inference
logger.info("\n===== STEP 4: Causal Inference (DoWhy) =====")
from modules.bio2p_dowhy import run_dowhy_batch_from_top_edges
run_dowhy_batch_from_top_edges()

# === Step 5: Explanation generation
logger.info("\n===== STEP 5: Explanation Generation =====")
from modules.explain import explain_significant_bios
explain_significant_bios()

logger.info("\n🎉 BioX Pipeline finished successfully!")
