import os
import sys
import pandas as pd
from config import GBIF_PARAMS, BACKGROUND_PARAMS, WORLDCLIM_BASE_URL, FILE_PATHS
from gbif_extractor import GBIFExtractor
from background_generator import BackgroundGenerator
from bioclim_extractor import BioclimExtractor
from utils.logger_setup import setup_logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = setup_logging()

def data_processing_pipeline(input_file=None, sampling_method=None):
    """
    Run the complete data processing pipeline for species presence/background collection.

    Args:
        input_file (str): Path to Excel file with species names.
        sampling_method (str): "buffer" or "env_stratified".
    """
    try:

        os.makedirs(FILE_PATHS["data_dir"], exist_ok=True)

        presence_file = FILE_PATHS["presence_output"]
        background_file = FILE_PATHS["background_buffer_output"]  # or env_stratified
        final_output = FILE_PATHS["final_dataset_output"]

        # === Step 1: Extract Presence Records ===
        logger.info("=== Step 1: A-Extracting Presence Records from GBIF ===")
        gbif = GBIFExtractor(input_file, presence_file, GBIF_PARAMS)
        presence_df = gbif.process_and_save()

        if presence_df.empty:
            logger.error("[❌Error] No presence data found.")
            return

        # === Step 2: Generate Background Points ===
        logger.info("=== Step 2: B-Generating Background Points ===")
        bg = BackgroundGenerator(presence_file, background_file, BACKGROUND_PARAMS)

        if sampling_method == "buffer":
            background_df = bg.generate_background_points()
        else:
            # Step 2.1: Extract BIO for presence (used for stratified)
            logger.info("\nExtracting BIO variables for presence points...")
            temp_presence_bio = FILE_PATHS["temp_presence_bio"]
            bio_extractor = BioclimExtractor(presence_file, temp_presence_bio, WORLDCLIM_BASE_URL, require_local=True)

            try:
                presence_bio_df = bio_extractor.extract_variables_for_dataset()
            except Exception as e:
                logger.error("[❌Error] Failed to extract BIO variables. Please manually download BIO .tif files from WorldClim")
                logger.error("       and place them in the ./temp/ directory.")
                logger.error(f"Error details: {str(e)}")
                return

            background_df = bg.generate_background_points(presence_bio_df)

        if background_df.empty:
            logger.error("[❌Error] No background data generated.")
            return

        # === Step 3: Extract BIO Variables for all points ===
        logger.info("=== Step 2: C-Extracting BIO Variables ===")
        if sampling_method == "buffer" or not any(col.startswith("BIO") for col in background_df.columns):
            bioclim_extractor = BioclimExtractor(presence_file, background_file, final_output, WORLDCLIM_BASE_URL, require_local=True)
            try:
                bioclim_extractor.extract_variables_for_dataset()
            except Exception as e:
                logger.error("[❌Error] Failed to extract BIO variables. Please manually download BIO .tif files from WorldClim")
                logger.error("       and place them in the ./temp/ directory.")
                logger.error(f"Error details: {str(e)}")
                return
        else:
            presence_bio_df = pd.read_excel(temp_presence_bio)
            final_df = pd.concat([presence_bio_df, background_df], ignore_index=True)
            final_df.to_excel(final_output, index=False, engine="openpyxl")
            logger.info(f"Combined dataset saved to {final_output}")

        logger.info("Data processing pipline completed successfully!")

    except Exception as e:
        logger.error(f"[❌Error] Pipeline execution failed: {e}")


if __name__ == "__main__":
    data_processing_pipeline(input_file = FILE_PATHS["top_species_csv"], sampling_method="buffer")