"""Module for extracting bioclimatic variables for points."""

import pandas as pd
import numpy as np
import rioxarray
import requests
import os
import time
from tqdm import tqdm
from utils.logger_setup import setup_logging
import sys
from config import WORLDCLIM_CACHE_DIR, WORLDCLIM_RESOLUTION, WORLDCLIM_FILENAME_TEMPLATE, FILE_PATHS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = setup_logging()

class BioclimExtractor:
    def __init__(self, presence_file, background_file, output_file, worldclim_url, require_local=False):
        self.presence_file = presence_file
        self.background_file = background_file
        self.output_file = output_file
        self.worldclim_url = worldclim_url
        self.require_local = require_local
        self.bioclim_vars = {}
        self.use_local_only = False

    def download_bioclim_layers(self):
        logger.info("Downloading and loading bioclimatic layers...")
        os.makedirs(WORLDCLIM_CACHE_DIR, exist_ok=True)

        for i in tqdm(range(1, 20), desc="Downloading BIO layers", dynamic_ncols=True):
            var_name = f"BIO{i}"
            filename = WORLDCLIM_FILENAME_TEMPLATE.format(resolution=WORLDCLIM_RESOLUTION, index=i)
            local_file = os.path.join(WORLDCLIM_CACHE_DIR, filename)
            url = self.worldclim_url.format(i)

            if self.require_local or self.use_local_only:
                if os.path.exists(local_file):
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(local_file, masked=True)
                    logger.info(f"Loaded {var_name} from local backup")
                else:
                    logger.error(f"{var_name} not found in local backup folder.")
                continue

            try:
                if os.path.exists(local_file):
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(local_file, masked=True)
                    logger.info(f"Loaded {var_name} from cache")
                else:
                    logger.info(f"Downloading {var_name} from WorldClim")
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    with open(local_file, 'wb') as f:
                        f.write(response.content)
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(local_file, masked=True)
                    logger.info(f"Loaded {var_name}")
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"[🤔Warning] Failed to download {var_name}: {e}")
                self.use_local_only = True
                if os.path.exists(local_file):
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(local_file, masked=True)
                    logger.info(f"Fallback: loaded {var_name} from local backup")
                else:
                    logger.error(f"[❌Error] {var_name} unavailable (not online or local).")

        logger.info(f"Loaded {len(self.bioclim_vars)}/19 bioclimatic variables")
        return self.bioclim_vars

    def extract_variables_for_point(self, lat, lon):
        result = {}
        for var_name, data_array in self.bioclim_vars.items():
            try:
                val = data_array.sel(x=lon, y=lat, method="nearest").values[0]
                result[var_name] = float(val) if not np.ma.is_masked(val) else np.nan
            except Exception as e:
                logger.error(f"Error extracting {var_name} at ({lat}, {lon}): {e}")
                result[var_name] = np.nan
        return result

    def extract_variables_for_dataset(self):
        try:
            presence_df = pd.read_excel(self.presence_file)
            background_df = pd.read_excel(self.background_file)

            if presence_df.empty or background_df.empty:
                logger.error("Presence or background file is empty")
                return pd.DataFrame()

            lat_col = "decimalLatitude"
            lon_col = "decimalLongitude"
            if lat_col not in presence_df.columns or lon_col not in presence_df.columns:
                logger.error(f"Presence must contain {lat_col} and {lon_col}")
                return pd.DataFrame()
            if lat_col not in background_df.columns or lon_col not in background_df.columns:
                logger.error(f"Background must contain {lat_col} and {lon_col}")
                return pd.DataFrame()

            logger.info(f"Processing {len(presence_df)} points")
            logger.info(f"Processing {len(background_df)} points")

            if not self.bioclim_vars:
                self.download_bioclim_layers()
            if not self.bioclim_vars:
                logger.error("No bioclimatic variables were loaded")
                return pd.DataFrame()

            def process_rows(df, label):
                results = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting BIO for {label}", dynamic_ncols=True):
                    lat = row[lat_col]
                    lon = row[lon_col]
                    bio_vars = self.extract_variables_for_point(lat, lon)
                    row_data = row.to_dict()
                    row_data.update(bio_vars)
                    results.append(row_data)
                return pd.DataFrame(results)

            df_presence = process_rows(presence_df, "presence points")
            df_background = process_rows(background_df, "background points")
            combined_df = pd.concat([df_presence, df_background], ignore_index=True)

            combined_df = combined_df.dropna(subset=[f"BIO{i}" for i in range(1, 20)], how="all")


            combined_df.to_excel(self.output_file, index=False, engine="openpyxl")
            logger.info(f"⭐ Final combined data saved to {self.output_file}")
            return combined_df

        except Exception as e:
            logger.error(f"Error extracting bioclimatic variables: {str(e)}")
            raise
