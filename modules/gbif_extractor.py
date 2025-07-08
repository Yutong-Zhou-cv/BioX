"""Extract data from GBIF."""

import pandas as pd
from pygbif import occurrences
import time
import sys
from tqdm import tqdm
import os
from config import FILE_PATHS, GBIF_PARAMS

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_setup import setup_logging

# Get logger from the unified logging setup
logger = setup_logging()

class GBIFExtractor:
    """
    Extract occurrence records for a list of species using GBIF API.
    """

    def __init__(self, input_file, output_file, gbif_params):
        """
        Initialize the GBIF extractor.

        Args:
            input_file (str): Path to the file with species names (.csv or .xlsx).
            output_file (str): Path to save extracted presence records.
            gbif_params (dict): Parameters for GBIF API queries.
        """
        self.input_file = input_file
        self.output_file = FILE_PATHS["presence_output"]
        self.gbif_params = gbif_params

    def read_species_list(self):
        """
        Read species names from the input file.

        Returns:
            list: List of unique species names.
        """
        try:
            if self.input_file.endswith('.csv'):
                df = pd.read_csv(self.input_file)
            else:
                df = pd.read_excel(self.input_file, engine='openpyxl')
        except Exception as e:
            logger.error(f"[❌Error] Failed to read input file: {e}")
            raise

        if "Scientific Name" not in df.columns:
            raise ValueError("Input file must contain a 'Scientific Name' column")
        return df["Scientific Name"].dropna().unique().tolist()

    def query_gbif(self, species_name):
        """
        Query GBIF API for a given species.

        Args:
            species_name (str): Scientific name.

        Returns:
            pd.DataFrame: Presence records.
        """
        logger.info(f"Querying GBIF for species: {species_name}")
        params = {"scientificName": species_name, **self.gbif_params}

        try:
            records = []
            limit = GBIF_PARAMS["limit"]
            sample_per_year = GBIF_PARAMS["sample_per_year"]
            years = list(range(GBIF_PARAMS["start_year"], GBIF_PARAMS["end_year"] + 1))

            for year in years:
                logger.info(f"{species_name} Data retrieval for year {year}")
                params.update({"year": year, "limit": limit, "offset": 0})
                response = occurrences.search(**params)
                batch = response.get("results", [])
                if not batch:
                    continue

                for r in batch:
                    lat = r.get("decimalLatitude")
                    lon = r.get("decimalLongitude")

                    if lat is None or lon is None:
                        continue

                    records.append({
                        "scientificName": r.get("scientificName", species_name),
                        "decimalLatitude": lat,
                        "decimalLongitude": lon,
                        "year": year,
                        "country": r.get("country"),
                        "countryCode": r.get("countryCode"),
                    })
                time.sleep(1)

            df = pd.DataFrame(records)
            if df.empty:
                return df

            df["year"] = df["year"].astype(int)
            years = list(range(GBIF_PARAMS["start_year"], GBIF_PARAMS["end_year"] + 1))
            sample_per_year = GBIF_PARAMS["sample_per_year"]

            df_sampled = pd.concat([
                df[df["year"] == y].sample(n=min(sample_per_year, len(df[df["year"] == y])), random_state=42)
                for y in years if y in df["year"].values
            ], ignore_index=True)

            return df_sampled
        except Exception as e:
            logger.warning(f"[❌Error] Failed to fetch data for {species_name}: {e}")
            return pd.DataFrame()

    def extract_all_species(self):
        """
        Extract presence records for all species in the list.

        Returns:
            pd.DataFrame: Combined presence records.
        """
        all_species = self.read_species_list()
        all_records = []

        for species in tqdm(all_species, desc="Querying GBIF", dynamic_ncols=True):
            df = self.query_gbif(species)
            if isinstance(df, pd.DataFrame) and not df.empty and df.columns.size > 0 and df.dropna(how='all').shape[0] > 0:
                all_records.append(df)
            time.sleep(1)

        if all_records:
            filtered = [df for df in all_records if not df.empty and df.columns.size > 0 and df.dropna(how='all').shape[0] > 0]
            return pd.concat(filtered, ignore_index=True)
        return pd.DataFrame()

    def process_and_save(self):
        """
        Run the full pipeline: fetch, process, and save GBIF presence data.

        Returns:
            pd.DataFrame: Final presence dataset with 'Presence' label.
        """
        presence_df = self.extract_all_species()
        if presence_df.empty:
            logger.warning("[❌Error] No presence data found.")
            return presence_df

        presence_df["Presence"] = 1

        try:
            if self.output_file.endswith('.csv'):
                presence_df.to_csv(self.output_file, index=False)
            else:
                presence_df.to_excel(self.output_file, index=False, engine='openpyxl')
        except Exception as e:
            logger.error(f"[❌Error] Failed to save presence file: {e}")
            raise

        logger.info(f"⭐ Presence data saved to {self.output_file}")
        return presence_df