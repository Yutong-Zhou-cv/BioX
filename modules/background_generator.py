"""Module for generating background (pseudo-absence) points."""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import requests
import zipfile
import time
import math
from utils.logger_setup import setup_logging
import sys
from config import LAND_MASK_URL, LAND_MASK_CACHE_PATH


# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get logger from the unified logging setup
logger = setup_logging()

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of the first point (in decimal degrees)
        lat2, lon2: Coordinates of the second point (in decimal degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def create_bounding_box(points_df, buffer_degree=1.0):
    """
    Create a bounding box around a set of points with a buffer.
    
    Args:
        points_df: DataFrame containing latitude and longitude columns
        buffer_degree: Buffer size in degrees to add to the bounding box
        
    Returns:
        Dictionary with min/max latitude and longitude
    """
    min_lat = points_df['decimalLatitude'].min() - buffer_degree
    max_lat = points_df['decimalLatitude'].max() + buffer_degree
    min_lon = points_df['decimalLongitude'].min() - buffer_degree
    max_lon = points_df['decimalLongitude'].max() + buffer_degree
    
    # Ensure coordinates are within valid ranges
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    
    return {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon
    }

def points_to_gdf(df, lat_col='decimalLatitude', lon_col='decimalLongitude'):
    """
    Convert a DataFrame with latitude and longitude columns to a GeoDataFrame.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of the latitude column
        lon_col: Name of the longitude column
        
    Returns:
        GeoDataFrame with Point geometry
    """
    import geopandas as gpd
    
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def is_on_land(lon, lat, land_mask_gdf):
    """
    Check if a point is on land using a land mask.
    
    Args:
        lon, lat: Coordinates of the point
        land_mask_gdf: GeoDataFrame with land polygons
        
    Returns:
        Boolean indicating if the point is on land
    """
    point = Point(lon, lat)
    return land_mask_gdf.contains(point).any()

def standardize_features(df, feature_cols):
    """
    Standardize features to have zero mean and unit variance.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with standardized features and scaling parameters
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df_std = df.copy()
    
    # Extract the features as a numpy array
    features = df[feature_cols].values
    
    # Fit the scaler and transform the features
    features_std = scaler.fit_transform(features)
    
    # Update the DataFrame with standardized values
    for i, col in enumerate(feature_cols):
        df_std[col] = features_std[:, i]
    
    # Return both the standardized DataFrame and the scaler for inverse transformation
    return df_std, scaler


class BackgroundGenerator:
    """Class for generating background (pseudo-absence) points."""
    
    def __init__(self, presence_file, output_file, params):
        """
        Initialize the background generator.
        
        Args:
            presence_file: Path to the presence data Excel file
            output_file: Path to save the generated background points
            params: Dictionary of parameters for background point generation
        """
        self.presence_file = presence_file
        self.output_file = output_file
        self.params = params
    
    def get_land_mask(self):
        """
        Get a land mask for filtering background points.
        
        Returns:
            GeoDataFrame with land polygons
        """
        logger.info("Obtaining land mask...")
        
        
        if os.path.exists(LAND_MASK_CACHE_PATH):
            logger.info("Loading cached land mask.")
            return gpd.read_file(LAND_MASK_CACHE_PATH)
        
        try:
            # Create a temporary directory for downloaded files
            os.makedirs("temp", exist_ok=True)
            zip_path = "temp/ne_land.zip"
            
            # Download the file
            response = requests.get(LAND_MASK_URL)
            response.raise_for_status()
            
            # Save to a temporary zip file
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the shapefile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp")
            
            logger.info("Land mask obtained successfully")
            return gpd.read_file(LAND_MASK_CACHE_PATH)
            
        except Exception as e:
            logger.error(f"Error obtaining land mask: {str(e)}")
            logger.warning("[🤔Warning]Proceeding without land mask - background points may include water areas")
            return None
    
    def buffer_method(self, presence_df, species_name):
        """
        Generate background points using the buffer method.
        
        Args:
            presence_df: DataFrame with presence points for a species
            species_name: Name of the species
            
        Returns:
            DataFrame with generated background points
        """
        logger.info(f"Generating background points for {species_name} using buffer method")
        
        # Get parameters
        buffer_degree = self.params["buffer_degree"]
        n_background_ratio = self.params["n_background_ratio"]
        min_distance_km = self.params["min_distance_km"]
        
        # Create bounding box
        bbox = create_bounding_box(presence_df, buffer_degree)
        
        # Calculate number of background points to generate
        n_presence = len(presence_df)
        n_background = int(n_presence * n_background_ratio)
        
        logger.info(f"Target: {n_background} background points for {n_presence} presence points")
        
        # Get land mask for filtering
        land_mask = self.get_land_mask()
        
        # Generate random points within the bounding box
        # We'll generate more than needed to account for filtering
        safety_factor = 3  # Generate 3x more points than needed initially
        background_points = []
        points_generated = 0
        
        max_attempts = n_background * safety_factor * 2  # Limit attempts to avoid infinite loops
        attempts = 0
        
        with tqdm(total=n_background, desc=f"Generating points for {species_name}", dynamic_ncols=True) as pbar:
            while points_generated < n_background and attempts < max_attempts:
                attempts += 1
                
                # Generate a random point within the bounding box
                lat = np.random.uniform(bbox["min_lat"], bbox["max_lat"])
                lon = np.random.uniform(bbox["min_lon"], bbox["max_lon"])
                
                # Check if the point is far enough from any presence point
                is_far_enough = True
                for _, row in presence_df.iterrows():
                    distance = haversine_distance(
                        lat, lon, row["decimalLatitude"], row["decimalLongitude"]
                    )
                    if distance < min_distance_km:
                        is_far_enough = False
                        break
                
                # Check if the point is on land (if land mask is available)
                on_land = True
                if land_mask is not None:
                    point = Point(lon, lat)
                    on_land = land_mask.contains(point).any()
                
                # Add the point if it passes all filters
                if is_far_enough and on_land:
                    background_points.append({
                        "scientificName": species_name,
                        "decimalLatitude": lat,
                        "decimalLongitude": lon,
                        "Presence": 0  # 0 for background/absence
                    })
                    points_generated += 1
                    pbar.update(1)
                
                # Break if we have enough points
                if points_generated >= n_background:
                    break
        
        if points_generated < n_background:
            logger.warning( 
                f"[🤔Warning]Could only generate {points_generated}/{n_background} background points "
                f"for {species_name} after {attempts} attempts"
            )
        
        # Convert to DataFrame
        if background_points:
            return pd.DataFrame(background_points)
        else:
            return pd.DataFrame(columns=["scientificName", "decimalLatitude", "decimalLongitude", "Presence"])
    
    
    def generate_background_points(self, bio_vars_df=None):
        """
        Generate background points for all species.
        
        Args:
            bio_vars_df: Optional DataFrame with bioclimatic variables for presence points
        
        Returns:
            DataFrame with generated background points for all species
        """
        try:
            # Read presence data
            presence_df = pd.read_excel(self.presence_file)
            
            if presence_df.empty:
                logger.error("No presence data found")
                return pd.DataFrame()
            
            # Get unique species
            species_list = presence_df["scientificName"].unique()
            
            # Generate background points for each species
            all_background = []
            
            for species in species_list:
                # Get presence points for this species
                species_presence = presence_df[presence_df["scientificName"] == species]
                
                background_df = self.buffer_method(species_presence, species)
                
                if not background_df.empty:
                    all_background.append(background_df)
                    
                # Sleep briefly to avoid resource overuse
                time.sleep(0.5)
            
            # Combine all background points
            if all_background:
                combined_df = pd.concat(all_background, ignore_index=True)
                logger.info(f"Total background points generated: {len(combined_df)}")
                
                # Save to Excel
                combined_df.to_excel(self.output_file, index=False, engine="openpyxl")
                logger.info(f"⭐ Background points saved to {self.output_file}")
                
                return combined_df
            else:
                logger.warning("[🤔Warning] No background points were generated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error generating background points: {str(e)}")
            raise