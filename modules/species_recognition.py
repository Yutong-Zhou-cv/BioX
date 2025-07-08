import os
import sys
import pandas as pd
from bioclip import TreeOfLifeClassifier, Rank
from collections import defaultdict
from config import FILE_PATHS

class Logger(object):

    def __init__(self, logfile='log.txt'):
        self.terminal = sys.stdout
        self.log = open(logfile, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_image_files(folder_path, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return [f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in extensions]


def predict_species(classifier, image_path, rank=Rank.SPECIES, top_k=5):
    """
    Predict the top-k species for a given image.

    Args:
        classifier (TreeOfLifeClassifier): Pre-trained species classifier.
        image_path (str): Path to the image file.
        rank (Rank): Taxonomic rank to predict.
        top_k (int): Number of top predictions to return.

    Returns:
        list: List of prediction dictionaries sorted by score.
    """
    predictions = classifier.predict(image_path, rank)
    predictions.sort(key=lambda x: x['score'], reverse=True)
    return predictions[:top_k]


def record_class_top_species(predictions, image_file, class_top_species, top_k=5):
    """
    Record the top species per class based on prediction score.

    Args:
        predictions (list): List of prediction dictionaries.
        image_file (str): Image filename.
        class_top_species (dict): Dictionary tracking top species per class.
        top_k (int): Number of top predictions per class to retain.

    Returns:
        list: Flattened list of top-k predictions across classes.
    """
    results = []
    
    # Group predictions by class
    class_groups = defaultdict(list)
    for pred in predictions:
        class_groups[pred['class']].append(pred)

    for cls, preds in class_groups.items():
        preds.sort(key=lambda x: x['score'], reverse=True)
        top_preds = preds[:top_k]

        for pred in top_preds:
            results.append({
                'image': image_file,
                'class': cls,
                'order': pred.get('order', ''),
                'family': pred.get('family', ''),
                'genus': pred.get('genus', ''),
                'species': pred['species'],
                'score': pred['score']
            })

        # Update best species for the class
        top_species = preds[0]
        if top_species['score'] > class_top_species[cls]['score']:
            class_top_species[cls] = {
                'score': top_species['score'],
                'data': {
                    'Image': image_file,
                    'Class': cls,
                    'Scientific Name': top_species['species'],
                    'Confidence': top_species['score'] * 100
                }
            }

    return results


def save_results(pred_results, class_top_species):
    """
    Save detailed predictions and top species per class to CSV.

    Args:
        pred_results (list): List of prediction dictionaries.
        class_top_species (dict): Dictionary with top species per class.
    """
    pd.DataFrame(pred_results).to_csv(FILE_PATHS["species_recognition_csv"], index=False)
    top_list = [info['data'] for info in class_top_species.values() if info['data']]
    pd.DataFrame(top_list).to_csv(FILE_PATHS["top_species_csv"], index=False)


def batch_predict_species(folder_path, rank=Rank.SPECIES, top_k=5):
    """
    Batch process images for species recognition and output results.

    Args:
        folder_path (str): Directory path containing image files.
        rank (Rank): Taxonomic rank to predict.
        top_k (int): Number of top predictions to retain.
    """
    classifier = TreeOfLifeClassifier()
    image_files = get_image_files(folder_path)
    os.makedirs(FILE_PATHS["species_recognition_dir"], exist_ok=True)

    prediction_results = []
    class_top_species = defaultdict(lambda: {'score': -1, 'data': None})

    print(f"===Step 1: Species Recognition===")

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}...")

        try:
            predictions = predict_species(classifier, image_path, rank, top_k)

            if not predictions:
                print(f"[🤔Warning] No predictions returned for image {image_file}.\n")
                continue

            top_score = predictions[0]['score']
            if top_score < 0.8:
                print(f"[🤔Warning] Low confidence ({top_score:.2f}) for species recognition in image {image_file}! Results may be unreliable.\n")

            with open(FILE_PATHS["species_recognition_csv"], "a") as f:
                f.write(f"{image_file}," + ",".join(
                    f"{pred['species']},{pred['score']:.4f}" for pred in predictions) + "\n")

            pred_details = record_class_top_species(predictions, image_file, class_top_species, top_k)
            prediction_results.extend(pred_details)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    save_results(prediction_results, class_top_species)

    species_recognition_csv = FILE_PATHS["species_recognition_csv"]
    top_species_csv = FILE_PATHS["top_species_csv"]
    print(f"⭐ Results saved to '{species_recognition_csv}' and '{top_species_csv}'.")


if __name__ == "__main__":
    sys.stdout = Logger('log.txt')

    folder_path = FILE_PATHS["input_image_dir"]
    batch_predict_species(folder_path)
