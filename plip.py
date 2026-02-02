#pip install -U transformers
from pathlib import Path
#collab tutorial 
# a pipeline as a high-level helper that creates text and image embeddings and runs inference
from transformers import pipeline
# # Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

DIR_PATH = Path("/Users/varshinivijay/Downloads/histopathology_images")

# processor = AutoProcessor.from_pretrained("vinid/plip")
# model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
def classify(img_file, labels):
    pipe = pipeline("zero-shot-image-classification", model="vinid/plip")
    results = pipe(
        img_file,
        candidate_labels=labels,
    )
    out = None
    max_score = float('-inf')
    for result in results:
        if result['score'] > max_score:
            max_score = result['score']
            out = result['label']

    # best_label = max(results, key=lambda x: x['score'])
    return out, results

def parse_out(file_name):
    annotations_csv = Path("/Users/varshinivijay/Downloads/histopathology_images") / "annotations.csv"
    with open(annotations_csv, 'r') as f:
        name = None
        while name != file_name:
            line = f.readline()
            name = line.strip().split(',')[0]
        line = f.readline()
        target = line.strip().split(',')[1]
    return target

        
def process_dir():
    for file_path in DIR_PATH.iterdir():
        if file_path.is_file():
            try:
                with file_path.open('r') as f:
                    classify(file_path, ["Hyperplastic Polyp", "Sessile Serrated Adenoma"])
            except IOError as e:
                print(f"An error occurred with your file:{e}")

if __name__ == "__main__":
    
