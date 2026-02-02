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

def process_dir():
    for file_path in DIR_PATH.iterdir():
        if file_path.is_file():
            try:
                with file_path.open('r') as f:
                    classify(file_path, labels)
            except IOError as e:
                print(f"An error occurred with your file:{e}")

if __name__ == "__main__":
    out, results = classify("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png", ["Hyperplastic Polyp", "Sessile Serrated Adenoma"])
    print(out)
