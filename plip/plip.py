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

"""
Tasks to complete:
1. Classification task 
2. Zero shot learning 
3. More benchmark results 
"""
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
    return out

def parse_out(file_name):
    annotations_csv = Path("/Users/varshinivijay/Downloads/histopathology_images") / "annotations.csv"
    with annotations_csv.open('r') as f:
        name = None
        while name != file_name:
            line = f.readline()
            name = line.strip().split(',')[0]
        line = f.readline()
        target = line.strip().split(',')[1]
    return target

        
def process_dir():
    data = {"true_label": [], "predicted_label": []}
    total, accuracy = 0, 0
    abrv = {"Hyperplastic Polyp": "HP",
                      "Sessile Serrated Adenoma": "SSA"}
    for file_path in DIR_PATH.iterdir():
        if file_path.is_file():
            file_path = str(file_path)
            if ".png" not in file_path:
                continue
            total += 1
            out = abrv[classify(file_path, ["Hyperplastic Polyp", "Sessile Serrated Adenoma"])]
            target = parse_out(file_path.split('/')[-1])
            data["true_label"].append(target)
            data["predicted_label"].append(out)
            print(f"Target: {target} | Output: {out}")
            if (out == target):
                accuracy += 1
    print(f"The accuracy is {accuracy/total}")
    return data

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(df):
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=["HP", "SSA"])
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["HP", "SSA"], yticklabels=["HP", "SSA"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix(process_dir())
