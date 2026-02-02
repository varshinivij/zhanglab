import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


images = np.load("/Users/varshinivijay/Downloads/pubmedset/images/fold1/images.npy")
labels = np.load("/Users/varshinivijay/Downloads/pubmedset/images/fold1/types.npy")  # already strings


def np_to_pil(img_array):
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# --- Zero-shot pipeline ---
pipe = pipeline("zero-shot-image-classification", model="vinid/plip")

candidate_labels = list(np.unique(labels))  

# --- zero-shot classification ---
preds = []
for img_array in images:
    img = np_to_pil(img_array)
    img = transform(img)  # optional resizing
    result = pipe(img, candidate_labels=candidate_labels)
    best_label = max(result, key=lambda x: x['score'])['label']
    preds.append(best_label)

# --- Evaluate ---
true_labels = labels.tolist()  # convert numpy array of strings to list
acc = accuracy_score(true_labels, preds)
print(f"Accuracy: {acc:.4f}")

cm = confusion_matrix(true_labels, preds, labels=candidate_labels)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=candidate_labels, yticklabels=candidate_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Zero-Shot Classification Confusion Matrix")
plt.show()
