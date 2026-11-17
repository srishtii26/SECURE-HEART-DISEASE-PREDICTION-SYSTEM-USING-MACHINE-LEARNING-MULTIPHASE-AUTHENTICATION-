from pathlib import Path
from PIL import Image

from ecg.ecg_predict import predict_ecg
from ecg.ecg_predict import get_ecg_interpretation

root = Path("dataset/ecg/cardiac_images/ECG_DATA/test")

image_path = next(root.rglob("*.jpg"))

print("Testing image:")
print(image_path)

image = Image.open(image_path)

result = predict_ecg(image)

print()
print("Predicted class:", result["predicted_class"])
print("Result:", result["class_name"])
print("Confidence:", f'{result["confidence"] * 100:.2f}%')

interpretation = get_ecg_interpretation(
    result["class_name"],
    result["confidence"]
)

print()
print("UI interpretation:")
print(interpretation)
