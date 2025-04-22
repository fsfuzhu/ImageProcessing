import torch
from transformers import ViTModel, ViTImageProcessor
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load fine-tuned ViT Transformer Model
def load_vit_model():
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    #TODO change the model path
    state_dict = torch.load("fine_tuned_resnet50.pth", map_location=device,weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model

# ViT Feature Extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Image Preprocessing Function
def preprocess_image(image):
    """Preprocesses an image for ViT feature extraction."""
    image = Image.fromarray(image).convert("RGB")  # Ensure correct format
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device)

# Extract features from ViT model
def extract_features(image, model):
    """Extracts features from an image tensor using the ViT model."""
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        features = model(image_tensor).pooler_output
    return features.cpu().numpy().flatten()

# Extract red mask from the ground truth image
def extract_red_mask(image_path):
    """Extracts the red mask from the ground truth image."""
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2  # Combine both red masks
    return red_mask

# Apply the red mask to retain only relevant regions
def apply_mask(image_path, mask):
    """Applies the red mask to an image to retain relevant regions."""
    image = cv2.imread(image_path)  # Load as 3-channel image
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Resize the mask to match the image size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to 3 channels if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    masked_image = cv2.bitwise_and(image, mask)  # Apply mask
    return masked_image

# Compute Cosine Similarity
def compare_images(segmented_path, ground_truth_path, model):
    """Compares a segmented image with ground truth using cosine similarity."""
    if not os.path.exists(ground_truth_path):
        print(f"Missing ground truth: {ground_truth_path}")
        return None

    # Extract red mask from ground truth
    red_mask = extract_red_mask(ground_truth_path)

    # Apply mask to both images
    segmented_image = cv2.imread(segmented_path)
    ground_truth_image = cv2.imread(ground_truth_path)

    # Ensure both images are resized to the same shape
    if segmented_image.shape[:2] != ground_truth_image.shape[:2]:
        ground_truth_image = cv2.resize(ground_truth_image, (segmented_image.shape[1], segmented_image.shape[0]))

    # Apply mask after resizing
    masked_segmented = apply_mask(segmented_path, red_mask)
    masked_ground_truth = apply_mask(ground_truth_path, red_mask)

    # Extract features
    seg_features = extract_features(masked_segmented, model)
    gt_features = extract_features(masked_ground_truth, model)

    # Compute cosine similarity
    similarity = cosine_similarity([seg_features], [gt_features])[0][0]
    return similarity

# Main evaluation function
model = load_vit_model()

segmented_folder = "output"
ground_truth_folder = "Dataset_2\ground_truths"

results = []
threshold = 0.9  # Adjust this threshold based on accuracy requirements

for filename in os.listdir(segmented_folder):
    if filename.lower() != 'image_0848' and filename.lower().endswith((".jpg", ".png")):
        segmented_path = os.path.join(segmented_folder, filename)
        ground_truth_path = os.path.join(ground_truth_folder, os.path.splitext(filename)[0] + ".png")

        similarity = compare_images(segmented_path, ground_truth_path, model)

        if similarity is not None:
            accuracy = 1 if similarity >= threshold else 0
            results.append([filename, similarity, accuracy])

# Create DataFrame
metrics_df = pd.DataFrame(results, columns=["Image", "Cosine Similarity", "Accuracy"])

# Compute overall accuracy and similarity
overall_accuracy = metrics_df['Cosine Similarity'].mean() * 100
overall_similarity = metrics_df['Accuracy'].mean() * 100

# Display results
print(metrics_df)
print(f"\nOverall Segmentation Accuracy: {overall_accuracy:.2f}%")
print(f"\nOverall Segmentation Similarity: {overall_similarity:.2f}%")
