import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import measure, morphology
import json
import joblib

# Load model & scaler
model_data = joblib.load("glenoid_ai_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]

# Load baseline features
with open("baseline_features.json", "r") as f:
    baseline = json.load(f)

# Load images
image = sitk.ReadImage("image.nii.gz")
label = sitk.ReadImage("label.nii.gz")
image_data = sitk.GetArrayFromImage(image)
label_data = sitk.GetArrayFromImage(label)
spacing = label.GetSpacing()
segmented_voxels = label_data > 0

# Compute features
voxel_volume_mm3 = np.prod(spacing)
volume_mm3 = np.sum(segmented_voxels) * voxel_volume_mm3
voxel_count = np.sum(segmented_voxels)

coords = np.array(np.where(segmented_voxels))
min_coords = coords.min(axis=1)
max_coords = coords.max(axis=1)
bbox_size = (max_coords - min_coords + 1) * spacing[::-1]
bbox_volume = np.prod(bbox_size)

extent = voxel_count / np.prod(segmented_voxels.shape)
bbox_ratio_xy = bbox_size[0] / bbox_size[1] if bbox_size[1] != 0 else 0
bbox_ratio_xz = bbox_size[0] / bbox_size[2] if bbox_size[2] != 0 else 0
bbox_ratio_yz = bbox_size[1] / bbox_size[2] if bbox_size[2] != 0 else 0

verts, faces, _, _ = measure.marching_cubes(segmented_voxels, level=0, spacing=spacing[::-1])
surface_area_mm2 = measure.mesh_surface_area(verts, faces)
compactness = volume_mm3 / (surface_area_mm2 ** (3/2)) if surface_area_mm2 != 0 else 0
sphericity = (np.pi ** (1/3)) * ((6 * volume_mm3) ** (2/3)) / surface_area_mm2 if surface_area_mm2 != 0 else 0

masked_image = image_data[segmented_voxels]
intensity_mean = masked_image.mean()
intensity_std = masked_image.std()
intensity_min = masked_image.min()
intensity_max = masked_image.max()
intensity_median = np.median(masked_image)
intensity_range = intensity_max - intensity_min

eroded = morphology.binary_erosion(segmented_voxels)
dilated = morphology.binary_dilation(segmented_voxels)
erosion_ratio = np.sum(eroded) / voxel_count if voxel_count != 0 else 0
dilation_ratio = np.sum(dilated) / voxel_count if voxel_count != 0 else 0

surface_area_eroded = measure.mesh_surface_area(*measure.marching_cubes(eroded, level=0, spacing=spacing[::-1])[:2]) if np.any(eroded) else 0
surface_roughness = surface_area_mm2 / surface_area_eroded if surface_area_eroded != 0 else 0

com = np.mean(coords, axis=1)
center_of_mass_x, center_of_mass_y, center_of_mass_z = com

features = {
    "volume_mm3": volume_mm3,
    "voxel_count": voxel_count,
    "bbox_volume": bbox_volume,
    "extent": extent,
    "bbox_ratio_xy": bbox_ratio_xy,
    "bbox_ratio_xz": bbox_ratio_xz,
    "bbox_ratio_yz": bbox_ratio_yz,
    "surface_area_mm2": surface_area_mm2,
    "compactness": compactness,
    "sphericity": sphericity,
    "intensity_mean": intensity_mean,
    "intensity_std": intensity_std,
    "intensity_min": intensity_min,
    "intensity_max": intensity_max,
    "intensity_median": intensity_median,
    "intensity_range": intensity_range,
    "erosion_ratio": erosion_ratio,
    "dilation_ratio": dilation_ratio,
    "surface_roughness": surface_roughness,
    "center_of_mass_z": center_of_mass_z,
    "center_of_mass_y": center_of_mass_y,
    "center_of_mass_x": center_of_mass_x
}

# Compute diff and ratio
features_final = {}
for k in features:
    baseline_val = baseline[k]
    features_final[f"{k}_diff"] = features[k] - baseline_val
    features_final[f"{k}_ratio"] = features[k] / baseline_val if baseline_val != 0 else 0

# Convert to DataFrame for scaler
features_df = pd.DataFrame([features_final])

# Scale and predict
scaled_features = scaler.transform(features_df)
prediction = model.predict(scaled_features)

print("\nPredicted bone loss:", prediction[0])

