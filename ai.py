import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import ndimage
from skimage import measure, morphology
import joblib
import warnings
warnings.filterwarnings('ignore')

class GlenoidBoneLossAI:
    def __init__(self, dataset_path="./glenoid/dataset/"):
        self.dataset_path = dataset_path
        self.patient_data = {}
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_patient_data(self):
        """Load all patient data from the dataset directory"""
        print("Loading patient data...")
        
        for patient_folder in os.listdir(self.dataset_path):
            if patient_folder.startswith('patient'):
                patient_path = os.path.join(self.dataset_path, patient_folder)
                
                image_path = os.path.join(patient_path, 'image.nii.gz')
                label_path = os.path.join(patient_path, 'label.nii.gz')
                
                if os.path.exists(image_path) and os.path.exists(label_path):
                    try:
                        # Load NIFTI files
                        img = sitk.ReadImage(image_path)
                        lbl = sitk.ReadImage(label_path)
                        
                        img_arr = sitk.GetArrayFromImage(img)
                        lbl_arr = sitk.GetArrayFromImage(lbl)
                        
                        # Validate shapes match
                        assert img_arr.shape == lbl_arr.shape, f"Shape mismatch for {patient_folder}"
                        
                        # Get voxel spacing
                        spacing = img.GetSpacing()
                        
                        self.patient_data[patient_folder] = {
                            'image': img_arr,
                            'label': lbl_arr,
                            'spacing': spacing,
                            'shape': img_arr.shape,
                            'image_sitk': img,
                            'label_sitk': lbl
                        }
                        
                        print(f"✓ Loaded {patient_folder}: {img_arr.shape}")
                        
                    except Exception as e:
                        print(f"✗ Error loading {patient_folder}: {str(e)}")
                        
        print(f"Successfully loaded {len(self.patient_data)} patients")
        return self.patient_data
    
    def extract_glenoid_features(self, image, label, spacing):
        """Extract comprehensive features from glenoid region"""
        features = {}
        
        # Convert spacing to numpy array (z, y, x order for SimpleITK)
        voxel_volume = np.prod(spacing)
        
        # Get glenoid mask (assuming label value 1 for glenoid)
        glenoid_mask = (label == 1).astype(np.uint8)
        
        if np.sum(glenoid_mask) == 0:
            print("Warning: No glenoid region found in label")
            return None
            
        # 1. VOLUMETRIC FEATURES
        glenoid_voxel_count = np.sum(glenoid_mask)
        features['volume_mm3'] = glenoid_voxel_count * voxel_volume
        features['voxel_count'] = glenoid_voxel_count
        
        # 2. GEOMETRIC FEATURES
        # Bounding box analysis
        coords = np.where(glenoid_mask)
        if len(coords[0]) > 0:
            bbox_dims = [
                np.max(coords[0]) - np.min(coords[0]) + 1,
                np.max(coords[1]) - np.min(coords[1]) + 1,
                np.max(coords[2]) - np.min(coords[2]) + 1
            ]
            features['bbox_volume'] = np.prod(bbox_dims) * voxel_volume
            features['extent'] = features['volume_mm3'] / features['bbox_volume']
            features['bbox_ratio_xy'] = bbox_dims[1] / bbox_dims[2] if bbox_dims[2] > 0 else 0
            features['bbox_ratio_xz'] = bbox_dims[0] / bbox_dims[2] if bbox_dims[2] > 0 else 0
            features['bbox_ratio_yz'] = bbox_dims[0] / bbox_dims[1] if bbox_dims[1] > 0 else 0
        
        # 3. SURFACE AREA AND SHAPE ANALYSIS
        try:
            # Calculate surface area using marching cubes
            verts, faces, _, _ = measure.marching_cubes(glenoid_mask, level=0.5, spacing=spacing)
            features['surface_area_mm2'] = measure.mesh_surface_area(verts, faces)
            
            # Compactness (sphere-like measure)
            if features['surface_area_mm2'] > 0:
                features['compactness'] = (36 * np.pi * features['volume_mm3']**2) ** (1/3) / features['surface_area_mm2']
            else:
                features['compactness'] = 0
                
            # Sphericity
            equivalent_diameter = 2 * (3 * features['volume_mm3'] / (4 * np.pi)) ** (1/3)
            features['sphericity'] = equivalent_diameter / np.max(bbox_dims) if np.max(bbox_dims) > 0 else 0
            
        except Exception as e:
            print(f"Warning: Surface analysis failed: {e}")
            features['surface_area_mm2'] = 0
            features['compactness'] = 0
            features['sphericity'] = 0
        
        # 4. INTENSITY FEATURES
        glenoid_intensities = image[glenoid_mask > 0]
        if len(glenoid_intensities) > 0:
            features['intensity_mean'] = np.mean(glenoid_intensities)
            features['intensity_std'] = np.std(glenoid_intensities)
            features['intensity_min'] = np.min(glenoid_intensities)
            features['intensity_max'] = np.max(glenoid_intensities)
            features['intensity_median'] = np.median(glenoid_intensities)
            features['intensity_range'] = features['intensity_max'] - features['intensity_min']
        
        # 5. MORPHOLOGICAL FEATURES
        # Erosion and dilation to analyze shape complexity
        kernel = np.ones((3,3,3))
        eroded = ndimage.binary_erosion(glenoid_mask, structure=kernel)
        dilated = ndimage.binary_dilation(glenoid_mask, structure=kernel)
        
        features['erosion_ratio'] = np.sum(eroded) / np.sum(glenoid_mask) if np.sum(glenoid_mask) > 0 else 0
        features['dilation_ratio'] = np.sum(dilated) / np.sum(glenoid_mask) if np.sum(glenoid_mask) > 0 else 0
        
        # Surface roughness approximation
        features['surface_roughness'] = features['dilation_ratio'] - features['erosion_ratio']
        
        # 6. SYMMETRY FEATURES (for bilateral comparison)
        # Center of mass
        com = ndimage.center_of_mass(glenoid_mask)
        features['center_of_mass_z'] = com[0] * spacing[2]
        features['center_of_mass_y'] = com[1] * spacing[1]
        features['center_of_mass_x'] = com[2] * spacing[0]
        
        return features
    
    def detect_bilateral_cases(self):
        """Detect which patients have bilateral data"""
        bilateral_patients = []
        unilateral_patients = []
        
        for patient_id, data in self.patient_data.items():
            label = data['label']
            unique_labels = np.unique(label)
            
            # Check if we have multiple glenoid regions (bilateral)
            # Assuming labels: 0=background, 1=left glenoid, 2=right glenoid
            # or similar labeling scheme
            glenoid_labels = [l for l in unique_labels if l > 0]
            
            if len(glenoid_labels) >= 2:
                bilateral_patients.append(patient_id)
            else:
                unilateral_patients.append(patient_id)
                
        return bilateral_patients, unilateral_patients
    
    def extract_all_features(self):
        """Extract features from all patients"""
        print("Extracting features from all patients...")
        
        bilateral_patients, unilateral_patients = self.detect_bilateral_cases()
        print(f"Bilateral patients: {len(bilateral_patients)}")
        print(f"Unilateral patients: {len(unilateral_patients)}")
        
        all_features = []
        
        for patient_id, data in self.patient_data.items():
            print(f"Processing {patient_id}...")
            
            image = data['image']
            label = data['label']
            spacing = data['spacing']
            
            # For bilateral cases, extract features for both sides
            if patient_id in bilateral_patients:
                # Separate left and right glenoids
                unique_labels = np.unique(label)
                glenoid_labels = [l for l in unique_labels if l > 0]
                
                for glenoid_label in glenoid_labels:
                    # Create binary mask for this specific glenoid
                    binary_label = (label == glenoid_label).astype(np.uint8)
                    
                    features = self.extract_glenoid_features(image, binary_label, spacing)
                    if features:
                        features['patient_id'] = patient_id
                        features['side'] = f'side_{glenoid_label}'
                        features['case_type'] = 'bilateral'
                        features['glenoid_label'] = glenoid_label
                        all_features.append(features)
            
            # For unilateral cases
            else:
                # Assume single glenoid region
                binary_label = (label > 0).astype(np.uint8)
                
                features = self.extract_glenoid_features(image, binary_label, spacing)
                if features:
                    features['patient_id'] = patient_id
                    features['side'] = 'single'
                    features['case_type'] = 'unilateral'
                    features['glenoid_label'] = 1
                    all_features.append(features)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(all_features)
        self.feature_names = [col for col in self.features_df.columns 
                            if col not in ['patient_id', 'side', 'case_type', 'glenoid_label']]
        
        print(f"Extracted {len(self.feature_names)} features from {len(all_features)} glenoid regions")
        print("Features:", self.feature_names)
        
        return self.features_df
    
    def create_training_data(self):
        """Create training data for bone loss prediction"""
        if self.features_df is None:
            print("Please extract features first!")
            return None
            
        # For bilateral cases, calculate bone loss as difference between sides
        bilateral_data = []
        bilateral_patients = self.features_df[self.features_df['case_type'] == 'bilateral']['patient_id'].unique()
        
        for patient_id in bilateral_patients:
            patient_features = self.features_df[self.features_df['patient_id'] == patient_id]
            
            if len(patient_features) == 2:  # Exactly two sides
                # Sort by glenoid label to ensure consistent ordering
                patient_features = patient_features.sort_values('glenoid_label')
                
                side1 = patient_features.iloc[0]
                side2 = patient_features.iloc[1]
                
                # Calculate volume difference (bone loss)
                volume_diff = abs(side1['volume_mm3'] - side2['volume_mm3'])
                volume_loss_pct = volume_diff / max(side1['volume_mm3'], side2['volume_mm3']) * 100
                
                # Create feature vector (difference between sides)
                feature_diff = {}
                for feature in self.feature_names:
                    if feature in side1 and feature in side2:
                        feature_diff[f'{feature}_diff'] = abs(side1[feature] - side2[feature])
                        feature_diff[f'{feature}_ratio'] = (side1[feature] / side2[feature]) if side2[feature] != 0 else 0
                
                feature_diff['patient_id'] = patient_id
                feature_diff['bone_loss_volume'] = volume_diff
                feature_diff['bone_loss_percentage'] = volume_loss_pct
                feature_diff['case_type'] = 'bilateral'
                
                bilateral_data.append(feature_diff)
        
        # Create training dataset
        if bilateral_data:
            training_df = pd.DataFrame(bilateral_data)
            feature_cols = [col for col in training_df.columns 
                          if col not in ['patient_id', 'bone_loss_volume', 'bone_loss_percentage', 'case_type']]
            
            print(f"Created training data with {len(training_df)} samples and {len(feature_cols)} features")
            return training_df, feature_cols
        else:
            print("No bilateral training data available")
            return None, None
    
    def train_model(self):
        """Train the AI model for bone loss prediction"""
        training_data, feature_cols = self.create_training_data()
        
        if training_data is None:
            print("No training data available")
            return None
        
        # Prepare training data
        X = training_data[feature_cols].fillna(0)  # Handle any NaN values
        y = training_data['bone_loss_percentage']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  MAE: {mae:.2f}%")
        print(f"  RMSE: {rmse:.2f}%")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"  Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def predict_bone_loss(self, patient_id):
        """Predict bone loss for a specific patient"""
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        patient_features = self.features_df[self.features_df['patient_id'] == patient_id]
        
        if len(patient_features) == 0:
            print(f"Patient {patient_id} not found")
            return None
            
        if len(patient_features) == 2:  # Bilateral case
            # Calculate feature differences
            patient_features = patient_features.sort_values('glenoid_label')
            side1 = patient_features.iloc[0]
            side2 = patient_features.iloc[1]
            
            feature_diff = {}
            for feature in self.feature_names:
                if feature in side1 and feature in side2:
                    feature_diff[f'{feature}_diff'] = abs(side1[feature] - side2[feature])
                    feature_diff[f'{feature}_ratio'] = (side1[feature] / side2[feature]) if side2[feature] != 0 else 0
            
            # Prepare for prediction
            training_data, feature_cols = self.create_training_data()
            if training_data is not None:
                X_pred = pd.DataFrame([feature_diff])[feature_cols].fillna(0)
                X_pred_scaled = self.scaler.transform(X_pred)
                
                predicted_loss = self.model.predict(X_pred_scaled)[0]
                
                actual_volume_diff = abs(side1['volume_mm3'] - side2['volume_mm3'])
                actual_loss_pct = actual_volume_diff / max(side1['volume_mm3'], side2['volume_mm3']) * 100
                
                result = {
                    'patient_id': patient_id,
                    'predicted_bone_loss_pct': predicted_loss,
                    'actual_bone_loss_pct': actual_loss_pct,
                    'volume_side1': side1['volume_mm3'],
                    'volume_side2': side2['volume_mm3'],
                    'volume_difference': actual_volume_diff,
                    'case_type': 'bilateral'
                }
                
                return result
        
        else:  # Unilateral case - need statistical normal comparison
            print(f"Unilateral case prediction not implemented yet for {patient_id}")
            return None
    
    def save_model(self, filepath="glenoid_model.pkl"):
        """Save the trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'features_df': self.features_df
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath="glenoid_model.pkl"):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.features_df = model_data['features_df']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the AI system
    ai_system = GlenoidBoneLossAI(dataset_path="./glenoid/dataset/")
    
    # Step 1: Load all patient data
    patient_data = ai_system.load_patient_data()
    
    # Step 2: Extract features
    features_df = ai_system.extract_all_features()
    print("\nFeatures DataFrame shape:", features_df.shape)
    print("\nFirst few rows:")
    print(features_df.head())
    
    # Step 3: Train the model
    model = ai_system.train_model()
    
    # Step 4: Test predictions on individual patients
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    for patient_id in list(patient_data.keys())[:5]:  # Test first 5 patients
        result = ai_system.predict_bone_loss(patient_id)
        if result:
            print(f"\nPatient: {result['patient_id']}")
            print(f"Predicted bone loss: {result['predicted_bone_loss_pct']:.2f}%")
            print(f"Actual bone loss: {result['actual_bone_loss_pct']:.2f}%")
            print(f"Volume difference: {result['volume_difference']:.2f} mm³")
    
    # Step 5: Save the model
    ai_system.save_model("glenoid_ai_model.pkl")
