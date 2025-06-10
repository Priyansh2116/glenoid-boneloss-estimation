import os
import json
import numpy as np
import SimpleITK as sitk
import pandas as pd
from collections import defaultdict

class DatasetValidator:
    def __init__(self, dataset_path="./glenoid/dataset/"):
        self.dataset_path = dataset_path
        self.validation_results = {}
        
    def validate_dataset(self):
        """Comprehensive dataset validation"""
        print("="*60)
        print("GLENOID DATASET VALIDATION")
        print("="*60)
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return False
            
        # Get all patient folders
        patient_folders = [f for f in os.listdir(self.dataset_path) 
                          if f.startswith('patient') and os.path.isdir(os.path.join(self.dataset_path, f))]
        
        if not patient_folders:
            print(f"❌ No patient folders found in {self.dataset_path}")
            return False
            
        print(f"Found {len(patient_folders)} patient folders")
        
        # Validate each patient
        valid_patients = []
        invalid_patients = []
        
        for patient_folder in sorted(patient_folders):
            print(f"\n📁 Validating {patient_folder}...")
            
            is_valid = self.validate_patient(patient_folder)
            if is_valid:
                valid_patients.append(patient_folder)
                print(f"  ✅ {patient_folder} - VALID")
            else:
                invalid_patients.append(patient_folder)
                print(f"  ❌ {patient_folder} - INVALID")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"VALIDATION SUMMARY")
        print(f"="*60)
        print(f"✅ Valid patients: {len(valid_patients)}")
        print(f"❌ Invalid patients: {len(invalid_patients)}")
        
        if invalid_patients:
            print(f"\nInvalid patients: {', '.join(invalid_patients)}")
            
        # Generate detailed report
        self.generate_dataset_report(valid_patients)
        
        return len(invalid_patients) == 0
    
    def validate_patient(self, patient_folder):
        """Validate individual patient data"""
        patient_path = os.path.join(self.dataset_path, patient_folder)
        
        # Check required files
        image_path = os.path.join(patient_path, 'image.nii.gz')
        label_path = os.path.join(patient_path, 'label.nii.gz')
        
        if not os.path.exists(image_path):
            print(f"    ❌ Missing image.nii.gz")
            return False
            
        if not os.path.exists(label_path):
            print(f"    ❌ Missing label.nii.gz")
            return False
        
        try:
            # Load images
            img = sitk.ReadImage(image_path)
            lbl = sitk.ReadImage(label_path)
            
            img_arr = sitk.GetArrayFromImage(img)
            lbl_arr = sitk.GetArrayFromImage(lbl)
            
            # Get metadata
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
            
            # Validate shapes
            if img_arr.shape != lbl_arr.shape:
                print(f"    ❌ Shape mismatch: image{img_arr.shape} vs label{lbl_arr.shape}")
                return False
            
            # Validate label values
            unique_labels = np.unique(lbl_arr)
            valid_labels = [0, 1, 2]  # Background, left, right
            
            invalid_labels = [l for l in unique_labels if l not in valid_labels]
            if invalid_labels:
                print(f"    ⚠️  Unexpected label values: {invalid_labels}")
            
            # Count glenoid regions
            glenoid_labels = [l for l in unique_labels if l > 0]
            case_type = "bilateral" if len(glenoid_labels) >= 2 else "unilateral"
            
            # Check for empty labels
            if len(glenoid_labels) == 0:
                print(f"    ❌ No glenoid regions found in label")
                return False
            
            # Store validation results
            self.validation_results[patient_folder] = {
                'image_shape': img_arr.shape,
                'label_shape': lbl_arr.shape,
                'spacing': spacing,
                'origin': origin,
                'unique_labels': list(unique_labels),
                'glenoid_labels': glenoid_labels,
                'case_type': case_type,
                'voxel_volume': np.prod(spacing),
                'image_size_mb': os.path.getsize(image_path) / (1024*1024),
                'label_size_mb': os.path.getsize(label_path) / (1024*1024)
            }
            
            # Basic info
            print(f"    📊 Shape: {img_arr.shape}")
            print(f"    📏 Spacing: {spacing}")
            print(f"    🏷️  Labels: {unique_labels}")
            print(f"    📋 Type: {case_type}")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error loading files: {str(e)}")
            return False
    
    def generate_dataset_report(self, valid_patients):
        """Generate comprehensive dataset report"""
        if not self.validation_results:
            return
            
        print(f"\n" + "="*60)
        print(f"DETAILED DATASET REPORT")
        print(f"="*60)
        
        # Convert to DataFrame for analysis
        df_data = []
        for patient, data in self.validation_results.items():
            if patient in valid_patients:
                row = {'patient': patient}
                row.update(data)
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Case type distribution
        case_counts = df['case_type'].value_counts()
        print(f"\n📊 Case Type Distribution:")
        for case_type, count in case_counts.items():
            print(f"  {case_type}: {count} patients")
        
        # Image dimensions
        print(f"\n📏 Image Dimensions:")
        shapes = df['image_shape'].apply(str).value_counts()
        for shape, count in shapes.items():
            print(f"  {shape}: {count} patients")
        
        # Voxel spacing
        print(f"\n🔍 Voxel Spacing:")
        spacings = df['spacing'].apply(str).value_counts()
        for spacing, count in spacings.items():
            print(f"  {spacing}: {count} patients")
        
        # File sizes
        print(f"\n💾 File Sizes:")
        print(f"  Image files: {df['image_size_mb'].mean():.1f} ± {df['image_size_mb'].std():.1f} MB")
        print(f"  Label files: {df['label_size_mb'].mean():.1f} ± {df['label_size_mb'].std():.1f} MB")
        
        # Label analysis
        print(f"\n🏷️  Label Analysis:")
        all_labels = []
        for labels in df['unique_labels']:
            all_labels.extend(labels)
        unique_all = sorted(list(set(all_labels)))
        print(f"  All unique labels found: {unique_all}")
        
        # Bilateral vs Unilateral details
        print(f"\n🔍 Detailed Case Analysis:")
        bilateral_patients = df[df['case_type'] == 'bilateral']
        unilateral_patients = df[df['case_type'] == 'unilateral']
        
        print(f"  Bilateral patients ({len(bilateral_patients)}):")
        for _, row in bilateral_patients.iterrows():
            print(f"    {row['patient']}: labels {row['glenoid_labels']}")
        
        print(f"  Unilateral patients ({len(unilateral_patients)}):")
        for _, row in unilateral_patients.iterrows():
            print(f"    {row['patient']}: labels {row['glenoid_labels']}")
    
    def check_data_consistency(self):
        """Check for data consistency issues"""
        print(f"\n" + "="*60)
        print(f"DATA CONSISTENCY CHECK")
        print(f"="*60)
        
        if not self.validation_results:
            print("No validation data available")
            return
        
        # Check spacing consistency
        spacings = [data['spacing'] for data in self.validation_results.values()]
        unique_spacings = list(set([str(s) for s in spacings]))
        
        if len(unique_spacings) > 1:
            print(f"⚠️  Warning: Multiple voxel spacings detected:")
            for spacing in unique_spacings:
                count = sum(1 for s in spacings if str(s) == spacing)
                print(f"    {spacing}: {count} patients")
            print(f"    Recommendation: Consider resampling to consistent spacing")
        else:
            print(f"✅ All images have consistent voxel spacing")
        
        # Check shape consistency
        shapes = [data['image_shape'] for data in self.validation_results.values()]
        unique_shapes = list(set([str(s) for s in shapes]))
        
        if len(unique_shapes) > 3:  # Allow some variation
            print(f"⚠️  Warning: High variation in image dimensions:")
            for shape in unique_shapes:
                count = sum(1 for s in shapes if str(s) == shape)
                print(f"    {shape}: {count} patients")
        else:
            print(f"✅ Image dimensions are reasonably consistent")
    
    def generate_metadata_template(self):
        """Generate metadata template files"""
        print(f"\n" + "="*60)
        print(f"GENERATING METADATA TEMPLATES")
        print(f"="*60)
        
        for patient_folder in self.validation_results.keys():
            patient_path = os.path.join(self.dataset_path, patient_folder)
            metadata_path = os.path.join(patient_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                data = self.validation_results[patient_folder]
                
                metadata = {
                    "patient_id": patient_folder,
                    "case_type": data['case_type'],
                    "glenoid_labels": data['glenoid_labels'],
                    "image_shape": data['image_shape'],
                    "spacing": data['spacing'],
                    "manual_measurements": {
                        "notes": "Add manual volume measurements here",
                        "healthy_volume_mm3": None,
                        "affected_volume_mm3": None,
                        "bone_loss_percentage": None
                    },
                    "clinical_info": {
                        "age": None,
                        "gender": None,
                        "affected_side": None,
                        "pathology": None,
                        "severity": None
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                print(f"  📝 Created metadata template: {patient_folder}/metadata.json")

# Usage example
if __name__ == "__main__":
    # Initialize validator
    validator = DatasetValidator(dataset_path="./glenoid/dataset/")
    
    # Run validation
    is_valid = validator.validate_dataset()
    
    # Check consistency
    validator.check_data_consistency()
    
    # Generate metadata templates
    validator.generate_metadata_template()
    
    if is_valid:
        print(f"\n🎉 Dataset is ready for AI training!")
    else:
        print(f"\n⚠️  Please fix the issues before proceeding with AI training.")
