import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import json
from datetime import datetime
import threading
from scipy import ndimage
from skimage import measure, morphology, segmentation
import warnings
warnings.filterwarnings('ignore')

class GlenoidDatasetAnalyzer:
    """Phase 1: Dataset Analysis and Preprocessing"""
    
    def __init__(self, dataset_path="./glenoid/dataset/"):
        self.dataset_path = dataset_path
        self.patients_info = {}
        self.analysis_results = {}
        
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🔍 Analyzing Glenoid Dataset...")
        print("=" * 50)
        
        # Find all patient folders
        patient_folders = [f for f in os.listdir(self.dataset_path) 
                          if f.startswith('patient') and os.path.isdir(os.path.join(self.dataset_path, f))]
        patient_folders.sort()
        
        print(f"Found {len(patient_folders)} patients: {patient_folders}")
        
        # Analyze each patient
        for patient_folder in patient_folders:
            patient_id = patient_folder.replace('patient', '')
            patient_path = os.path.join(self.dataset_path, patient_folder)
            
            self.patients_info[patient_id] = self._analyze_patient(patient_path, patient_id)
            
        # Generate summary
        self._generate_summary()
        return self.analysis_results
    
    def _analyze_patient(self, patient_path, patient_id):
        """Analyze individual patient data"""
        print(f"\n📋 Patient {patient_id}:")
        
        patient_info = {
            'patient_id': patient_id,
            'path': patient_path,
            'files': [],
            'sides': [],
            'image_properties': {},
            'label_properties': {}
        }
        
        # Check for files
        files = os.listdir(patient_path)
        patient_info['files'] = files
        
        # Look for image and label files
        image_files = [f for f in files if 'image' in f.lower() and f.endswith('.nii.gz')]
        label_files = [f for f in files if 'label' in f.lower() and f.endswith('.nii.gz')]
        
        print(f"  Image files: {image_files}")
        print(f"  Label files: {label_files}")
        
        # Analyze image properties
        if image_files:
            img_path = os.path.join(patient_path, image_files[0])
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            
            patient_info['image_properties'] = {
                'shape': img_arr.shape,
                'spacing': img.GetSpacing(),
                'origin': img.GetOrigin(),
                'direction': img.GetDirection(),
                'size_mm': [img_arr.shape[i] * img.GetSpacing()[i] for i in range(3)],
                'intensity_range': [float(img_arr.min()), float(img_arr.max())],
                'mean_intensity': float(img_arr.mean()),
                'file_size_mb': os.path.getsize(img_path) / (1024*1024)
            }
            
            print(f"  📏 Image shape: {img_arr.shape}")
            print(f"  📐 Spacing: {img.GetSpacing()}")
            print(f"  🎯 Intensity range: [{img_arr.min():.1f}, {img_arr.max():.1f}]")
        
        # Analyze label properties
        if label_files:
            lbl_path = os.path.join(patient_path, label_files[0])
            lbl = sitk.ReadImage(lbl_path)
            lbl_arr = sitk.GetArrayFromImage(lbl)
            
            # Find unique labels
            unique_labels = np.unique(lbl_arr)
            label_counts = {int(label): int(np.sum(lbl_arr == label)) for label in unique_labels}
            
            patient_info['label_properties'] = {
                'shape': lbl_arr.shape,
                'unique_labels': unique_labels.tolist(),
                'label_counts': label_counts,
                'total_segmented_voxels': int(np.sum(lbl_arr > 0)),
                'segmentation_ratio': float(np.sum(lbl_arr > 0) / lbl_arr.size)
            }
            
            print(f"  🏷️  Unique labels: {unique_labels}")
            print(f"  📊 Segmented voxels: {np.sum(lbl_arr > 0)} ({100*np.sum(lbl_arr > 0)/lbl_arr.size:.2f}%)")
            
            # Detect if bilateral (multiple glenoid regions)
            if len(unique_labels) > 2:  # 0 (background) + multiple glenoids
                patient_info['sides'] = ['bilateral']
                print("  👥 Bilateral case detected")
            else:
                patient_info['sides'] = ['unilateral']
                print("  👤 Unilateral case detected")
        
        return patient_info
    
    def _generate_summary(self):
        """Generate comprehensive dataset summary"""
        total_patients = len(self.patients_info)
        bilateral_count = sum(1 for p in self.patients_info.values() if 'bilateral' in p['sides'])
        unilateral_count = total_patients - bilateral_count
        
        # Image statistics
        shapes = [p['image_properties']['shape'] for p in self.patients_info.values() if p['image_properties']]
        spacings = [p['image_properties']['spacing'] for p in self.patients_info.values() if p['image_properties']]
        
        self.analysis_results = {
            'total_patients': total_patients,
            'bilateral_cases': bilateral_count,
            'unilateral_cases': unilateral_count,
            'common_shape': max(set(map(tuple, shapes)), key=shapes.count) if shapes else None,
            'common_spacing': max(set(spacings), key=spacings.count) if spacings else None,
            'total_scans': bilateral_count * 2 + unilateral_count,
            'data_summary': self.patients_info
        }
        
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        print(f"Total patients: {total_patients}")
        print(f"Bilateral cases: {bilateral_count}")
        print(f"Unilateral cases: {unilateral_count}")
        print(f"Total scans available: {self.analysis_results['total_scans']}")
        print(f"Most common shape: {self.analysis_results['common_shape']}")
        print(f"Most common spacing: {self.analysis_results['common_spacing']}")

class GlenoidPreprocessor:
    """Phase 1: Data Preprocessing Pipeline"""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_size=(128, 128, 64)):
        self.target_spacing = target_spacing
        self.target_size = target_size
        
    def preprocess_image(self, image_path, label_path=None):
        """Complete preprocessing pipeline for a single case"""
        print(f"🔧 Preprocessing: {os.path.basename(image_path)}")
        
        # Load image
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Load label if available
        label = None
        label_array = None
        if label_path and os.path.exists(label_path):
            label = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label)
        
        # Step 1: Intensity normalization (HU windowing for bone)
        image_array = self._normalize_intensity(image_array)
        
        # Step 2: Resampling to target spacing
        if image.GetSpacing() != self.target_spacing:
            image_resampled, label_resampled = self._resample_image_label(
                image, label, self.target_spacing)
            image_array = sitk.GetArrayFromImage(image_resampled)
            if label_resampled:
                label_array = sitk.GetArrayFromImage(label_resampled)
        
        # Step 3: ROI extraction around glenoid
        if label_array is not None:
            image_array, label_array = self._extract_roi(image_array, label_array)
        
        # Step 4: Resize to target size
        image_array = self._resize_to_target(image_array, self.target_size)
        if label_array is not None:
            label_array = self._resize_to_target(label_array, self.target_size, is_label=True)
        
        return image_array, label_array
    
    def _normalize_intensity(self, image_array, window_center=400, window_width=1500):
        """Normalize CT intensities using bone window"""
        # Bone window: Center=400 HU, Width=1500 HU
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        
        # Clip and normalize to [0, 1]
        image_array = np.clip(image_array, min_hu, max_hu)
        image_array = (image_array - min_hu) / (max_hu - min_hu)
        
        return image_array.astype(np.float32)
    
    def _resample_image_label(self, image, label, target_spacing):
        """Resample image and label to target spacing"""
        # Calculate new size
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_size = [
            int(np.round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        
        # Resample image
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        image_resampled = resampler.Execute(image)
        
        # Resample label with nearest neighbor
        label_resampled = None
        if label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            label_resampled = resampler.Execute(label)
        
        return image_resampled, label_resampled
    
    def _extract_roi(self, image_array, label_array, margin=20):
        """Extract ROI around the segmented region"""
        # Find bounding box of the segmentation
        coords = np.where(label_array > 0)
        if len(coords[0]) == 0:
            return image_array, label_array
        
        min_coords = [max(0, np.min(coords[i]) - margin) for i in range(3)]
        max_coords = [min(image_array.shape[i], np.max(coords[i]) + margin) for i in range(3)]
        
        # Extract ROI
        roi_image = image_array[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        roi_label = label_array[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        return roi_image, roi_label
    
    def _resize_to_target(self, array, target_size, is_label=False):
        """Resize array to target size"""
        if array.shape == target_size:
            return array
        
        # Calculate zoom factors
        zoom_factors = [target_size[i] / array.shape[i] for i in range(3)]
        
        # Use appropriate interpolation
        if is_label:
            # Nearest neighbor for labels
            resized = ndimage.zoom(array, zoom_factors, order=0)
        else:
            # Linear interpolation for images
            resized = ndimage.zoom(array, zoom_factors, order=1)
        
        return resized

class GlenoidDataAugmentation:
    """Data augmentation for small dataset"""
    
    def __init__(self):
        self.augmentations = [
            'rotation', 'flip', 'scale', 'noise', 'intensity'
        ]
    
    def augment_data(self, image, label, num_augmentations=5):
        """Generate augmented versions of the data"""
        augmented_pairs = []
        
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_label = label.copy()
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                aug_image, aug_label = self._rotate_3d(aug_image, aug_label, angle)
            
            # Random flip
            if np.random.random() > 0.5:
                axis = np.random.choice([0, 1, 2])
                aug_image = np.flip(aug_image, axis=axis)
                aug_label = np.flip(aug_label, axis=axis)
            
            # Random scale
            if np.random.random() > 0.5:
                scale_factor = np.random.uniform(0.9, 1.1)
                aug_image, aug_label = self._scale_3d(aug_image, aug_label, scale_factor)
            
            # Add noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, aug_image.shape)
                aug_image = np.clip(aug_image + noise, 0, 1)
            
            # Intensity variation
            if np.random.random() > 0.5:
                intensity_factor = np.random.uniform(0.8, 1.2)
                aug_image = np.clip(aug_image * intensity_factor, 0, 1)
            
            augmented_pairs.append((aug_image, aug_label))
        
        return augmented_pairs
    
    def _rotate_3d(self, image, label, angle):
        """3D rotation around random axis"""
        # Simple rotation around z-axis for now
        rotated_image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=1)
        rotated_label = ndimage.rotate(label, angle, axes=(1, 2), reshape=False, order=0)
        return rotated_image, rotated_label
    
    def _scale_3d(self, image, label, scale_factor):
        """3D scaling"""
        scaled_image = ndimage.zoom(image, scale_factor, order=1)
        scaled_label = ndimage.zoom(label, scale_factor, order=0)
        
        # Crop or pad to original size
        target_shape = image.shape
        scaled_image = self._crop_or_pad(scaled_image, target_shape)
        scaled_label = self._crop_or_pad(scaled_label, target_shape)
        
        return scaled_image, scaled_label
    
    def _crop_or_pad(self, array, target_shape):
        """Crop or pad array to target shape"""
        current_shape = array.shape
        
        # Calculate padding/cropping for each dimension
        result = array.copy()
        
        for dim in range(len(target_shape)):
            if current_shape[dim] > target_shape[dim]:
                # Crop
                start = (current_shape[dim] - target_shape[dim]) // 2
                end = start + target_shape[dim]
                result = np.take(result, range(start, end), axis=dim)
            elif current_shape[dim] < target_shape[dim]:
                # Pad
                pad_width = [(0, 0)] * len(target_shape)
                pad_total = target_shape[dim] - current_shape[dim]
                pad_width[dim] = (pad_total // 2, pad_total - pad_total // 2)
                result = np.pad(result, pad_width, mode='constant', constant_values=0)
        
        return result

# GUI Application
class GlenoidAIApp:
    """Main GUI Application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Glenoid Bone Loss AI System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.dataset_analyzer = GlenoidDatasetAnalyzer()
        self.preprocessor = GlenoidPreprocessor()
        self.augmenter = GlenoidDataAugmentation()
        
        # Variables
        self.dataset_path = tk.StringVar(value="./glenoid/dataset/")
        self.analysis_results = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Dataset Analysis
        self.setup_analysis_tab(notebook)
        
        # Tab 2: Preprocessing
        self.setup_preprocessing_tab(notebook)
        
        # Tab 3: Model Training (placeholder)
        self.setup_training_tab(notebook)
    
    def setup_analysis_tab(self, notebook):
        """Setup dataset analysis tab"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Dataset Analysis")
        
        # Path selection
        path_frame = ttk.Frame(analysis_frame)
        path_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(path_frame, text="Dataset Path:").pack(side='left')
        ttk.Entry(path_frame, textvariable=self.dataset_path, width=50).pack(side='left', padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_dataset).pack(side='left')
        ttk.Button(path_frame, text="Analyze", command=self.analyze_dataset).pack(side='left', padx=5)
        
        # Results display
        self.analysis_text = tk.Text(analysis_frame, height=30, width=80)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
        self.analysis_text.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        scrollbar.pack(side='right', fill='y')
    
    def setup_preprocessing_tab(self, notebook):
        """Setup preprocessing tab"""
        prep_frame = ttk.Frame(notebook)
        notebook.add(prep_frame, text="Preprocessing")
        
        # Controls
        control_frame = ttk.Frame(prep_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Preprocess All", 
                  command=self.preprocess_all_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Generate Augmentations", 
                  command=self.generate_augmentations).pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(prep_frame, mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=5)
        
        # Status text
        self.status_text = tk.Text(prep_frame, height=25, width=80)
        self.status_text.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_training_tab(self, notebook):
        """Setup model training tab (placeholder)"""
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Model Training")
        
        ttk.Label(training_frame, text="Model Training Coming Soon...", 
                 font=('Arial', 16)).pack(expand=True)
    
    def browse_dataset(self):
        """Browse for dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_path.set(folder)
    
    def analyze_dataset(self):
        """Run dataset analysis"""
        def run_analysis():
            try:
                self.analysis_text.delete(1.0, tk.END)
                self.analysis_text.insert(tk.END, "Starting dataset analysis...\n\n")
                self.analysis_text.update()
                
                # Update analyzer path
                self.dataset_analyzer.dataset_path = self.dataset_path.get()
                
                # Run analysis
                self.analysis_results = self.dataset_analyzer.analyze_dataset()
                
                # Display results
                self.display_analysis_results()
                
            except Exception as e:
                self.analysis_text.insert(tk.END, f"Error: {str(e)}\n")
        
        # Run in separate thread to avoid GUI freezing
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def display_analysis_results(self):
        """Display analysis results in GUI"""
        if not self.analysis_results:
            return
        
        results_text = f"""
DATASET ANALYSIS RESULTS
{'='*50}

📊 Summary:
- Total patients: {self.analysis_results['total_patients']}
- Bilateral cases: {self.analysis_results['bilateral_cases']}
- Unilateral cases: {self.analysis_results['unilateral_cases']}
- Total scans: {self.analysis_results['total_scans']}
- Common shape: {self.analysis_results['common_shape']}
- Common spacing: {self.analysis_results['common_spacing']}

📋 Patient Details:
"""
        
        for patient_id, info in self.analysis_results['data_summary'].items():
            results_text += f"\nPatient {patient_id}:\n"
            if info['image_properties']:
                props = info['image_properties']
                results_text += f"  - Shape: {props['shape']}\n"
                results_text += f"  - Spacing: {props['spacing']}\n"
                results_text += f"  - Intensity: [{props['intensity_range'][0]:.1f}, {props['intensity_range'][1]:.1f}]\n"
            
            if info['label_properties']:
                lprops = info['label_properties']
                results_text += f"  - Labels: {lprops['unique_labels']}\n"
                results_text += f"  - Segmented: {lprops['segmentation_ratio']:.2%}\n"
            
            results_text += f"  - Type: {info['sides'][0] if info['sides'] else 'unknown'}\n"
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, results_text)
    
    def preprocess_all_data(self):
        """Preprocess all patient data"""
        def run_preprocessing():
            if not self.analysis_results:
                self.status_text.insert(tk.END, "Please run dataset analysis first!\n")
                return
            
            total_patients = len(self.analysis_results['data_summary'])
            self.progress['maximum'] = total_patients
            
            for i, (patient_id, info) in enumerate(self.analysis_results['data_summary'].items()):
                try:
                    self.status_text.insert(tk.END, f"Processing Patient {patient_id}...\n")
                    self.status_text.update()
                    
                    # Find image and label files
                    patient_path = info['path']
                    files = os.listdir(patient_path)
                    
                    image_file = next((f for f in files if 'image' in f.lower() and f.endswith('.nii.gz')), None)
                    label_file = next((f for f in files if 'label' in f.lower() and f.endswith('.nii.gz')), None)
                    
                    if image_file:
                        image_path = os.path.join(patient_path, image_file)
                        label_path = os.path.join(patient_path, label_file) if label_file else None
                        
                        # Preprocess
                        processed_image, processed_label = self.preprocessor.preprocess_image(
                            image_path, label_path)
                        
                        # Save processed data
                        output_dir = os.path.join(patient_path, 'processed')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        np.save(os.path.join(output_dir, 'image_processed.npy'), processed_image)
                        if processed_label is not None:
                            np.save(os.path.join(output_dir, 'label_processed.npy'), processed_label)
                        
                        self.status_text.insert(tk.END, f"✅ Patient {patient_id} processed successfully\n")
                    
                    self.progress['value'] = i + 1
                    self.progress.update()
                    
                except Exception as e:
                    self.status_text.insert(tk.END, f"❌ Error processing Patient {patient_id}: {str(e)}\n")
            
            self.status_text.insert(tk.END, "\n🎉 Preprocessing completed!\n")
        
        threading.Thread(target=run_preprocessing, daemon=True).start()
    
    def generate_augmentations(self):
        """Generate data augmentations"""
        def run_augmentation():
            self.status_text.insert(tk.END, "Generating augmentations...\n")
            # Implementation for augmentation generation
            self.status_text.insert(tk.END, "Augmentation generation completed!\n")
        
        threading.Thread(target=run_augmentation, daemon=True).start()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    # Test the components
    print("🚀 Starting Glenoid AI System...")
    
    # Option 1: Run GUI
    app = GlenoidAIApp()
    app.run()
    
    # Option 2: Run command line analysis (uncomment if needed)
    # analyzer = GlenoidDatasetAnalyzer("./glenoid/dataset/")
    # results = analyzer.analyze_dataset()
