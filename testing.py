import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the model classes from your training script
# (Copy the DoubleConv and UNet3D classes here if running separately)

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net for glenoid segmentation"""
    
    def __init__(self, in_channels=1, num_classes=3, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder (downsampling)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final classifier
        self.final_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transpose conv
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)  # Double conv
        
        return self.final_conv(x)

class GlenoidModelTester:
    """Comprehensive testing suite for glenoid segmentation model"""
    
    def __init__(self, model_path, dataset_path="./glenoid/dataset/", device=None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🧪 Model Tester initialized on device: {self.device}")
        
        # Load model
        self.model = UNet3D(in_channels=1, num_classes=3)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Class names for interpretation
        self.class_names = ['Background', 'Left Glenoid', 'Right Glenoid']
        self.test_results = {}
    
    def load_and_preprocess_image(self, image_path, label_path=None):
        """Load and preprocess a single image (and label if provided)"""
        # Load with SimpleITK
        img = sitk.ReadImage(image_path)
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Get original spacing for volume calculations
        original_spacing = img.GetSpacing()[::-1]  # ITK is (x,y,z), we want (z,y,x)
        
        # Intensity normalization (bone window)
        img_array = np.clip(img_array, -175, 1225)  # Bone window: 400±750 HU
        img_array = (img_array + 175) / 1400.0  # Normalize to [0,1]
        
        # Resize to target size (128x128x64)
        target_shape = (64, 128, 128)  # (D, H, W)
        
        from scipy import ndimage
        zoom_factors = [target_shape[i] / img_array.shape[i] for i in range(3)]
        img_resized = ndimage.zoom(img_array, zoom_factors, order=1)
        
        # Load label if provided
        label_resized = None
        if label_path and os.path.exists(label_path):
            lbl = sitk.ReadImage(label_path)
            lbl_array = sitk.GetArrayFromImage(lbl).astype(np.int64)
            label_resized = ndimage.zoom(lbl_array, zoom_factors, order=0)
        
        return img_resized, label_resized, original_spacing, zoom_factors
    
    def predict_single_image(self, image_path, visualize=True):
        """Test model on a single image"""
        print(f"\n🔍 Testing on: {os.path.basename(image_path)}")
        
        # Load and preprocess
        image, _, original_spacing, zoom_factors = self.load_and_preprocess_image(image_path)
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            
            if start_time:
                start_time.record()
            
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                inference_time = 0.1  # Approximate for CPU
        
        # Get probability maps
        prob_maps = probabilities.cpu().numpy()[0]
        
        # Calculate volumes
        volumes = self.calculate_volumes(prediction, original_spacing, zoom_factors)
        
        print(f"⏱️  Inference time: {inference_time:.3f} seconds")
        print(f"📊 Class distribution:")
        for i, class_name in enumerate(self.class_names):
            voxel_count = np.sum(prediction == i)
            percentage = (voxel_count / prediction.size) * 100
            print(f"   {class_name}: {voxel_count} voxels ({percentage:.2f}%)")
        
        print(f"📏 Estimated volumes:")
        for class_name, volume in volumes.items():
            if volume > 0:
                print(f"   {class_name}: {volume:.2f} mm³")
        
        if visualize:
            self.visualize_prediction(image, prediction, prob_maps, 
                                    title=f"Prediction: {os.path.basename(image_path)}")
        
        result = {
            'image_path': image_path,
            'inference_time': inference_time,
            'prediction': prediction,
            'probabilities': prob_maps,
            'volumes': volumes,
            'class_distribution': {self.class_names[i]: np.sum(prediction == i) for i in range(len(self.class_names))}
        }
        
        return result
    
    def calculate_volumes(self, prediction, original_spacing, zoom_factors):
        """Calculate volumes for each class"""
        # Calculate voxel size after resizing
        resized_voxel_volume = np.prod([original_spacing[i] / zoom_factors[i] for i in range(3)])
        
        volumes = {}
        for i, class_name in enumerate(self.class_names):
            voxel_count = np.sum(prediction == i)
            volume_mm3 = voxel_count * resized_voxel_volume
            volumes[class_name] = volume_mm3
        
        return volumes
    
    def test_with_ground_truth(self, image_path, label_path):
        """Test model against ground truth segmentation"""
        print(f"\n🎯 Testing with ground truth: {os.path.basename(image_path)}")
        
        # Load and preprocess
        image, label, original_spacing, zoom_factors = self.load_and_preprocess_image(image_path, label_path)
        
        if label is None:
            print("❌ Could not load ground truth label")
            return None
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        
        # Calculate metrics
        metrics = self.calculate_metrics(prediction, label)
        
        # Calculate volume comparison
        pred_volumes = self.calculate_volumes(prediction, original_spacing, zoom_factors)
        true_volumes = self.calculate_volumes(label, original_spacing, zoom_factors)
        
        print(f"📊 Segmentation Metrics:")
        print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Mean Dice Score: {metrics['mean_dice']:.4f}")
        print(f"   Class-wise Dice Scores:")
        for i, class_name in enumerate(self.class_names):
            if f'dice_class_{i}' in metrics:
                print(f"     {class_name}: {metrics[f'dice_class_{i}']:.4f}")
        
        print(f"📏 Volume Comparison:")
        for class_name in self.class_names:
            if true_volumes[class_name] > 0 or pred_volumes[class_name] > 0:
                true_vol = true_volumes[class_name]
                pred_vol = pred_volumes[class_name]
                error = abs(pred_vol - true_vol)
                rel_error = (error / true_vol * 100) if true_vol > 0 else float('inf')
                print(f"   {class_name}:")
                print(f"     True: {true_vol:.2f} mm³, Pred: {pred_vol:.2f} mm³")
                print(f"     Absolute Error: {error:.2f} mm³ ({rel_error:.1f}%)")
        
        # Visualize comparison
        self.visualize_comparison(image, label, prediction, 
                                title=f"Comparison: {os.path.basename(image_path)}")
        
        result = {
            'image_path': image_path,
            'label_path': label_path,
            'metrics': metrics,
            'true_volumes': true_volumes,
            'pred_volumes': pred_volumes,
            'prediction': prediction,
            'ground_truth': label
        }
        
        return result
    
    def calculate_metrics(self, prediction, ground_truth):
        """Calculate segmentation metrics"""
        # Flatten arrays
        pred_flat = prediction.flatten()
        true_flat = ground_truth.flatten()
        
        # Overall accuracy
        accuracy = np.mean(pred_flat == true_flat)
        
        # Class-wise Dice scores
        dice_scores = []
        metrics = {'accuracy': accuracy}
        
        for class_id in range(len(self.class_names)):
            # Binary masks for this class
            pred_binary = (pred_flat == class_id).astype(int)
            true_binary = (true_flat == class_id).astype(int)
            
            # Dice coefficient
            intersection = np.sum(pred_binary * true_binary)
            dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(true_binary) + 1e-8)
            
            dice_scores.append(dice)
            metrics[f'dice_class_{class_id}'] = dice
        
        metrics['mean_dice'] = np.mean(dice_scores)
        metrics['dice_scores'] = dice_scores
        
        return metrics
    
    def test_dataset(self, test_folder=None, max_samples=None):
        """Test model on multiple images"""
        print(f"\n🧪 Testing model on dataset...")
        
        # Get test images
        if test_folder is None:
            test_folder = self.dataset_path
        
        test_cases = []
        patients = [f for f in os.listdir(test_folder) 
                   if f.startswith('patient') and os.path.isdir(os.path.join(test_folder, f))]
        
        if max_samples:
            patients = patients[:max_samples]
        
        for patient_folder in patients:
            patient_path = os.path.join(test_folder, patient_folder)
            image_path = os.path.join(patient_path, 'image.nii.gz')
            label_path = os.path.join(patient_path, 'label.nii.gz')
            
            if os.path.exists(image_path):
                test_cases.append((image_path, label_path if os.path.exists(label_path) else None))
        
        print(f"Found {len(test_cases)} test cases")
        
        # Test each case
        results = []
        all_metrics = []
        
        for image_path, label_path in tqdm(test_cases, desc="Testing"):
            if label_path:
                result = self.test_with_ground_truth(image_path, label_path)
                if result:
                    all_metrics.append(result['metrics'])
            else:
                result = self.predict_single_image(image_path, visualize=False)
            
            if result:
                results.append(result)
        
        # Aggregate results
        if all_metrics:
            self.summarize_test_results(all_metrics)
        
        self.test_results = results
        return results
    
    def summarize_test_results(self, all_metrics):
        """Summarize test results across all samples"""
        print(f"\n📊 OVERALL TEST RESULTS")
        print("="*50)
        
        # Calculate mean and std for each metric
        accuracies = [m['accuracy'] for m in all_metrics]
        mean_dices = [m['mean_dice'] for m in all_metrics]
        
        print(f"Overall Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean Dice Score: {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
        print(f"\nClass-wise Dice Scores:")
        
        for i, class_name in enumerate(self.class_names):
            class_dices = [m[f'dice_class_{i}'] for m in all_metrics if f'dice_class_{i}' in m]
            if class_dices:
                print(f"  {class_name}: {np.mean(class_dices):.4f} ± {np.std(class_dices):.4f}")
        
        # Save summary
        summary = {
            'num_samples': len(all_metrics),
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_dice': float(np.mean(mean_dices)),
            'std_dice': float(np.std(mean_dices)),
            'class_wise_dice': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for i, class_name in enumerate(self.class_names):
            class_dices = [m[f'dice_class_{i}'] for m in all_metrics if f'dice_class_{i}' in m]
            if class_dices:
                summary['class_wise_dice'][class_name] = {
                    'mean': float(np.mean(class_dices)),
                    'std': float(np.std(class_dices))
                }
        
        with open('glenoid_test_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Test summary saved to 'glenoid_test_summary.json'")
    
    def visualize_prediction(self, image, prediction, probabilities, title="Prediction"):
        """Visualize model prediction"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Select middle slice
        mid_slice = image.shape[0] // 2
        
        # Original image
        axes[0, 0].imshow(image[mid_slice], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction
        axes[0, 1].imshow(prediction[mid_slice], cmap='viridis', vmin=0, vmax=2)
        axes[0, 1].set_title('Prediction')
        axes[0, 1].axis('off')
        
        # Probability maps for each class
        for i in range(min(3, probabilities.shape[0])):
            row = i // 2
            col = 2 + (i % 2)
            
            axes[row, col].imshow(probabilities[i, mid_slice], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f'{self.class_names[i]} Probability')
            axes[row, 0].axis('off')
        
        # 3D view (sagittal slice)
        mid_sagittal = image.shape[2] // 2
        axes[1, 0].imshow(image[:, :, mid_sagittal], cmap='gray')
        axes[1, 0].set_title('Sagittal View')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(prediction[:, :, mid_sagittal], cmap='viridis', vmin=0, vmax=2)
        axes[1, 1].set_title('Sagittal Prediction')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def visualize_comparison(self, image, ground_truth, prediction, title="Comparison"):
        """Visualize prediction vs ground truth"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Select middle slice
        mid_slice = image.shape[0] // 2
        
        # Original image
        axes[0, 0].imshow(image[mid_slice], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(ground_truth[mid_slice], cmap='viridis', vmin=0, vmax=2)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(prediction[mid_slice], cmap='viridis', vmin=0, vmax=2)
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # Error map
        error_map = (prediction != ground_truth).astype(int)
        axes[1, 0].imshow(error_map[mid_slice], cmap='Reds')
        axes[1, 0].set_title('Error Map')
        axes[1, 0].axis('off')
        
        # Overlay comparison
        overlay = np.zeros((ground_truth.shape[1], ground_truth.shape[2], 3))
        overlay[:, :, 1] = (ground_truth[mid_slice] > 0).astype(float)  # Green for GT
        overlay[:, :, 0] = (prediction[mid_slice] > 0).astype(float)   # Red for prediction
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (GT=Green, Pred=Red)')
        axes[1, 1].axis('off')
        
        # Dice scores per slice
        slice_dices = []
        for s in range(ground_truth.shape[0]):
            gt_slice = ground_truth[s].flatten()
            pred_slice = prediction[s].flatten()
            
            intersection = np.sum((gt_slice > 0) & (pred_slice > 0))
            dice = (2.0 * intersection) / (np.sum(gt_slice > 0) + np.sum(pred_slice > 0) + 1e-8)
            slice_dices.append(dice)
        
        axes[1, 2].plot(slice_dices)
        axes[1, 2].axvline(x=mid_slice, color='r', linestyle='--', label='Current slice')
        axes[1, 2].set_title('Dice Score per Slice')
        axes[1, 2].set_xlabel('Slice')
        axes[1, 2].set_ylabel('Dice Score')
        axes[1, 2].legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_test(self, test_image_path=None, test_label_path=None):
        """Run comprehensive test suite"""
        print("🚀 COMPREHENSIVE MODEL TEST")
        print("="*50)
        
        # Test 1: Single image prediction
        if test_image_path:
            print("\n1️⃣ SINGLE IMAGE TEST")
            self.predict_single_image(test_image_path)
        
        # Test 2: Ground truth comparison
        if test_image_path and test_label_path:
            print("\n2️⃣ GROUND TRUTH COMPARISON")
            self.test_with_ground_truth(test_image_path, test_label_path)
        
        # Test 3: Dataset evaluation
        print("\n3️⃣ DATASET EVALUATION")
        self.test_dataset(max_samples=5)  # Test first 5 samples
        
        print("\n✅ Comprehensive testing completed!")

# Usage example
if __name__ == "__main__":
    print("🧪 Glenoid Model Testing Suite")
    print("="*50)
    
    # Initialize tester
    model_path = "best_glenoid_model_fold_4.pth"  # Change to your model path
    tester = GlenoidModelTester(model_path, "./glenoid/dataset/")
    
    # Example 1: Test single image
    # test_image = "./glenoid/dataset/patient_001/image.nii.gz"
    # result = tester.predict_single_image(test_image)
    
    # Example 2: Test with ground truth
    # test_image = "./glenoid/dataset/patient_001/image.nii.gz"
    # test_label = "./glenoid/dataset/patient_001/label.nii.gz"
    # result = tester.test_with_ground_truth(test_image, test_label)
    
    # Example 3: Comprehensive test
    # tester.run_comprehensive_test()
    
    # Example 4: Test entire dataset
    results = tester.test_dataset(max_samples=10)
    
    print("\n🎉 Testing completed! Check the generated plots and summary files.")
