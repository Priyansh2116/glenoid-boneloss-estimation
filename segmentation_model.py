import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import SimpleITK as sitk
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GlenoidDataset(Dataset):
    """Dataset class for glenoid segmentation"""
    
    def __init__(self, patient_data, transform=None, augment=False):
        self.patient_data = patient_data  # List of (image_path, label_path, patient_id)
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.patient_data)
    
    def __getitem__(self, idx):
        image_path, label_path, patient_id = self.patient_data[idx]
        
        # Load preprocessed data
        if os.path.exists(image_path.replace('.nii.gz', '_processed.npy')):
            image = np.load(image_path.replace('.nii.gz', '_processed.npy'))
            label = np.load(label_path.replace('.nii.gz', '_processed.npy'))
        else:
            # Load and preprocess on the fly
            image, label = self.load_and_preprocess(image_path, label_path)
        
        # Convert to tensors
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor(label)
        
        # Apply augmentations if enabled
        if self.augment and torch.rand(1) > 0.5:
            image, label = self.apply_augmentation(image, label)
        
        return image, label, patient_id
    
    def load_and_preprocess(self, image_path, label_path):
        """Load and preprocess image/label pair"""
        # Load with SimpleITK
        img = sitk.ReadImage(image_path)
        lbl = sitk.ReadImage(label_path)
        
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)
        lbl_array = sitk.GetArrayFromImage(lbl).astype(np.int64)
        
        # Intensity normalization (bone window)
        img_array = np.clip(img_array, -175, 1225)  # Bone window: 400±750 HU
        img_array = (img_array + 175) / 1400.0  # Normalize to [0,1]
        
        # Resize to target size (128x128x64)
        target_shape = (64, 128, 128)  # (D, H, W)
        
        # Simple resize using zoom
        from scipy import ndimage
        zoom_factors = [target_shape[i] / img_array.shape[i] for i in range(3)]
        
        img_resized = ndimage.zoom(img_array, zoom_factors, order=1)
        lbl_resized = ndimage.zoom(lbl_array, zoom_factors, order=0)
        
        return img_resized, lbl_resized
    
    def apply_augmentation(self, image, label):
        """Apply random augmentations"""
        # Random flip
        if torch.rand(1) > 0.5:
            axis = torch.randint(1, 4, (1,)).item()  # Skip batch/channel dim
            image = torch.flip(image, [axis])
            label = torch.flip(label, [axis-1])  # Label has no channel dim
        
        # Random rotation (small angles)
        if torch.rand(1) > 0.5:
            angle = torch.randn(1).item() * 10  # ±10 degrees
            # Note: For full implementation, use torchio or custom rotation
        
        # Add noise
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
        
        return image, label

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

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(num_classes):
            pred_i = predictions[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            dice = (2. * intersection + self.smooth) / (pred_i.sum() + target_i.sum() + self.smooth)
            dice_scores.append(dice)
        
        # Return 1 - mean dice (so we minimize)
        return 1 - torch.mean(torch.stack(dice_scores))

class GlenoidSegmentationTrainer:
    """Training pipeline for glenoid segmentation"""
    
    def __init__(self, dataset_path="./glenoid/dataset/", device=None):
        self.dataset_path = dataset_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Using device: {self.device}")
        
        # Initialize model
        self.model = UNet3D(in_channels=1, num_classes=3)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
        # Training history
        self.train_history = {'loss': [], 'dice': []}
        self.val_history = {'loss': [], 'dice': []}
        
    def prepare_data(self):
        """Prepare training and validation data"""
        print("📊 Preparing dataset...")
        
        # Get all patient data
        patient_data = []
        patients = [f for f in os.listdir(self.dataset_path) 
                   if f.startswith('patient') and os.path.isdir(os.path.join(self.dataset_path, f))]
        
        for patient_folder in patients:
            patient_path = os.path.join(self.dataset_path, patient_folder)
            
            image_path = os.path.join(patient_path, 'image.nii.gz')
            label_path = os.path.join(patient_path, 'label.nii.gz')
            
            if os.path.exists(image_path) and os.path.exists(label_path):
                patient_data.append((image_path, label_path, patient_folder))
        
        print(f"Found {len(patient_data)} patients")
        return patient_data
    
    def train_with_cross_validation(self, k_folds=4, epochs=50):
        """Train model with k-fold cross-validation"""
        print(f"🔄 Starting {k_folds}-fold cross-validation training...")
        
        patient_data = self.prepare_data()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(patient_data)):
            print(f"\n{'='*20} FOLD {fold+1}/{k_folds} {'='*20}")
            
            # Split data
            train_data = [patient_data[i] for i in train_idx]
            val_data = [patient_data[i] for i in val_idx]
            
            print(f"Train patients: {len(train_data)}, Val patients: {len(val_data)}")
            
            # Create datasets
            train_dataset = GlenoidDataset(train_data, augment=True)
            val_dataset = GlenoidDataset(val_data, augment=False)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
            
            # Reset model for each fold
            self.model = UNet3D(in_channels=1, num_classes=3).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
            # Train for this fold
            fold_history = self.train_fold(train_loader, val_loader, epochs, fold)
            fold_results.append(fold_history)
            
            # Save fold model
            torch.save(self.model.state_dict(), f'glenoid_model_fold_{fold+1}.pth')
        
        # Analyze cross-validation results
        self.analyze_cv_results(fold_results)
        return fold_results
    
    def train_fold(self, train_loader, val_loader, epochs, fold):
        """Train model for one fold"""
        best_val_dice = 0.0
        fold_history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            
            train_bar = tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs}')
            for batch_idx, (images, labels, patient_ids) in enumerate(train_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                
                # Combined loss: CrossEntropy + Dice
                ce_loss = self.criterion(outputs, labels)
                dice_loss = self.dice_loss(outputs, labels)
                loss = ce_loss + dice_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_dice = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(self.model.state_dict(), f'best_glenoid_model_fold_{fold+1}.pth')
            
            # Record history
            fold_history['train_loss'].append(avg_train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_dice'].append(val_dice)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
            
            # Early stopping
            if epoch > 20 and val_dice < best_val_dice - 0.1:
                print("Early stopping triggered")
                break
        
        return fold_history
    
    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for images, labels, patient_ids in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                
                # Calculate losses
                ce_loss = self.criterion(outputs, labels)
                dice_loss = self.dice_loss(outputs, labels)
                loss = ce_loss + dice_loss
                
                # Calculate Dice score
                dice_score = 1 - dice_loss.item()
                
                total_loss += loss.item()
                total_dice += dice_score
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return avg_loss, avg_dice
    
    def analyze_cv_results(self, fold_results):
        """Analyze cross-validation results"""
        print("\n" + "="*50)
        print("📊 CROSS-VALIDATION RESULTS")
        print("="*50)
        
        # Calculate mean and std for each metric
        final_dice_scores = [fold['val_dice'][-1] for fold in fold_results]
        best_dice_scores = [max(fold['val_dice']) for fold in fold_results]
        
        print(f"Final Dice Scores: {final_dice_scores}")
        print(f"Mean Final Dice: {np.mean(final_dice_scores):.4f} ± {np.std(final_dice_scores):.4f}")
        print(f"Best Dice Scores: {best_dice_scores}")
        print(f"Mean Best Dice: {np.mean(best_dice_scores):.4f} ± {np.std(best_dice_scores):.4f}")
        
        # Save results
        results = {
            'fold_results': fold_results,
            'final_dice_mean': float(np.mean(final_dice_scores)),
            'final_dice_std': float(np.std(final_dice_scores)),
            'best_dice_mean': float(np.mean(best_dice_scores)),
            'best_dice_std': float(np.std(best_dice_scores)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('glenoid_cv_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to 'glenoid_cv_results.json'")
    
    def predict(self, image_path, model_path=None):
        """Predict segmentation for a new image"""
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # Load and preprocess image
        dataset = GlenoidDataset([(image_path, None, 'test')], augment=False)
        image, _, _ = dataset[0]
        
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        return prediction
    
    def plot_training_history(self, fold_results):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot each fold
        for i, fold_history in enumerate(fold_results):
            epochs = range(1, len(fold_history['train_loss']) + 1)
            
            axes[0, 0].plot(epochs, fold_history['train_loss'], label=f'Fold {i+1}', alpha=0.7)
            axes[0, 1].plot(epochs, fold_history['val_loss'], label=f'Fold {i+1}', alpha=0.7)
            axes[1, 0].plot(epochs, fold_history['val_dice'], label=f'Fold {i+1}', alpha=0.7)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Validation Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('glenoid_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    print("🧠 Glenoid Segmentation Model Training")
    print("="*50)
    
    # Initialize trainer
    trainer = GlenoidSegmentationTrainer("./glenoid/dataset/")
    
    # Check if CUDA is available
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Start training
    print("\n🚀 Starting cross-validation training...")
    fold_results = trainer.train_with_cross_validation(k_folds=4, epochs=50)
    
    # Plot results
    trainer.plot_training_history(fold_results)
    
    print("\n✅ Training completed! Check the saved models and results.")
