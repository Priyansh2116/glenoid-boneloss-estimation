import os
import sys
from pathlib import Path

def interactive_model_tester():
    """Interactive interface for testing the glenoid segmentation model"""
    
    print("🧪 GLENOID MODEL TESTING INTERFACE")
    print("="*50)
    
    # Check if model exists
    available_models = []
    for file in os.listdir('.'):
        if file.endswith('.pth') and 'glenoid' in file.lower():
            available_models.append(file)
    
    if not available_models:
        print("❌ No model files found! Please ensure your trained model (.pth file) is in the current directory.")
        print("Expected files: best_glenoid_model_fold_*.pth or similar")
        return
    
    print("📦 Available Models:")
    for i, model in enumerate(available_models):
        print(f"   {i+1}. {model}")
    
    # Select model
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}): ")
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"✅ Selected model: {selected_model}")
    
    # Import and initialize tester
    try:
        # Assuming the tester class is available
        from glenoid_model_tester import GlenoidModelTester
        tester = GlenoidModelTester(selected_model)
    except ImportError:
        print("❌ Could not import GlenoidModelTester. Please ensure the testing script is available.")
        return
    
    # Main testing loop
    while True:
        print("\n" + "="*50)
        print("🔬 TESTING OPTIONS")
        print("="*50)
        print("1. Test single image (prediction only)")
        print("2. Test with ground truth comparison")
        print("3. Test multiple images from dataset")
        print("4. Quick performance benchmark")
        print("5. Model architecture summary")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            test_single_image(tester)
        elif choice == '2':
            test_with_ground_truth(tester)
        elif choice == '3':
            test_dataset(tester)
        elif choice == '4':
            performance_benchmark(tester)
        elif choice == '5':
            model_summary(tester)
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def test_single_image(tester):
    """Test single image interface"""
    print("\n📸 SINGLE IMAGE TEST")
    print("-" * 30)
    
    image_path = input("Enter path to image file (.nii.gz): ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    if not image_path.endswith('.nii.gz'):
        print("⚠️  Warning: Expected .nii.gz file")
    
    try:
        print("🔄 Processing...")
        result = tester.predict_single_image(image_path, visualize=True)
        
        # Save prediction
        save_choice = input("\n💾 Save prediction? (y/n): ").lower()
        if save_choice == 'y':
            import numpy as np
            output_path = image_path.replace('.nii.gz', '_prediction.npy')
            np.save(output_path, result['prediction'])
            print(f"✅ Prediction saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def test_with_ground_truth(tester):
    """Test with ground truth interface"""
    print("\n🎯 GROUND TRUTH COMPARISON")
    print("-" * 30)
    
    image_path = input("Enter path to image file (.nii.gz): ").strip()
    label_path = input("Enter path to label file (.nii.gz): ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return
    
    try:
        print("🔄 Processing...")
        result = tester.test_with_ground_truth(image_path, label_path)
        
        if result:
            # Show detailed metrics
            metrics = result['metrics']
            print(f"\n📊 DETAILED METRICS:")
            print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Mean Dice Score: {metrics['mean_dice']:.4f}")
            
            # Volume comparison
            print(f"\n📏 VOLUME ANALYSIS:")
            true_vols = result['true_volumes']
            pred_vols = result['pred_volumes']
            
            for class_name in ['Left Glenoid', 'Right Glenoid']:
                if true_vols[class_name] > 0:
                    true_vol = true_vols[class_name]
                    pred_vol = pred_vols[class_name]
                    error = abs(pred_vol - true_vol)
                    rel_error = (error / true_vol * 100)
                    print(f"   {class_name}:")
                    print(f"     Ground Truth: {true_vol:.1f} mm³")
                    print(f"     Prediction: {pred_vol:.1f} mm³")
                    print(f"     Error: {error:.1f} mm³ ({rel_error:.1f}%)")
            
            # Clinical interpretation
            print(f"\n🏥 CLINICAL ASSESSMENT:")
            left_true = true_vols['Left Glenoid']
            right_true = true_vols['Right Glenoid']
            left_pred = pred_vols['Left Glenoid']
            right_pred = pred_vols['Right Glenoid']
            
            if left_true > 0 and right_true > 0:
                true_asymmetry = abs(left_true - right_true) / max(left_true, right_true) * 100
                pred_asymmetry = abs(left_pred - right_pred) / max(left_pred, right_pred) * 100
                print(f"   True L/R Asymmetry: {true_asymmetry:.1f}%")
                print(f"   Predicted L/R Asymmetry: {pred_asymmetry:.1f}%")
                print(f"   Asymmetry Error: {abs(true_asymmetry - pred_asymmetry):.1f}%")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

def test_dataset(tester):
    """Test dataset interface"""
    print("\n📊 DATASET TESTING")
    print("-" * 30)
    
    dataset_path = input("Enter dataset path (or press Enter for default ./glenoid/dataset/): ").strip()
    if not dataset_path:
        dataset_path = "./glenoid/dataset/"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Count available patients
    patients = [f for f in os.listdir(dataset_path) 
               if f.startswith('patient') and os.path.isdir(os.path.join(dataset_path, f))]
    
    if not patients:
        print(f"❌ No patient folders found in: {dataset_path}")
        return
    
    print(f"📁 Found {len(patients)} patient folders")
    
    while True:
        try:
            max_samples = input(f"How many samples to test? (1-{len(patients)}, or 'all'): ").strip()
            if max_samples.lower() == 'all':
                max_samples = len(patients)
                break
            else:
                max_samples = int(max_samples)
                if 1 <= max_samples <= len(patients):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(patients)}")
        except ValueError:
            print("Please enter a valid number or 'all'")
    
    print(f"🔄 Testing {max_samples} samples...")
    
    try:
        results = tester.test_dataset(test_folder=dataset_path, max_samples=max_samples)
        
        print(f"\n✅ Dataset testing completed!")
        print(f"📊 Tested {len(results)} samples")
        print(f"📄 Detailed results saved to 'glenoid_test_summary.json'")
        
        # Quick summary
        if results:
            inference_times = [r.get('inference_time', 0) for r in results if 'inference_time' in r]
            if inference_times:
                avg_time = sum(inference_times) / len(inference_times)
                print(f"⏱️  Average inference time: {avg_time:.3f} seconds")
        
    except Exception as e:
        print(f"❌ Error during dataset testing: {e}")

def performance_benchmark(tester):
    """Performance benchmark interface"""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("-" * 30)
    
    # Create a dummy image for benchmarking
    import torch
    import time
    
    print("🔄 Running performance benchmark...")
    
    # Warm up
    dummy_input = torch.randn(1, 1, 64, 128, 128).to(tester.device)
    with torch.no_grad():
        for _ in range(5):
            _ = tester.model(dummy_input)
    
    # Benchmark
    times = []
    num_runs = 20
    
    torch.cuda.synchronize() if tester.device.type == 'cuda' else None
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            _ = tester.model(dummy_input)
        
        torch.cuda.synchronize() if tester.device.type == 'cuda' else None
        end_time = time.time()
        
        times.append(end_time - start_time)
        print(f"Run {i+1}/{num_runs}: {times[-1]:.4f}s", end='\r')
    
    print()  # New line
    
    # Statistics
    import numpy as np
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   Mean inference time: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"   Min/Max time: {min_time:.4f} / {max_time:.4f} seconds")
    print(f"   Throughput: {1/mean_time:.2f} images/second")
    
    # Memory usage
    if tester.device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"   GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    # Model parameters
    total_params = sum(p.numel() for p in tester.model.parameters())
    trainable_params = sum(p.numel() for p in tester.model.parameters() if p.requires_grad)
    
    print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")

def model_summary(tester):
    """Model architecture summary"""
    print("\n🏗️  MODEL ARCHITECTURE SUMMARY")
    print("-" * 40)
    
    # Basic info
    total_params = sum(p.numel() for p in tester.model.parameters())
    trainable_params = sum(p.numel() for p in tester.model.parameters() if p.requires_grad)
    
    print(f"Model Type: 3D U-Net")
    print(f"Input Shape: (1, 64, 128, 128)")
    print(f"Output Classes: 3 (Background, Left Glenoid, Right Glenoid)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    # Architecture details
    print(f"\n📐 ARCHITECTURE DETAILS:")
    print(f"   Encoder Features: [32, 64, 128, 256]")
    print(f"   Decoder: Transpose convolutions + skip connections")
    print(f"   Activation: ReLU + BatchNorm")
    print(f"   Final Layer: 1x1x1 Conv3D")
    
    # Try to show detailed summary if torchsummary is available
    try:
        from torchsummary import summary
        print(f"\n📋 DETAILED LAYER SUMMARY:")
        summary(tester.model, (1, 64, 128, 128), device=str(tester.device))
    except ImportError:
        print(f"\n💡 Install torchsummary for detailed layer information:")
        print(f"   pip install torchsummary")
    
    # Training info (if available)
    model_path = tester.model_path
    if os.path.exists(model_path):
        import os
        file_size = os.path.getsize(model_path) / 1024**2  # MB
        mod_time = os.path.getmtime(model_path)
        import datetime
        mod_date = datetime.datetime.fromtimestamp(mod_time)
        
        print(f"\n📁 MODEL FILE INFO:")
        print(f"   File: {model_path}")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main entry point"""
    try:
        interactive_model_tester()
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
