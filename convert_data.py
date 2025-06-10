import SimpleITK as sitk

# Load .nrrd
image = sitk.ReadImage("./dataset/patient13/label.nrrd")

# Save as .nii.gz
sitk.WriteImage(image, "./dataset/patient13/label.nii.gz")

