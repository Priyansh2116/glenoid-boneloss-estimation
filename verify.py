import SimpleITK as sitk

img = sitk.ReadImage("./glenoid/dataset/patient8/image.nii.gz")
lbl = sitk.ReadImage("./glenoid/dataset/patient8/label.nii.gz")

img_arr = sitk.GetArrayFromImage(img)
lbl_arr = sitk.GetArrayFromImage(lbl)

print("Image shape:", img_arr.shape)
print("Label shape:", lbl_arr.shape)

assert img_arr.shape == lbl_arr.shape, "Mismatch in shape"

