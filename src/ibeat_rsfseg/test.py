import SimpleITK as sitk, glob, os

d = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build\rsfseg\stage_8_build_training_data\Dataset001_iBEAtRSF"

def size(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()   # alleen header, geen pixeldata
    return r.GetSize()

for lbl in sorted(glob.glob(os.path.join(d, "labelsTr", "*.nii.gz"))):
    case = os.path.basename(lbl)[:-7]
    ls = size(lbl)
    for ch in sorted(glob.glob(os.path.join(d, "imagesTr", f"{case}_000*.nii.gz"))):
        s = size(ch)
        if s != ls:
            print(case, os.path.basename(ch), s, ls, flush=True)