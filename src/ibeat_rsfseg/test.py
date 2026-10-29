import os
import dbdicom as db
import stage_5_display as s5

base = r"X:\abdominal_imaging\Shared\Benthe"
b = base + r"\iBEAt_Build"
dp = base + r"\dixon\stage_5_clean_dixon_data\Controls"
mp = b + r"\rsfseg\stage_4_edit\Controls\rsf_masks"
kp = b + r"\kidneyvol\stage_3_edit\Controls"
out = b + r"\rsfseg\stage_5_display\Controls"

s5.run_site(dp, mp, kp, out, task="rsf_masks_edited", n_jobs=1)

target = os.path.join(out, "mosaic_fat_fraction_rsf_masks_edited")
print(os.listdir(target))