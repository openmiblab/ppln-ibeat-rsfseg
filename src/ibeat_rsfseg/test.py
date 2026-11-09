import json

p = "/mnt/parscratch/users/cmr26ba/data/iBEAt_Build/rsfseg/stage_8_build_training_data/Dataset001_iBEAtRSF/dataset.json"
dj = json.load(open(p))
dj["numTraining"] = 703
json.dump(dj, open(p, "w"), indent=2)
print(dj["numTraining"])