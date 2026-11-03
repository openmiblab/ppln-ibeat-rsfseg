import os, json, time

folder = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build\dixon\stage_5_clean_dixon_data\Patients\Turku"
ip = os.path.join(folder, "index.json")

print("index.json bestaat:", os.path.exists(ip))
if os.path.exists(ip):
    print("grootte (MB):", os.path.getsize(ip) / 1e6)
    print("laatst gewijzigd:", time.ctime(os.path.getmtime(ip)))
    try:
        with open(ip, encoding="utf-8") as f:
            idx = json.load(f)
        print("leesbaar, type:", type(idx), "lengte:", len(idx))
    except Exception as e:
        print("NIET leesbaar:", repr(e))

# aantal .dcm-bestanden in de map (snel, zonder te openen)
n = 0
for root, dirs, files in os.walk(folder):
    n += sum(f.lower().endswith(".dcm") for f in files)
print("aantal .dcm-bestanden:", n)