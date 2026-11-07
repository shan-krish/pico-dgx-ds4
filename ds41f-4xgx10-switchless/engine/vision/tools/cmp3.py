import sys, numpy as np
truth = np.fromfile(sys.argv[1], np.float32).reshape(-1, 5120)
for name in sys.argv[2:]:
    a = np.fromfile(name, np.float32).reshape(-1, 5120)
    cos = (a * truth).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(truth, axis=1)
    g = (a * truth).sum() / np.linalg.norm(a) / np.linalg.norm(truth)
    print(f"{name:28s} vs fp32: cosine {g:.6f}, worst row {cos.min():.6f}, 1st pct {np.percentile(cos, 1):.6f}, mean|d| {np.abs(a - truth).mean():.5f}")
