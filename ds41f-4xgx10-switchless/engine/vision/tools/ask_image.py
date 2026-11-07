#!/usr/bin/env python3
"""ask_image.py IMAGE QUESTION [MAX_TOKENS] [MODEL] [URL]: one image + question through the proxy, greedy."""
import base64, json, sys, time, urllib.request
path, q = sys.argv[1], sys.argv[2]
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 200
model = sys.argv[4] if len(sys.argv) > 4 else "DeepSeek-v4.1-Flash"
url = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:8000/v1/chat/completions"
kind = "png" if path.endswith("png") else "jpeg"
u = f"data:image/{kind};base64," + base64.b64encode(open(path, "rb").read()).decode()
body = {"model": model, "temperature": 0, "max_tokens": mt, "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}, {"type": "text", "text": q}]}]}
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=600))
    print(f"--- {path} ({time.time() - t:.1f}s, prompt {r['usage']['prompt_tokens']} tok)")
    print(r["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print(path, e.code, e.read()[:500].decode(errors="replace"))
