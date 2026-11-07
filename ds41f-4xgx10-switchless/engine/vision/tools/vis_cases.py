#!/usr/bin/env python3
"""Vision cases through the proxy: cache reuse across turns, a different image under the same text, two
images in one request, streaming, and refused inputs."""
import base64, json, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
M = "DeepSeek-v4.1-Flash"
def durl(p):
    return f"data:image/{'png' if p.endswith('png') else 'jpeg'};base64," + base64.b64encode(open(p, "rb").read()).decode()
def post(messages, mt=80, stream=False):
    body = {"model": M, "temperature": 0, "max_tokens": mt, "messages": messages, "chat_template_kwargs": {"thinking": False}}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", 0
    if not stream:
        j = json.load(r)
        return j, j["choices"][0]["message"]["content"], time.time() - t
    text, usage = "", None
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices") or []:
                text += (c.get("delta") or {}).get("content") or ""
    return {"usage": usage}, text, time.time() - t
def cached(j):
    u = j["usage"]
    return f"prompt {u['prompt_tokens']}, cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')}"
img = lambda p: {"type": "image_url", "image_url": {"url": durl(p)}}
txt = lambda s: {"type": "text", "text": s}

print("== 1. turn one, then a follow-up on the same image (expect the image's tokens cached)")
m1 = [{"role": "user", "content": [img("shapes.png"), txt("What color is the circle? One word.")]}]
j, a, dt = post(m1, 20)
print(f"  turn 1 ({dt:.1f}s, {cached(j)}): {a!r}")
m2 = m1 + [{"role": "assistant", "content": a}, {"role": "user", "content": "And the triangle? One word."}]
j, b, dt = post(m2, 20)
print(f"  turn 2 ({dt:.1f}s, {cached(j)}): {b!r}")

print("== 2. same text, a different image (expect no reuse past the image's start, and the new image's answer)")
m3 = [{"role": "user", "content": [img("big.jpg"), txt("What color is the circle? One word.")]}]
j, c, dt = post(m3, 40)
print(f"  ({dt:.1f}s, {cached(j)}): {c!r}")

print("== 3. two images in one request")
m4 = [{"role": "user", "content": [txt("Image A:"), img("shapes.png"), txt("Image B:"), img("big.jpg"),
                                   txt("What text is written in image A, and what text in image B? Answer as 'A: ..., B: ...'.")]}]
j, d, dt = post(m4, 40)
print(f"  ({dt:.1f}s, {cached(j)}): {d!r}")

print("== 4. streaming")
j, e, dt = post([{"role": "user", "content": [img("tall.png"), txt("Describe this image in one sentence.")]}], 60, stream=True)
print(f"  ({dt:.1f}s, usage {j['usage']['prompt_tokens'] if j and j.get('usage') else None}): {e!r}")

print("== 5. refused inputs")
for name, content in [("not an image", [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"hello").decode()}}, txt("hi")]),
                      ("a local path", [{"type": "image_url", "image_url": {"url": "/etc/passwd"}}, txt("hi")]),
                      ("audio", [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}, txt("hi")])]:
    j, f, dt = post([{"role": "user", "content": content}], 10)
    print(f"  {name}: {f[:160]!r}")

print("== 6. text after all that (no image): unchanged")
j, g, dt = post([{"role": "user", "content": "What is 19 + 23? Answer with the number."}], 10)
print(f"  {g!r}")
