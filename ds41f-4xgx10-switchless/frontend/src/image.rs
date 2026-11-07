//! Images for the vision tower (ref/inference/image_processor.py): load an image record's bytes, decode
//! them, plan the patch and token grids, and letterbox the picture onto the patch grid as the reference
//! does (`ImageOps.pad` with Pillow's bicubic resampling, grey 127 bars). The engine gets the letterboxed
//! uint8 RGB pixels and the grids; it runs the ViT and the aligner itself.
//!
//! Pillow's resampling (libImaging/Resample.c, 12.3.0) is ported exactly: the same coefficients, the same
//! fixed-point rounding and the uint8 image between the horizontal and the vertical pass, so the pixels
//! are the reference's bit for bit for any decoded image. Decoding is the `image` crate's: PNG, GIF, BMP
//! and WebP give Pillow's pixels; a JPEG's can differ from libjpeg-turbo's by a level here and there.

use serde_json::Value;

pub const PATCH: usize = 14;
pub const DOWN: usize = 3;
pub const MAX_TOKENS: usize = 1024;
pub const MIN_PIXELS: f64 = (544 * 544) as f64;
/// A single image's encoded bytes may not exceed this.
pub const MAX_BYTES: usize = 32 << 20;
/// Nor its decoded size, in pixels (a decompression bomb is refused before it is decoded).
pub const MAX_PIXELS: u64 = 64 << 20;

/// One image, ready for the engine.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    /// The letterboxed picture: uint8 RGB, (n_vit_h * 14) rows of (n_vit_w * 14) pixels.
    pub pixels: Vec<u8>,
    /// A digest of the encoded bytes: what tells two images apart in the session caches.
    pub digest: u64,
}

impl Prepared {
    /// Positions the image takes in the prompt: IMAGE_START, n_llm_h rows of n_llm_w IMAGE and one
    /// IMAGE_NEW_LINE, IMAGE_END.
    pub fn span(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }
}

pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

fn llm_grid(best_height: usize, best_width: usize) -> (usize, usize) {
    // math.ceil((best // patch) / ratio)
    ((best_height / PATCH).div_ceil(DOWN), (best_width / PATCH).div_ceil(DOWN))
}

/// Python's round() of a float: half to even, to an integer.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// image_processor.solve_resize_ratio
fn solve_resize_ratio(height: f64, width: f64) -> (usize, usize) {
    let r = height / width;
    let max_w_float = ((MAX_TOKENS as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = (PATCH * DOWN) as f64;
    if max_w_float < 1.0 {
        return ((MAX_TOKENS - 2) / 2 * PATCH * DOWN, PATCH * DOWN);
    }
    if max_h_float < 1.0 {
        return (PATCH * DOWN, (MAX_TOKENS - 3) * PATCH * DOWN);
    }
    let beta = (max_w_float.floor() * cell / width).min(max_h_float.floor() * cell / height);
    let p = PATCH as f64;
    (((height * beta / p).floor() * p) as usize, ((width * beta / p).floor() * p) as usize)
}

/// image_processor.plan_image_grid (vision_max_wh_ratio is None): (n_llm_h, n_llm_w, best_height, best_width)
pub fn plan_image_grid(width: usize, height: usize) -> (usize, usize, usize, usize) {
    let (mut w, mut h) = (width as f64, height as f64);
    let area = width as f64 * height as f64;
    if 0.0 < area && area < MIN_PIXELS {
        let ratio = (MIN_PIXELS / area).sqrt();
        w = (width as f64 * ratio).trunc();
        h = (height as f64 * ratio).trunc();
    }
    let p = PATCH as f64;
    let best_width = ((w / p).ceil() * p) as usize;
    let best_height = ((h / p).ceil() * p) as usize;
    // safe_resize
    let (n_llm_h, n_llm_w) = llm_grid(best_height, best_width);
    if num_image_tokens(n_llm_h, n_llm_w) <= MAX_TOKENS {
        return (n_llm_h, n_llm_w, best_height, best_width);
    }
    let (bh, bw) = solve_resize_ratio(h, w);
    let (n_llm_h, n_llm_w) = llm_grid(bh, bw);
    (n_llm_h, n_llm_w, bh, bw)
}

// ---------------------------------------------------------------- Pillow's bicubic resampling

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// precompute_coeffs + normalize_coeffs_8bpc for the whole axis (box [0, in_size)): per output index its
/// first input index, the count, and the fixed-point coefficients (ksize of them).
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_size as f32 as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv = 1.0 / filterscale;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let n = xmax.saturating_sub(xmin);
        let mut ww = 0.0;
        for x in 0..n {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * inv);
            k[x] = w;
            ww += w;
        }
        for x in 0..ksize {
            let v = if x < n { if ww != 0.0 { k[x] / ww } else { k[x] } } else { 0.0 };
            let f = v * (1u32 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, n));
    }
    (ksize, bounds, kk)
}

fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Image.resize((out_w, out_h), BICUBIC) of an RGB image (ImagingResampleInner): horizontal pass into a uint8
/// image of the rows the vertical pass reads, then the vertical pass.
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    if (out_w, out_h) == (w, h) {
        return src.to_vec();
    }
    let need_h = out_w != w;
    let need_v = out_h != h;
    let (ksv, bv, kv) = coeffs(h, out_h);
    let first = bv[0].0;
    let last = bv[out_h - 1].0 + bv[out_h - 1].1;
    let (tmp, tw, t0) = if need_h {
        let (ksh, bh, kh) = coeffs(w, out_w);
        let rows = last - first;
        let mut t = vec![0u8; rows * out_w * 3];
        for yy in 0..rows {
            let line = &src[(yy + first) * w * 3..(yy + first + 1) * w * 3];
            for xx in 0..out_w {
                let (xmin, n) = bh[xx];
                let k = &kh[xx * ksh..xx * ksh + n];
                let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &c) in k.iter().enumerate() {
                    let p = &line[(x + xmin) * 3..(x + xmin) * 3 + 3];
                    for ch in 0..3 {
                        ss[ch] += p[ch] as i32 * c;
                    }
                }
                for ch in 0..3 {
                    t[(yy * out_w + xx) * 3 + ch] = clip8(ss[ch]);
                }
            }
        }
        (t, out_w, first)
    } else {
        (src.to_vec(), w, 0)
    };
    if !need_v {
        return tmp;
    }
    let mut out = vec![0u8; out_h * tw * 3];
    for yy in 0..out_h {
        let (ymin, n) = bv[yy];
        let k = &kv[yy * ksv..yy * ksv + n];
        for xx in 0..tw {
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for (y, &c) in k.iter().enumerate() {
                let p = &tmp[((y + ymin - t0) * tw + xx) * 3..((y + ymin - t0) * tw + xx) * 3 + 3];
                for ch in 0..3 {
                    ss[ch] += p[ch] as i32 * c;
                }
            }
            for ch in 0..3 {
                out[(yy * tw + xx) * 3 + ch] = clip8(ss[ch]);
            }
        }
    }
    out
}

/// ImageOps.pad(image, (bw, bh), color=(127, 127, 127)): ImageOps.contain, then the picture centred on grey.
pub fn letterbox(src: &[u8], w: usize, h: usize, bw: usize, bh: usize) -> Vec<u8> {
    let im_ratio = w as f64 / h as f64;
    let dest_ratio = bw as f64 / bh as f64;
    let (mut rw, mut rh) = (bw, bh);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let nh = py_round(h as f64 / w as f64 * bw as f64) as usize;
            if nh != bh {
                rh = nh;
            }
        } else {
            let nw = py_round(w as f64 / h as f64 * bh as f64) as usize;
            if nw != bw {
                rw = nw;
            }
        }
    }
    let resized = resize_bicubic(src, w, h, rw, rh);
    if (rw, rh) == (bw, bh) {
        return resized;
    }
    let mut out = vec![127u8; bw * bh * 3];
    let (x0, y0) = if rw != bw { (py_round((bw - rw) as f64 * 0.5) as usize, 0) } else { (0, py_round((bh - rh) as f64 * 0.5) as usize) };
    for y in 0..rh {
        let dst = ((y + y0) * bw + x0) * 3;
        out[dst..dst + rw * 3].copy_from_slice(&resized[y * rw * 3..(y + 1) * rw * 3]);
    }
    out
}

// ---------------------------------------------------------------- loading

fn b64(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let s: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(s.trim_end_matches('=').as_bytes()))
        .map_err(|e| format!("image: bad base64: {e}"))
}

fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url)
        .config()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .call()
        .map_err(|e| format!("image: cannot fetch {url}: {e}"))?;
    let mut body = resp.into_body();
    body.with_config().limit(MAX_BYTES as u64).read_to_vec().map_err(|e| format!("image: cannot read {url}: {e}"))
}

/// image_processor.load_image_bytes for an image record (encoding.rs extract_image's): raw or base64
/// data, an Anthropic source, a data: or http(s) URL. Local paths are refused: the requester is remote.
pub fn load_bytes(record: &Value) -> Result<Vec<u8>, String> {
    if let Some(d) = record.get("data").and_then(Value::as_str) {
        return b64(d);
    }
    if let Some(src) = record.get("source").filter(|v| v.is_object()) {
        if let Some(d) = src.get("data").and_then(Value::as_str) {
            return b64(d);
        }
        if let Some(u) = src.get("url").and_then(Value::as_str) {
            return load_url(u);
        }
    }
    if let Some(u) = record.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()) {
        return load_url(u);
    }
    Err("image: no data, source or url in the image block".into())
}

fn load_url(url: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, payload) = rest.split_once(',').ok_or("image: a data: URL without a comma")?;
        if !header.contains(";base64") {
            return Err(format!("image: unsupported data URL encoding: data:{header}"));
        }
        return b64(payload);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch(url);
    }
    Err("image: only data: and http(s) URLs are accepted".into())
}

pub fn digest(bytes: &[u8]) -> u64 {
    // FNV-1a over the bytes, then a 64-bit mix: stable across runs and builds
    let mut h = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

/// Decode, convert to RGB (alpha dropped, as Pillow's convert("RGB")), plan the grids and letterbox.
pub fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    if bytes.len() > MAX_BYTES {
        return Err(format!("image: {} bytes, more than {MAX_BYTES}", bytes.len()));
    }
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: {e}"))?;
    let (w, h) = reader.into_dimensions().map_err(|e| format!("image: cannot read its size: {e}"))?;
    if w == 0 || h == 0 || w as u64 * h as u64 > MAX_PIXELS {
        return Err(format!("image: {w}x{h} pixels is out of range"));
    }
    let img = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: {e}"))?
        .decode()
        .map_err(|e| format!("image: cannot decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (n_llm_h, n_llm_w, bh, bw) = plan_image_grid(w, h);
    let pixels = letterbox(rgb.as_raw(), w, h, bw, bh);
    Ok(Prepared { n_vit_h: bh / PATCH, n_vit_w: bw / PATCH, n_llm_h, n_llm_w, pixels, digest: digest(bytes) })
}

/// <｜deepseek_image｜>: the placeholder encoding.rs puts in the prompt, one per image.
pub const IMAGE_TOKEN: u32 = 129264;

/// The id every position of an image's span carries: 0x80000000 | 30 bits of its digest. The engine reads a
/// negative id as an image position; the digest keeps two images from sharing a prefix in its caches.
pub fn span_id(digest: u64) -> u32 {
    0x8000_0000 | (digest as u32 & 0x3FFF_FFFF)
}

/// The engine's prompt: each placeholder id of `ids` replaced by its image's span, and the images to send.
pub fn expand(ids: &[u32], images: Vec<Prepared>) -> Result<(Vec<u32>, Vec<crate::protocol::SubmitImage>), String> {
    let n = ids.iter().filter(|&&t| t == IMAGE_TOKEN).count();
    if n != images.len() {
        return Err(format!("the prompt has {n} image placeholders for {} images", images.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + images.iter().map(Prepared::span).sum::<usize>());
    let mut subs = Vec::with_capacity(images.len());
    let mut it = images.into_iter();
    for &t in ids {
        if t != IMAGE_TOKEN {
            out.push(t);
            continue;
        }
        let im = it.next().expect("counted");
        let pos = out.len() as u32;
        out.extend(std::iter::repeat(span_id(im.digest)).take(im.span()));
        subs.push(crate::protocol::SubmitImage {
            pos,
            n_vit_h: im.n_vit_h as u32,
            n_vit_w: im.n_vit_w as u32,
            n_llm_h: im.n_llm_h as u32,
            n_llm_w: im.n_llm_w as u32,
            pixels: im.pixels,
        });
    }
    Ok((out, subs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_match_the_reference() {
        // (w, h) -> (n_llm_h, n_llm_w, best_h, best_w), from image_processor.plan_image_grid
        assert_eq!(plan_image_grid(800, 600), (15, 20, 602, 812));
        assert_eq!(plan_image_grid(1188, 365), (9, 29, 378, 1190));
        assert_eq!(plan_image_grid(4000, 3000), (27, 36, 1134, 1512));
        assert_eq!(plan_image_grid(300, 1200), (29, 8, 1204, 308));
        assert_eq!(plan_image_grid(100, 80), (12, 15, 490, 616));
    }

    #[test]
    fn identity_resize_copies() {
        let px: Vec<u8> = (0..4 * 3 * 3).map(|i| i as u8).collect();
        assert_eq!(resize_bicubic(&px, 4, 3, 4, 3), px);
    }
}
