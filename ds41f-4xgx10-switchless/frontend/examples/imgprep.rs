//! The front-end's image preparation on one file, for comparing with the reference (vit_ref.py's .u8):
//!     cargo run --release --example imgprep -- IMAGE OUT.u8
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("read");
    let p = dsfront::image::prepare(&bytes).expect("prepare");
    std::fs::write(&a[2], &p.pixels).expect("write");
    println!("GRID {} {} {} {} span {}", p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w, p.span());
}
