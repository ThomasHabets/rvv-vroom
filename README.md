# rvv-vroom

The beginnings of a crate for manually tuned vector instruction based RISC-V
inner loops.

Currently requires nightly rust, because of the benchmark library, and the
manual messing around with target features.

Some more words about vector instructions are in [this blog
post](https://blog.habets.se/2025/03/Exploring-RISC-V-vector-instructions.html).

## How to enable vector instructions

It seems that LLVM doesn't know about the Ky X1 CPU, so you'll need to manually
enable it.

```
$ cat ~/.cargo/config.toml
[target.riscv64gc-unknown-linux-gnu]
rustflags = ["-Ctarget-cpu=native", "-Ctarget-feature=+v"]
```

## How to turn *off* vectoring by default

Say your default config enables the vector instructions, either because it's auto
detected or because you enabled it per above, but you still want to compare
vector and non-vector for pure Rust code.

```
$ cargo +nightly bench --target  target-riscv64-no-vector.json -Zbuild-std
[…]
running 13 tests
test bench_mul_cvec_asm_m2_segment ... bench:       2,922.91 ns/iter (+/- 30.98)
test bench_mul_cvec_asm_m4_segment ... bench:       2,969.60 ns/iter (+/- 442.15)
test bench_mul_cvec_asm_m4_stride  ... bench:       4,767.27 ns/iter (+/- 42.19)
test bench_mul_cvec_asm_m8_stride  ... bench:       5,436.98 ns/iter (+/- 48.10)
test bench_mul_cvec_rust           ... bench:      10,006.30 ns/iter (+/- 41.26)
test bench_mul_cvec_rust_v         ... bench:       3,020.90 ns/iter (+/- 28.26)
test bench_mul_fvec_asm_m4         ... bench:         843.10 ns/iter (+/- 51.86)
test bench_mul_fvec_asm_m8         ... bench:         799.99 ns/iter (+/- 24.97)
test bench_mul_fvec_rust           ... bench:       4,090.87 ns/iter (+/- 19.58)
test bench_mul_fvec_rust_v         ... bench:       1,081.94 ns/iter (+/- 15.05)
test bench_mul_sum_cvec_asm_m4     ... bench:       2,274.16 ns/iter (+/- 14.16)
test bench_mul_sum_cvec_rust       ... bench:       8,993.45 ns/iter (+/- 28.03)
test bench_mul_sum_cvec_rust_v     ... bench:       4,431.18 ns/iter (+/- 192.85)
```

Ok, there are easier ways of doing that, like:

```
$ RUSTFLAGS="-Ctarget-cpu=native -Ctarget-feature=-v" cargo +nightly bench
```
