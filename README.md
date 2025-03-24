## OrangePI RV2 findings

* stride load appears to be much slower than regular load.
* Why does vlseg2e32.v v0 not work in m8?

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
running 10 tests
test bench_mul_cvec_asm_m2_segment ... bench:       2,912.42 ns/iter (+/- 29.51)
test bench_mul_cvec_asm_m4_segment ... bench:       3,024.25 ns/iter (+/- 28.78)
test bench_mul_cvec_asm_m4_stride  ... bench:       4,671.07 ns/iter (+/- 49.04)
test bench_mul_cvec_asm_m8_stride  ... bench:       5,366.63 ns/iter (+/- 94.93)
test bench_mul_cvec_rust           ... bench:       9,932.82 ns/iter (+/- 41.45)
test bench_mul_cvec_rust_v         ... bench:       3,020.22 ns/iter (+/- 24.52)
test bench_mul_fvec_asm_m4         ... bench:         850.08 ns/iter (+/- 32.91)
test bench_mul_fvec_asm_m8         ... bench:         820.47 ns/iter (+/- 17.26)
test bench_mul_fvec_rust           ... bench:       4,095.23 ns/iter (+/- 15.66)
test bench_mul_fvec_rust_v         ... bench:       1,083.80 ns/iter (+/- 9.79)
```

TODO:
* link to blog post.
