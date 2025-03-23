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
cargo bench --target  target-riscv64-no-vector.json -Zbuild-std
```

TODO:
* link to blog post.

