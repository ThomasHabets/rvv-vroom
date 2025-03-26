#![feature(riscv_target_feature)]

use std::arch::asm;

type Complex = num_complex::Complex<f32>;

// Make a buf with uninitialized contents.
//
// It'll be filled by assembly code.
unsafe fn mkbuf<T>(len: usize) -> Vec<T> {
    use std::mem::MaybeUninit;
    let mut ret: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
    unsafe {
        ret.set_len(len);
        std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(ret)
    }
}

/// Multiply two vectors, and sum the results.
#[inline(never)]
pub fn mul_sum_vec(left: &[Complex], right: &[Complex]) -> Complex {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).sum()
}

/// Multiply two vectors, and sum the results.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_sum_vec_v(left: &[Complex], right: &[Complex]) -> Complex {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).sum()
}

/// Multiply two complex vectors, summing up the results.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_sum_cvec_asm_m4(left: &[Complex], right: &[Complex]) -> Complex {
    unsafe {
        let mut real;
        let mut imag;
        asm!(
            // Zero out registers.
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "vfmv.v.f v24, {fzero}",
            "vfmv.v.f v28, {fzero}",

            "1:",

            "vsetvli t0, {len}, e32, m4, ta, ma",
            // (ac - bd) + (ad + bc)i

            // v0:  left.real    a
            // v4:  left.imag    b
            // v8:  right.real   c
            // v12: right.imag   d
            // v16: tmp.real
            // v20: tmp.imag
            // v24: acc.real
            // v28: acc.imag
            "slli t1,t0,3", // bytes per loop.

            // Load
            "vlseg2e32.v v0, ({a_ptr})",
            "add {a_ptr}, {a_ptr}, t1",
            "vlseg2e32.v v8, ({b_ptr})",
            "add {b_ptr}, {b_ptr}, t1",

            // ac
            "vfmul.vv v16, v0, v8",

            // ad
            "vfmul.vv v20, v0, v12",

            // ac - bd
            "vfnmsac.vv v16, v4, v12",

            // ad + bc
            "vfmacc.vv v20, v4, v8",

            "vsetvli zero,{len},e32,m8,ta,ma",
            "vfadd.vv v24, v24, v16",
            "sub {len}, {len}, t0",

            "bnez {len}, 1b",

            "li t0, 256",
            "vsetvli t0,t0,e32,m4,ta,ma",

            "vfmv.v.f v0, {fzero}",
            "vfredusum.vs v24, v24, v0",
            "vfredusum.vs v28, v28, v0",
            "vfmv.f.s {real}, v24",
            "vfmv.f.s {imag}, v28",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            real = inout(freg) 0.0f32 => real,
            imag = inout(freg) 0.0f32 => imag,
            fzero = inout(freg) 0.0f32 => _,
        );
        Complex::new(real, imag)
    }
}

/// Multiply two float vectors, putting the result in a third.
///
/// Pure Rust code, for comparison.
#[inline(never)]
pub fn mul_fvec(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

/// Multiply two float vectors, putting the result in a third.
///
/// Pure Rust code, but with `v` extension force-enabled.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_fvec_v(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

/// Multiply two float vectors, putting the result in a third.
///
/// Using vector instruction assembly and 4-register (M4) grouping.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_fvec_asm_m4(left: &[f32], right: &[f32]) -> Vec<f32> {
    unsafe {
        let ret: Vec<f32> = mkbuf(left.len());
        asm!(
        "1:",
        "vsetvli t0, {len}, e32, m4, ta, ma",
        // Loop unrolled 4x.
        // v0,v4,v8,v12 = left, and output.
        // v16,v20,v24,v28 = right.
        "slli t2,t0,2", // entries per loop.
        "slli t1,t0,2", // bytes per register.

        "sub {len}, {len}, t2",

        // Load
        "vle32.v v0, ({a_ptr})",
        "add {a_ptr}, {a_ptr}, t1",
        "vle32.v v16, ({b_ptr})",
        "add {b_ptr}, {b_ptr}, t1",

        "vle32.v v4, ({a_ptr})",
        "add {a_ptr}, {a_ptr}, t1",
        "vle32.v v20, ({b_ptr})",
        "add {b_ptr}, {b_ptr}, t1",

        "vle32.v v8, ({a_ptr})",
        "add {a_ptr}, {a_ptr}, t1",
        "vle32.v v24, ({b_ptr})",
        "add {b_ptr}, {b_ptr}, t1",

        "vle32.v v12, ({a_ptr})",
        "add {a_ptr}, {a_ptr}, t1",
        "vle32.v v28, ({b_ptr})",
        "add {b_ptr}, {b_ptr}, t1",

        // Multiply.
        "vfmul.vv v0, v0, v16",
        "vfmul.vv v4, v4, v20",
        "vfmul.vv v8, v8, v24",
        "vfmul.vv v12, v12, v28",

        // Store.
        "vse32.v v0, ({o_ptr})",
        "add {o_ptr}, {o_ptr}, t1",

        "vse32.v v4, ({o_ptr})",
        "add {o_ptr}, {o_ptr}, t1",

        "vse32.v v8, ({o_ptr})",
        "add {o_ptr}, {o_ptr}, t1",

        "vse32.v v12, ({o_ptr})",
        "add {o_ptr}, {o_ptr}, t1",

        "bnez {len}, 1b",
        len = inout(reg) left.len() => _,
        a_ptr = inout(reg) left.as_ptr() => _,
        b_ptr = inout(reg) right.as_ptr() => _,
        o_ptr = inout(reg) ret.as_ptr() => _,
        );
        ret
    }
}

/// Multiply two float vectors, putting the result in a third.
///
/// Using vector instruction assembly and 8-register (M8) grouping.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_fvec_asm_m8(left: &[f32], right: &[f32]) -> Vec<f32> {
    unsafe {
        let ret: Vec<f32> = mkbuf(left.len());
        asm!(
            "1:",
            "vsetvli t0, {len}, e32, m8, ta, ma",
            "slli t1,t0,2", // t1 = bytes per register.
            // Unrolled by 2.
            "slli t2,t0,1", // t2 = entries per loop.
                            //
            "sub {len}, {len}, t2",

            "vle32.v v0, ({a_ptr})",
            "add {a_ptr}, {a_ptr}, t1",
            "vle32.v v8, ({b_ptr})",
            "add {b_ptr}, {b_ptr}, t1",

            "vle32.v v16, ({a_ptr})",
            "add {a_ptr}, {a_ptr}, t1",

            "vfmul.vv v0, v0, v8",

            "vle32.v v24, ({b_ptr})",
            "add {b_ptr}, {b_ptr}, t1",

            "vfmul.vv v16, v16, v24",

            "vse32.v v0, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t1",

            "vse32.v v16, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t1",
            "bnez {len}, 1b",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
        );
        ret
    }
}

/// Multiply two complex vectors, putting the result in a third.
///
/// In pure Rust, for comparison.
#[inline(never)]
pub fn mul_cvec(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

/// Multiply two complex vectors, putting the result in a third.
///
/// In pure Rust, but with vector instructions forced on.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_cvec_v(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

/// Multiply two complex vectors, putting the result in a third.
///
/// Using vector instruction assembly with 8-register (M8) grouping, reading
/// strided.
///
/// This version does an extra memory load (hopefully from cache), due to
/// running out of registers.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_cvec_asm_m8_stride(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    unsafe {
        let ret: Vec<Complex> = mkbuf(left.len());
        asm!(
            "li t1, 8",
            "1:",
            "vsetvli t0, {len}, e32, m8, ta, ma",
            "slli t2, t0, 3", // t2 = byte size of loop.

            // (ac - bd) + (ad + bc)i
            "sub {len}, {len}, t0",

            // Load everything.
            "vlse32.v v0, ({a_ptr}), t1",
            "addi {a_ptr}, {a_ptr}, 4",
            "vlse32.v v8, ({b_ptr}), t1",
            "addi {b_ptr}, {b_ptr}, 4",
            "vlse32.v v16, ({a_ptr}), t1",
            "addi {a_ptr}, {a_ptr}, -4",
            "vlse32.v v24, ({b_ptr}), t1",
            "addi {b_ptr}, {b_ptr}, -4",

            // Calculate real.
            "vfmul.vv v0, v0, v8",
            "vfnmsac.vv v0, v16, v24",
            "vsse32.v v0, ({o_ptr}), t1",
            "addi {o_ptr}, {o_ptr}, 4",

            // We ran out of registers, so reload v0.
            // Turns out undoing the math is slower.
            "vlse32.v v0, ({a_ptr}), t1",
            //"vfmacc.vv v0, v16, v24",
            //"vfdiv.vv v0, v0, v8",

            // Calculate complex.
            "vfmul.vv v0, v0, v24",
            "vfmacc.vv v0, v8, v16",
            "vsse32.v v0, ({o_ptr}), t1",
            "addi {o_ptr}, {o_ptr}, -4",
            "add {o_ptr}, {o_ptr}, t2",

            // Update pointers / counters.
            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        );
        ret
    }
}

/// Multiply two complex vectors, putting the result in a third.
///
/// Using vector instruction assembly with 2-register (M2) grouping, reading
/// segmented.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_cvec_asm_m2_segment(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    unsafe {
        let ret: Vec<Complex> = mkbuf(left.len());
        asm!(
            "1:",
            "vsetvli t0, {len}, e32, m2, ta, ma",
            "slli t2, t0, 3", // t2 = byte size of loop.
            "sub {len}, {len}, t0",
            // (ac - bd) + (ad + bc)i

            // v0:  left.real    a
            // v4:  left.imag    b
            // v8:  right.real   c
            // v12: right.imag   d
            // v16: output.real
            // v20: output.imag
            //
            // Interleaved seems to help.
            "vlseg2e32.v v0, ({a_ptr})",
            "vlseg2e32.v v8, ({b_ptr})",

            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",

            // ac
            "vfmul.vv v16, v0, v8",

            // ad
            "vfmul.vv v18, v0, v10",

            // ac - bd
            "vfnmsac.vv v16, v2, v10",

            // ad + bc
            "vfmacc.vv v18, v2, v8",

            //"vs2r.v v0, ({o_ptr})",
            "vsseg2e32.v v16, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t2",

            "bnez {len}, 1b",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
        );
        ret
    }
}

/// Multiply two complex vectors, putting the result in a third.
///
/// Using vector instruction assembly with 4-register (M4) grouping, reading
/// segmented.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_cvec_asm_m4_segment(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    unsafe {
        let ret: Vec<Complex> = mkbuf(left.len());
        asm!(
            "1:",
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "slli t2, t0, 3", // t2 = byte size of loop.
            "sub {len}, {len}, t0",
            // (ac - bd) + (ad + bc)i

            // v0:  left.real    a
            // v4:  left.imag    b
            // v8:  right.real   c
            // v12: right.imag   d
            // v16: output.real
            // v20: output.imag
            //
            // Interleaved seems to help.
            "vlseg2e32.v v0, ({a_ptr})",
            "vlseg2e32.v v8, ({b_ptr})",

            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",

            // ac
            "vfmul.vv v16, v0, v8",

            // ad
            "vfmul.vv v20, v0, v12",

            // ac - bd
            "vfnmsac.vv v16, v4, v12",

            // ad + bc
            "vfmacc.vv v20, v4, v8",

            "vsseg2e32.v v16, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t2",

            "bnez {len}, 1b",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
        );
        ret
    }
}

/// Multiply two complex vectors, putting the result in a third.
///
/// Using vector instruction assembly with 4-register (M4) grouping, reading
/// strided.
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
#[inline(never)]
pub fn mul_cvec_asm_m4_stride(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    unsafe {
        let ret: Vec<Complex> = mkbuf(left.len());
        asm!(
            "li t1, 8",  // Skip every other float.
            "1:",
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "slli t2, t0, 3", // byte size of loop.
            "sub {len}, {len}, t0",
            // t4 is o_ptr plus im offset.
            "addi t4, {o_ptr}, 4",
            // (ac - bd) + (ad + bc)i

            // v0:  left.real
            // v8:  left.imag
            // v16: right.real
            // v24: right. imag
            // v4:  output.real
            // v12: output.imag
            //
            // Interleaved seems to help.
            "vlse32.v v0, ({a_ptr}), t1",
            "addi {a_ptr}, {a_ptr}, 4",
            "vlse32.v v8, ({b_ptr}), t1",
            "addi {b_ptr}, {b_ptr}, 4",

            "vlse32.v v16, ({a_ptr}), t1",
            "addi {a_ptr}, {a_ptr}, -4",

            "vfmul.vv v4, v0, v8",
            "vlse32.v v24, ({b_ptr}), t1",
            "addi {b_ptr}, {b_ptr}, -4",
            "add {a_ptr}, {a_ptr}, t2",

            "vfmul.vv v12, v0, v24",
            "add {b_ptr}, {b_ptr}, t2",
            "vfnmsac.vv v4, v16,v24",
            "vsse32.v v4, ({o_ptr}), t1",

            "vfmacc.vv v12, v8, v16",
            "vsse32.v v12, (t4), t1",

            "add {o_ptr}, {o_ptr}, t2",

            // Update pointers / counters.
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) left.len() => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        );
        ret
    }
}

// Wrapper to help create very aligned buffers.
#[repr(align(1024))]
pub struct BF32 {
    pub data: [f32; 1024],
}
#[repr(align(1024))]
pub struct BC32 {
    pub data: [Complex; 1024],
}

/// Generate two aligned f32 buffers.
pub fn gen_ftest() -> (BF32, BF32) {
    let n = 1024;
    (
        BF32 {
            data: (0..n)
                .map(|v| v as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
        BF32 {
            data: (0..n)
                .map(|v| (n - v) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
    )
}

/// Generate two aligned complex buffers.
pub fn gen_ctest() -> (BC32, BC32) {
    let n = 1024;
    (
        BC32 {
            data: (0..n)
                .map(|v| Complex::new(v as f32, -(v as f32) / 1.5))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
        BC32 {
            data: (0..n)
                .map(|v| Complex::new((n - v) as f32, -((n - v) as f32) / 1.5))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_diff_f(got: &[f32], want: &[f32], name: &str) {
        got.iter()
            .zip(want.iter())
            .enumerate()
            .for_each(|(n, (&a, &b))| {
                if (a - b).abs() > 0.1 {
                    assert_eq!(a, b, "{name}: diff at pos {n}");
                }
            });
    }
    fn find_diff_c(got: &[Complex], want: &[Complex], name: &str) {
        got.iter()
            .zip(want.iter())
            .enumerate()
            .for_each(|(n, (&a, &b))| {
                if (a - b).norm().abs() > 0.1 {
                    assert_eq!(a, b, "{name}: diff at pos {n}");
                }
            });
    }

    #[test]
    fn test_mul_fvec() {
        let (left, right) = gen_ftest();
        let want = mul_fvec(&left.data, &right.data);
        let t: &[(&str, unsafe fn(&[f32], &[f32]) -> Vec<f32>)] = &[
            ("mul_fvec_asm_m4", mul_fvec_asm_m4),
            ("mul_fvec_asm_m8", mul_fvec_asm_m8),
        ];
        for (name, f) in t {
            let got = unsafe { f(&left.data, &right.data) };
            find_diff_f(&got, &want, name);
            assert_eq!(got, want, "failed {name}");
        }
    }
    #[test]
    fn test_mul_cvec() {
        let (left, right) = gen_ctest();
        let want = mul_cvec(&left.data, &right.data);
        let t: &[(&str, unsafe fn(&[Complex], &[Complex]) -> Vec<Complex>)] = &[
            ("mul_cvec_asm_m2_segment", mul_cvec_asm_m2_segment),
            ("mul_cvec_asm_m4_segment", mul_cvec_asm_m4_segment),
            ("mul_cvec_asm_m4_stride", mul_cvec_asm_m4_stride),
            ("mul_cvec_asm_m8_stride", mul_cvec_asm_m8_stride),
        ];
        for (name, f) in t {
            let got = unsafe { f(&left.data, &right.data) };
            (0..16).for_each(|n| {
                println!(
                    "{n}: {} {} => {}   Got: {}",
                    left.data[n],
                    right.data[n],
                    left.data[n] * right.data[n],
                    got[n],
                );
            });
            find_diff_c(&got, &want, name);
        }
    }
    #[test]
    fn test_mul_sum_cvec() {
        let (left, right) = gen_ctest();
        let want = mul_sum_vec(&left.data, &right.data);
        let t: &[(&str, unsafe fn(&[Complex], &[Complex]) -> Complex)] =
            &[("mul_sum_cvec_asm_m4", mul_sum_cvec_asm_m4)];
        for (name, f) in t {
            let got = unsafe { f(&left.data, &right.data) };
            let diff = got - want;
            assert!(
                (diff.re / got.re).abs() < 0.01,
                "{name}: Real out of range: {got} {want}"
            );
            assert!(
                (diff.im / got.im).abs() < 0.01,
                "{name}: Imag out of range: {got} {want}"
            );
        }
    }
}
