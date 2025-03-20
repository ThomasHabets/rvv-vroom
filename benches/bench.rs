#![feature(test)]

extern crate test;

use test::Bencher;

type Complex = num_complex::Complex<f32>;

#[inline(never)]
fn mul_cvec(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

#[inline(never)]
fn mul_fvec(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right.iter()).map(|(x, y)| x * y).collect()
}

#[inline(never)]
fn mul_fvec_rvv_m4(left: &[f32], right: &[f32]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len() / 2;
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "add t2,t0,t0",
            "1:",
            "vle32.v v0, ({a_ptr})",
            "add {a_ptr}, {a_ptr}, t0",
            "vle32.v v8, ({b_ptr})",
            "add {b_ptr}, {b_ptr}, t0",
            "vle32.v v16, ({a_ptr})",
            "vfmul.vv v0, v0, v8",
            "vle32.v v24, ({b_ptr})",
            "vfmul.vv v16, v16, v24",
            "vse32.v v0, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t0",
            "vse32.v v16, ({o_ptr})",
            "sub {len}, {len}, t2",
            "add {a_ptr}, {a_ptr}, t0",
            "add {b_ptr}, {b_ptr}, t0",
            "add {o_ptr}, {o_ptr}, t0",
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[inline(never)]
fn mul_fvec_rvv_m8(left: &[f32], right: &[f32]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len() / 2;
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m8, ta, ma",
            "add t2,t0,t0",
            "1:",
            "vle32.v v0, ({a_ptr})",
            "add {a_ptr}, {a_ptr}, t0",
            "vle32.v v8, ({b_ptr})",
            "add {b_ptr}, {b_ptr}, t0",
            "vle32.v v16, ({a_ptr})",
            "vfmul.vv v0, v0, v8",
            "vle32.v v24, ({b_ptr})",
            "vfmul.vv v16, v16, v24",
            "vse32.v v0, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t0",
            "vse32.v v16, ({o_ptr})",
            "sub {len}, {len}, t2",
            "add {a_ptr}, {a_ptr}, t0",
            "add {b_ptr}, {b_ptr}, t0",
            "add {o_ptr}, {o_ptr}, t0",
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[inline(never)]
fn mul_cvec_rvv_m8_v2l(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len();
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m8, ta, ma",
            "add t2, t0, t0",
            "li t1, 8",
            "1:",
            // (ac - bd) + (ad + bc)i

            // Load everything.
            "vl2re32.v v0, ({a_ptr})",
            "vl2re32.v v16, ({b_ptr})",

            // Calculate real.
            "vfmul.vv v0, v0, v16",
            "vfnmacc.vv v0, v8,v24",
            "vsse32.v v0, ({o_ptr}), t1",
            "add {o_ptr}, {o_ptr}, t0",

            // We ran out of registers, so reload v0.
            // Turns out undoing the math is slower.
            "vlse32.v v0, ({a_ptr}), t1",
            //"vfmacc.vv v0, v16, v24",
            //"vfdiv.vv v0, v0, v8",

            // Calculate complex.
            "vfmul.vv v0, v0, v24",
            "vfmacc.vv v0, v8, v16",
            "vsse32.v v0, ({o_ptr}), t1",
            "add {o_ptr}, {o_ptr}, t0",

            // Update pointers / counters.
            "sub {len}, {len}, t2",
            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[inline(never)]
fn mul_cvec_rvv_m8_stride(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len();
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m8, ta, ma",
            "add t2, t0, t0",
            "li t1, 8",
            "1:",
            // (ac - bd) + (ad + bc)i

            // Load everything.
            "vlse32.v v0, ({a_ptr}), t1",
            "vlse32.v v8, ({b_ptr}), t1",
            "vlse32.v v16, ({a_ptr}), t1",
            "vlse32.v v24, ({b_ptr}), t1",

            // Calculate real.
            "vfmul.vv v0, v0, v8",
            "vfnmacc.vv v0, v16,v24",
            "vsse32.v v0, ({o_ptr}), t1",
            "add {o_ptr}, {o_ptr}, t0",

            // We ran out of registers, so reload v0.
            // Turns out undoing the math is slower.
            "vlse32.v v0, ({a_ptr}), t1",
            //"vfmacc.vv v0, v16, v24",
            //"vfdiv.vv v0, v0, v8",

            // Calculate complex.
            "vfmul.vv v0, v0, v24",
            "vfmacc.vv v0, v8, v16",
            "vsse32.v v0, ({o_ptr}), t1",
            "add {o_ptr}, {o_ptr}, t0",

            // Update pointers / counters.
            "sub {len}, {len}, t2",
            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[inline(never)]
fn mul_cvec_rvv_m4_vl2(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len();
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "add t2, t0, t0",
            "li t1, 8",  // Skip every other float.
            "1:",
            "sub {len}, {len}, t2",
            // (ac - bd) + (ad + bc)i

            // v0:  left.real
            // v4:  left.imag
            // v8:  right.real
            // v12: right.imag
            // v16: output.real
            // v20: output.imag
            //
            // Interleaved seems to help.
            "vl2re32.v v0, ({a_ptr})",
            "vl2re32.v v8, ({b_ptr})",
            "add {a_ptr}, {a_ptr}, t2",
            "add {b_ptr}, {b_ptr}, t2",
            "vfmul.vv v16, v0, v8",

            "vfmul.vv v20, v0, v12",
            "vfnmacc.vv v16, v4,v12",

            "vfmacc.vv v12, v8, v16",
            "vs2r.v v16, ({o_ptr})",
            "add {o_ptr}, {o_ptr}, t2",

            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[inline(never)]
fn mul_cvec_rvv_m4_stride(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
    use std::arch::asm;
    use std::mem::MaybeUninit;
    let len = left.len();
    let mut ret: Vec<MaybeUninit<Complex>> = Vec::with_capacity(len);
    let ret = unsafe {
        ret.set_len(left.len());
        std::mem::transmute::<Vec<MaybeUninit<Complex>>, Vec<Complex>>(ret)
    };
    unsafe {
        asm!(
            "vsetvli t0, {len}, e32, m4, ta, ma",
            "add t2, t0, t0",
            "li t1, 8",  // Skip every other float.
            "1:",
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
            "vlse32.v v8, ({b_ptr}), t1",
            "vlse32.v v16, ({a_ptr}), t1",
            "vfmul.vv v4, v0, v8",
            "vlse32.v v24, ({b_ptr}), t1",
            "add {a_ptr}, {a_ptr}, t2",

            "vfmul.vv v12, v0, v24",
            "add {b_ptr}, {b_ptr}, t2",
            "vfnmacc.vv v4, v16,v24",
            "vsse32.v v4, ({o_ptr}), t1",
            "add {o_ptr}, {o_ptr}, t0",

            "vfmacc.vv v12, v8, v16",
            "vsse32.v v12, ({o_ptr}), t1",
            "sub {len}, {len}, t2",
            "add {o_ptr}, {o_ptr}, t0",

            // Update pointers / counters.
            "bnez {len}, 1b",
            // "vfredosum.vs ft0, v0, ft0",
            len = inout(reg) len => _,
            a_ptr = inout(reg) left.as_ptr() => _,
            b_ptr = inout(reg) right.as_ptr() => _,
            o_ptr = inout(reg) ret.as_ptr() => _,
            //x = inout(reg)
        )
    }
    ret
}

#[bench]
fn bench_mul_cvec(b: &mut Bencher) {
    let n = 1024;
    let left = vec![Complex::default(); n];
    b.iter(|| mul_cvec(&left, &left));
}

#[bench]
fn bench_mul_fvec(b: &mut Bencher) {
    let n = 1024;
    let left = vec![f32::default(); n];
    b.iter(|| mul_fvec(&left, &left));
}

#[bench]
fn bench_mul_fvec_rvv_m8(b: &mut Bencher) {
    let n = 1024;
    let left = vec![f32::default(); n];
    b.iter(|| mul_fvec_rvv_m8(&left, &left));
}

#[bench]
fn bench_mul_fvec_rvv_m4(b: &mut Bencher) {
    let n = 1024;
    let left = vec![f32::default(); n];
    b.iter(|| mul_fvec_rvv_m4(&left, &left));
}

#[bench]
fn bench_mul_cvec_rvv_m4_stride(b: &mut Bencher) {
    let n = 1024;
    let left = vec![Complex::default(); n];
    b.iter(|| mul_cvec_rvv_m4_stride(&left, &left));
}

#[bench]
fn bench_mul_cvec_rvv_m4_vl2(b: &mut Bencher) {
    let n = 1024;
    let left = vec![Complex::default(); n];
    b.iter(|| mul_cvec_rvv_m4_vl2(&left, &left));
}

#[bench]
fn bench_mul_cvec_rvv_m8_stride(b: &mut Bencher) {
    let n = 1024;
    let left = vec![Complex::default(); n];
    b.iter(|| mul_cvec_rvv_m8_stride(&left, &left));
}

#[bench]
fn bench_mul_cvec_rvv_m8_v2l(b: &mut Bencher) {
    let n = 1024;
    let left = vec![Complex::default(); n];
    b.iter(|| mul_cvec_rvv_m8_v2l(&left, &left));
}
