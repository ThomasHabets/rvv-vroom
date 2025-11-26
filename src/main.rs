#![feature(riscv_target_feature)]
#![feature(stdarch_riscv_feature_detection)]
type Complex = num_complex::Complex<f32>;
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
fn mul_vec_rvv(left: &[Complex], right: &[Complex]) -> Vec<Complex> {
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
            "1:",
            "vle32.v v0, ({a_ptr})",
            "vle32.v v8, ({b_ptr})",
            "vfmul.vv v0, v0, v8",
            "vse32.v v0, ({o_ptr})",
            "sub {len}, {len}, t0",
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

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
fn check_rvv() {
    let enabled = cfg!(target_feature = "v");
    let detected = std::arch::is_riscv_feature_detected!("v");
    println!("RVV enabled={enabled} detected={detected}");
    if !detected {
        eprintln!("Warning! Vector extension not detected");
    }
}

fn main() {
    if false {
        #[cfg(target_arch = "riscv64")]
        {
            unsafe { check_rvv() };
            let n = 1024;
            let left = vec![Complex::default(); n];
            unsafe { mul_vec_rvv(&left, &left) };
        }
    }
    let mut inp = Vec::new();
    let mut correct = Vec::new();
    for t in -100_000..100_000 {
        let t = t as f32 / 10_000.0;
        //println!("{}", t as f32 / 10_000.0);
        inp.push(t);
        correct.push(t.atan());
    }
    let mut my_6 = vec![0.0f32; inp.len()];
    rvv_vroom::my_atan_6_m4(&mut my_6, &inp);
    let mut my_7_m2 = vec![0.0f32; inp.len()];
    rvv_vroom::my_atan_7_m2(&mut my_7_m2, &inp);
    let mut my_7_m4 = vec![0.0f32; inp.len()];
    rvv_vroom::my_atan_7_m4(&mut my_7_m4, &inp);
    let mut my_7_full_m8 = vec![0.0f32; inp.len()];
    rvv_vroom::my_atan_7_full_m8(&mut my_7_full_m8, &inp);
    let mut vol = vec![0.0f32; inp.len()];
    volk::volk_32f_atan_32f(&mut vol, &inp);
    for (n, v) in inp.iter().enumerate() {
        println!(
            "{v} {} {} {} {} {} {}",
            correct[n], my_6[n], my_7_m2[n], my_7_m4[n], my_7_full_m8[n], vol[n]
        );
    }
}
