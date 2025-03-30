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

fn main() {
    #[cfg(target_arch = "riscv64")]
    {
        println!("RVV compiled={} detected={}", cfg!(target_feature="v"), std::arch::is_riscv_feature_detected!("v"));
        let n = 1024;
        let left = vec![Complex::default(); n];
        unsafe { mul_vec_rvv(&left, &left) };
    }
}
