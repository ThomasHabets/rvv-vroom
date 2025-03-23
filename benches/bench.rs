#![feature(test)]

extern crate rvvwroom;
extern crate test;

//use vec::{mul_fvec, mul_fvec_rvv_m4, mul_fvec_rvv_m8};
use rvvwroom::*;

use test::Bencher;

#[bench]
fn bench_mul_cvec_rust(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| mul_cvec(&left.data, &right.data));
}

#[bench]
fn bench_mul_cvec_rust_v(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_v(&left.data, &right.data) });
}

#[bench]
fn bench_mul_fvec_rust(b: &mut Bencher) {
    let n = 1024;
    let left = vec![f32::default(); n];
    b.iter(|| mul_fvec(&left, &left));
}

#[bench]
fn bench_mul_fvec_rust_v(b: &mut Bencher) {
    let n = 1024;
    let left = vec![f32::default(); n];
    b.iter(|| unsafe { mul_fvec_v(&left, &left) });
}

#[bench]
fn bench_mul_fvec_asm_m8(b: &mut Bencher) {
    let (left, right) = gen_ftest();
    b.iter(|| unsafe { mul_fvec_rvv_m8(&left.data, &right.data) });
}

#[bench]
fn bench_mul_fvec_asm_m4(b: &mut Bencher) {
    let (left, right) = gen_ftest();
    b.iter(|| unsafe { mul_fvec_rvv_m4(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m4_stride(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_rvv_m4_stride(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m4_vl2(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_rvv_m4_vl2(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m8_stride(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_rvv_m8_stride(&left.data, &right.data) });
}
