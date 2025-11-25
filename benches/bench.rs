#![feature(test)]

extern crate rvv_vroom;
extern crate test;

//use vec::{mul_fvec, mul_fvec_asm_m4, mul_fvec_rvv_m8};
use rvv_vroom::*;

use test::Bencher;

#[bench]
fn bench_mul_sum_cvec_rust(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| mul_sum_vec(&left.data, &right.data));
}

#[bench]
fn bench_mul_sum_cvec_rust_v(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_sum_vec_v(&left.data, &right.data) });
}

#[bench]
fn bench_mul_sum_cvec_asm_m4(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_sum_cvec_asm_m4(&left.data, &right.data) });
}

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
    b.iter(|| unsafe { mul_fvec_asm_m8(&left.data, &right.data) });
}

#[bench]
fn bench_atan_fvec_asm_6_m4(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_6_m4(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_m4(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_m4(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_full_m4(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_full_m4(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_m2(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_m2(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_full_m2(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_full_m2(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_m8(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_m8(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_asm_7_full_m8(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| rvv_vroom::my_atan_7_full_m8(&mut left.data, &right.data));
}

#[bench]
fn bench_atan_fvec_volk(b: &mut Bencher) {
    let (mut left, right) = gen_ftest();
    b.iter(|| volk::volk_32f_atan_32f(&mut left.data, &right.data));
}

#[bench]
fn bench_mul_fvec_asm_m4(b: &mut Bencher) {
    let (left, right) = gen_ftest();
    b.iter(|| unsafe { mul_fvec_asm_m4(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m4_stride(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_asm_m4_stride(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m8_stride(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_asm_m8_stride(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m2_segment(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_asm_m2_segment(&left.data, &right.data) });
}

#[bench]
fn bench_mul_cvec_asm_m4_segment(b: &mut Bencher) {
    let (left, right) = gen_ctest();
    b.iter(|| unsafe { mul_cvec_asm_m4_segment(&left.data, &right.data) });
}
