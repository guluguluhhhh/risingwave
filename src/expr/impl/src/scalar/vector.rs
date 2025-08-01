// Copyright 2025 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use risingwave_common::array::Finite32;
use risingwave_common::types::{F64, VectorRef, VectorVal};
use risingwave_common::util::iter_util::ZipEqFast;
use risingwave_expr::{ExprError, Result, function};

fn check_dims(name: &'static str, lhs: &[f32], rhs: &[f32]) -> Result<()> {
    if lhs.len() != rhs.len() {
        return Err(ExprError::InvalidParam {
            name,
            reason: format!(
                "different vector dimensions {} and {}",
                lhs.len(),
                rhs.len()
            )
            .into(),
        });
    }
    Ok(())
}

/// ```slt
/// query R
/// SELECT l2_distance('[0,0]'::vector(2), '[3,4]');
/// ----
/// 5
///
/// query R
/// SELECT l2_distance('[0,0]'::vector(2), '[0,1]');
/// ----
/// 1
///
/// query error dimensions
/// SELECT l2_distance('[1,2]'::vector(2), '[3]');
///
/// query R
/// SELECT l2_distance('[3e38]'::vector(1), '[-3e38]');
/// ----
/// Infinity
///
/// query R
/// SELECT l2_distance('[1,1,1,1,1,1,1,1,1]'::vector(9), '[1,1,1,1,1,1,1,4,5]');
/// ----
/// 5
///
/// query R
/// SELECT '[0,0]'::vector(2) <-> '[3,4]';
/// ----
/// 5
/// ```
#[function("l2_distance(vector, vector) -> float8"/*, type_infer = "unreachable"*/)]
fn l2_distance(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<F64> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("l2_distance", lhs, rhs)?;

    let mut sum = 0.0f32;
    for (l, r) in lhs.iter().zip_eq_fast(rhs.iter()) {
        let diff = l - r;
        sum += diff * diff;
    }
    Ok((sum as f64).sqrt().into())
}

/// ```slt
/// query R
/// SELECT cosine_distance('[1,2]'::vector(2), '[2,4]');
/// ----
/// 0
///
/// query R
/// SELECT cosine_distance('[1,2]'::vector(2), '[0,0]');
/// ----
/// NaN
///
/// query R
/// SELECT cosine_distance('[1,1]'::vector(2), '[1,1]');
/// ----
/// 0
///
/// query R
/// SELECT cosine_distance('[1,0]'::vector(2), '[0,2]');
/// ----
/// 1
///
/// query R
/// SELECT cosine_distance('[1,1]'::vector(2), '[-1,-1]');
/// ----
/// 2
///
/// query error dimensions
/// SELECT cosine_distance('[1,2]'::vector(2), '[3]');
///
/// query R
/// SELECT cosine_distance('[1,1]'::vector(2), '[1.1,1.1]');
/// ----
/// 0
///
/// query R
/// SELECT cosine_distance('[1,1]'::vector(2), '[-1.1,-1.1]');
/// ----
/// 2
///
/// query R
/// SELECT cosine_distance('[3e38]'::vector(1), '[3e38]');
/// ----
/// NaN
///
/// query R
/// SELECT cosine_distance('[1,2,3,4,5,6,7,8,9]'::vector(9), '[1,2,3,4,5,6,7,8,9]');
/// ----
/// 0
///
/// query R
/// SELECT cosine_distance('[1,2,3,4,5,6,7,8,9]'::vector(9), '[-1,-2,-3,-4,-5,-6,-7,-8,-9]');
/// ----
/// 2
///
/// query R
/// SELECT '[1,2]'::vector(2) <=> '[2,4]';
/// ----
/// 0
/// ```
#[function("cosine_distance(vector, vector) -> float8")]
fn cosine_distance(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<F64> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("cosine_distance", lhs, rhs)?;

    let mut dot_product = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let mut rhs_norm = 0.0f32;
    for (l, r) in lhs.iter().zip_eq_fast(rhs.iter()) {
        dot_product += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }
    let similarity = dot_product as f64 / (lhs_norm as f64 * rhs_norm as f64).sqrt();

    Ok((1.0 - similarity.clamp(-1.0, 1.0)).into())
}

/// ```slt
/// query R
/// SELECT l1_distance('[0,0]'::vector(2), '[3,4]');
/// ----
/// 7
///
/// query R
/// SELECT l1_distance('[0,0]'::vector(2), '[0,1]');
/// ----
/// 1
///
/// query error dimensions
/// SELECT l1_distance('[1,2]'::vector(2), '[3]');
///
/// query R
/// SELECT l1_distance('[3e38]'::vector(1), '[-3e38]');
/// ----
/// Infinity
///
/// query R
/// SELECT l1_distance('[1,2,3,4,5,6,7,8,9]'::vector(9), '[1,2,3,4,5,6,7,8,9]');
/// ----
/// 0
///
/// query R
/// SELECT l1_distance('[1,2,3,4,5,6,7,8,9]'::vector(9), '[0,3,2,5,4,7,6,9,8]');
/// ----
/// 9
///
/// query R
/// SELECT '[0,0]'::vector(2) <+> '[3,4]';
/// ----
/// 7
/// ```
#[function("l1_distance(vector, vector) -> float8")]
fn l1_distance(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<F64> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("l1_distance", lhs, rhs)?;

    let mut sum = 0.0f32;
    for (l, r) in lhs.iter().zip_eq_fast(rhs.iter()) {
        sum += (l - r).abs();
    }
    Ok((sum as f64).into())
}

/// ```slt
/// query R
/// SELECT inner_product('[1,2]'::vector(2), '[3,4]');
/// ----
/// 11
///
/// query error dimensions
/// SELECT inner_product('[1,2]'::vector(2), '[3]');
///
/// query R
/// SELECT inner_product('[3e38]'::vector(1), '[3e38]');
/// ----
/// Infinity
///
/// query R
/// SELECT inner_product('[1,1,1,1,1,1,1,1,1]'::vector(9), '[1,2,3,4,5,6,7,8,9]');
/// ----
/// 45
///
/// query R
/// SELECT '[1,2]'::vector(2) <#> '[3,4]';
/// ----
/// -11
/// ```
#[function("inner_product(vector, vector) -> float8")]
fn inner_product(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<F64> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("inner_product", lhs, rhs)?;

    let mut sum = 0.0f32;
    for (l, r) in lhs.iter().zip_eq_fast(rhs.iter()) {
        sum += l * r;
    }
    Ok((sum as f64).into())
}

/// ```slt
/// query R
/// SELECT '[1,2,3]'::vector(3) + '[4,5,6]';
/// ----
/// [5,7,9]
///
/// query error out of range: overflow
/// SELECT '[3e38]'::vector(1) + '[3e38]';
///
/// query error dimensions
/// SELECT '[1,2]'::vector(2) + '[3]';
/// ```
#[function("add(vector, vector) -> vector", type_infer = "unreachable")]
fn vector_add(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<VectorVal> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("vector_add", lhs, rhs)?;

    let result = lhs
        .iter()
        .zip_eq_fast(rhs.iter())
        .map(|(l, r)| Finite32::try_from(l + r))
        .try_collect()
        .map_err(|_| ExprError::NumericOverflow)?;
    Ok(result)
}

/// ```slt
/// query R
/// SELECT '[1,2,3]'::vector(3) - '[4,5,6]';
/// ----
/// [-3,-3,-3]
///
/// query error out of range: overflow
/// SELECT '[-3e38]'::vector(1) - '[3e38]';
///
/// query error dimensions
/// SELECT '[1,2]'::vector(2) - '[3]';
/// ```
#[function("subtract(vector, vector) -> vector", type_infer = "unreachable")]
fn vector_subtract(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<VectorVal> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("vector_subtract", lhs, rhs)?;

    let result = lhs
        .iter()
        .zip_eq_fast(rhs.iter())
        .map(|(l, r)| Finite32::try_from(l - r))
        .try_collect()
        .map_err(|_| ExprError::NumericOverflow)?;
    Ok(result)
}

/// ```slt
/// query R
/// SELECT '[1,2,3]'::vector(3) * '[4,5,6]';
/// ----
/// [4,10,18]
///
/// query error out of range: overflow
/// SELECT '[1e37]'::vector(1) * '[1e37]';
///
/// query error out of range: underflow
/// SELECT '[1e-37]'::vector(1) * '[1e-37]';
///
/// query error dimensions
/// SELECT '[1,2]'::vector(2) * '[3]';
/// ```
#[function("multiply(vector, vector) -> vector", type_infer = "unreachable")]
fn vector_multiply(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<VectorVal> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();
    check_dims("vector_multiply", lhs, rhs)?;

    let result = lhs
        .iter()
        .zip_eq_fast(rhs.iter())
        .map(|(l, r)| {
            let v = l * r;
            match v == 0. && !(*l == 0. || *r == 0.) {
                true => Err(ExprError::NumericUnderflow),
                false => Finite32::try_from(v).map_err(|_| ExprError::NumericOverflow),
            }
        })
        .try_collect()?;
    Ok(result)
}

/// ```slt
/// query R
/// SELECT '[1,2,3]'::vector(3) || '[4,5]';
/// ----
/// [1,2,3,4,5]
///
/// query error cast
/// SELECT '[1,2,3]'::vector(3) || null;
///
/// query R
/// SELECT '[1,2,3]'::vector(3) || null::vector(4);
/// ----
/// NULL
///
/// query error vector cannot have more than 16000 dimensions
/// SELECT null::vector(16000) || '[1]';
/// ```
#[function("vec_concat(vector, vector) -> vector", type_infer = "unreachable")]
fn vector_concat(lhs: VectorRef<'_>, rhs: VectorRef<'_>) -> Result<VectorVal> {
    let lhs = lhs.into_slice();
    let rhs = rhs.into_slice();

    let result = lhs
        .iter()
        .chain(rhs)
        .copied()
        .map(Finite32::try_from)
        .try_collect()
        .map_err(|_| ExprError::NumericOverflow)?;
    Ok(result)
}
