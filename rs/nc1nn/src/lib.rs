#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::exceptions::Exception;

#[macro_use(s)]
extern crate ndarray;
use ndarray::{ Axis, Array, Array1, Array2, ArrayView
             , ArrayView1, ArrayViewMut1, ArrayView2
             , Dimension, RemoveAxis, Zip, stack };
use ndarray_parallel::prelude::*;

use numpy::{PyArray1, PyArray2};
use numpy::convert::{IntoPyArray};

use std::f64::INFINITY as inf;
use std::sync::{Arc, Mutex};
use std::fmt::Debug;
use std::marker::Copy;

static MSG_STACK:  &str = "Error stacking";
static MSG_SHAPE:  &str = "Error with shape";
static MSG_UPDATE: &str = "Error nn operation failed";

// types {{{
type Matrix<'a>     = ArrayView2<'a, f64>;
type PyMatrix       = PyArray2<f64>;
type MatrixOwned    = Array2<f64>;

type Vector<'a>     = ArrayView1<'a, f64>;
type PyVector       = PyArray1<f64>;
type VectorOwned    = Array1<f64>;

type Labels<'a>     = ArrayView1<'a, i64>;
type PyLabels       = PyArray1<i64>;
type LabelsOwned    = Array1<i64>;

type SharedVar<T>   = Arc<Mutex<T>>;

type PyUpdateResult = PyResult<
    (Py<PyMatrix>,Py<PyLabels>,Py<PyMatrix>,Py<PyVector>)
  >;

type NNResult       = PyResult<(f64, f64, MatrixOwned)>;
// }}}

// traits {{{

// Score {{{
trait Score { fn score(&self) -> f64; }

impl<'a> Score for ArrayViewMut1<'a, f64> {
  fn score(&self) -> f64 { (self[0], self[1]).score() }
}

impl Score for (f64, f64) {
  fn score(&self) -> f64 {
         if self.0 == self.1 { 0. }
    else if self.1 >  0.     { self.0 / self.1 }
    else                     { inf }
  }
}
// }}}

// DMap {{{
trait DMap {
  fn update(&mut self, eq: f64, neq: f64);
}

impl<'a> DMap for ArrayViewMut1<'a, f64> {
  fn update(&mut self, eq: f64, neq: f64) {
    self[0] = eq; self[1] = neq;
  }
}
// }}}

// SharedOps {{{
trait SharedOps<T> {
  fn init(v: T) -> Self;
  fn unpack(self) -> T;
}

impl<T: Debug> SharedOps<T> for SharedVar<T> {
  fn init(v: T) -> Self { Arc::new(Mutex::new(v)) }
  fn unpack(self) -> T {
    Arc::try_unwrap(self).unwrap().into_inner().unwrap()
  }
}
// }}}

// }}}

// macros {{{

// as_array! {{{
macro_rules! as_array {
  ( $( $x:expr ),* ) => {( $( $x.as_array() ),* )};
}
// }}}

// into_pyarray! {{{
macro_rules! into_pyarray {
  ( $py: expr, $x:expr $(,)* ) => {
    $x.into_pyarray($py).to_owned()
  };
  ( $py: expr, $( $x: expr ),* ) => {
    ( $( $x.into_pyarray($py).to_owned() ),* )
  };
}
// }}}

// }}}

#[pymodule]
fn nc1nn(_py: Python, m: &PyModule) -> PyResult<()> {

  // py_update_seq {{{
  #[pyfn(m, "update_seq")]
  fn py_update_seq( py     : Python
                  , X_new  : &PyMatrix
                  , y_new  : &PyLabels
                  , X_seen : &PyMatrix
                  , y_seen : &PyLabels
                  , dists  : &PyMatrix
                  , scores : &PyVector ) -> PyUpdateResult
  {
    let (X_new,y_new,X_seen,y_seen,dists,scores) =
      as_array!(X_new,y_new,X_seen,y_seen,dists,scores);

    let (len_seen, len_new) = (len(&X_seen)?,len(&X_new)?);

    let X = stack0(&[X_seen, X_new])?;
    let y = stack0(&[y_seen, y_new])?;

    let dists_new = Array::from_elem( (len_new, 2), inf );
    let mut dists  = stack0(&[dists, dists_new.view()])?;

    let scores_new = Array::zeros((len_new, ));
    let mut scores = stack0(&[scores, scores_new.view()])?;

    update_seq(&X, &y, &mut dists, &mut scores, len_seen)?;

    Ok(into_pyarray!(py, X, y, dists, scores))
  }
  // }}}

  // py_update_par {{{
  #[pyfn(m, "update_par")]
  fn py_update_par( py     : Python
                  , X_new  : &PyMatrix
                  , y_new  : &PyLabels
                  , X_seen : &PyMatrix
                  , y_seen : &PyLabels
                  , dists  : &PyMatrix
                  , scores : &PyVector ) -> PyUpdateResult
  {
    let (X_new,y_new,X_seen,y_seen,dists,scores) =
      as_array!(X_new,y_new,X_seen,y_seen,dists,scores);

    let (len_seen, len_new) = (len(&X_seen)?,len(&X_new)?);

    let X = stack0(&[X_seen, X_new])?;
    let y = stack0(&[y_seen, y_new])?;

    let dists_new = Array::from_elem( (len_new, 2), inf );
    let mut dists  = stack0(&[dists, dists_new.view()])?;

    let scores_new = Array::zeros((len_new, ));
    let mut scores = stack0(&[scores, scores_new.view()])?;

    py.allow_threads(||
      update_par(&X, &y, &mut dists, &mut scores, len_seen)
    )?;

    Ok(into_pyarray!(py, X, y, dists, scores))
  }
  // }}}

  // py_scores_seq {{{
  #[pyfn(m, "scores_seq")]
  fn py_scores_seq( py     : Python
                  , x_     : &PyVector
                  , y_     : i64
                  , X      : &PyMatrix
                  , y      : &PyLabels
                  , dists  : &PyMatrix
                  , scores : &PyVector )
    -> PyResult<Py<PyVector>>
  {
    let (x_, X, y, dists, scores) =
      as_array!( x_, X, y, dists, scores );

    let scores = scores_seq(x_, y_, X, y, dists, scores)?;

    Ok(into_pyarray!(py, scores))
  }
  // }}}

  // py_scores_par {{{
  #[pyfn(m, "scores_par")]
  fn py_scores_par( py     : Python
                  , x_     : &PyVector
                  , y_     : i64
                  , X      : &PyMatrix
                  , y      : &PyLabels
                  , dists  : &PyMatrix
                  , scores : &PyVector )
    -> PyResult<Py<PyVector>>
  {
    let (x_, X, y, dists, scores) =
      as_array!( x_, X, y, dists, scores );

    let scores = py.allow_threads(||
      scores_par(x_, y_, X, y, dists, scores)
    )?;

    Ok(into_pyarray!(py, scores))
  }
  // }}}

  Ok(())
}

// update_seq {{{
fn update_seq( X      : &MatrixOwned
             , y      : &LabelsOwned
             , dists  : &mut MatrixOwned
             , scores : &mut VectorOwned
             , seen   : usize            ) -> PyResult<()>
{
  let X_new = X.slice(s![seen .., ..]);
  let y_new = y.slice(s![seen ..]);

  let mut iter_ok_flag = true;
  let mut round = 0;
  Zip::from(X_new.genrows()).and(&y_new)
    .apply(|x_, y_| {

      let curr_idx = seen + round;

      let X_prev = X.slice(s![.. curr_idx, ..]);
      let y_prev = y.slice(s![.. curr_idx]);

      match nn_seq(X_prev, y_prev, x_, *y_) {
        Ok((d_eq, d_neq, d_map)) => {
          {
            let mut dists_prev  =
              dists.slice_mut(s![.. curr_idx, ..]);
            let mut scores_prev =
              scores.slice_mut(s![.. curr_idx]);

            Zip::from(dists_prev.genrows_mut())
              .and(d_map.genrows())
              .and(&mut scores_prev)
              .apply(|mut ds, d_map, s| {
                     if ds[0] > d_map[0] {ds[0] = d_map[0];
                                          *s = ds.score();}
                else if ds[1] > d_map[1] {ds[1] = d_map[1];
                                          *s = ds.score();}
              });
            }

            dists[[curr_idx,0]] = d_eq;
            dists[[curr_idx,1]] = d_neq;

            scores[curr_idx] = (d_eq, d_neq).score();

            round += 1;
        },
        Err(_) => { iter_ok_flag = false; }
      }
    });

  if iter_ok_flag {
    Ok(())
  } else {
    Err(except(MSG_UPDATE))
  }
}
// }}}

// update_par {{{
fn update_par( X      : &MatrixOwned
             , y      : &LabelsOwned
             , dists  : &mut MatrixOwned
             , scores : &mut VectorOwned
             , seen   : usize            ) -> PyResult<()>
{
  let X_new = X.slice(s![seen .., ..]);
  let y_new = y.slice(s![seen ..]);

  let mut iter_ok_flag = true;
  let mut round = 0;
  Zip::from(X_new.genrows()).and(&y_new)
    .apply(|x_, y_| {

      let curr_idx = seen + round;

      let X_prev = X.slice(s![.. curr_idx, ..]);
      let y_prev = y.slice(s![.. curr_idx]);

      match nn_par(X_prev, y_prev, x_, *y_) {
        Ok((d_eq, d_neq, d_map)) => {
          {
            let mut dists_prev  =
              dists.slice_mut(s![.. curr_idx, ..]);
            let mut scores_prev =
              scores.slice_mut(s![.. curr_idx]);

            Zip::from(dists_prev.genrows_mut())
              .and(d_map.genrows())
              .and(&mut scores_prev)
              .par_apply(|mut ds, d_map, s| {
                     if ds[0] > d_map[0] {ds[0] = d_map[0];
                                          *s = ds.score();}
                else if ds[1] > d_map[1] {ds[1] = d_map[1];
                                          *s = ds.score();}
              });
            }

            dists[[curr_idx,0]] = d_eq;
            dists[[curr_idx,1]] = d_neq;

            scores[curr_idx] = (d_eq, d_neq).score();

            round += 1;
        },
        Err(_) => { iter_ok_flag = false; }
      }
    });

  if iter_ok_flag {
    Ok(())
  } else {
    Err(except(MSG_UPDATE))
  }
}
// }}}

// scores_seq {{{
fn scores_seq( x_: Vector, y_: i64, X: Matrix, y: Labels
             , dists: Matrix, scores: Vector )
  -> PyResult<VectorOwned>
{
    let scores_res_len = len(&X)? + 1;

    let mut scores_res =
      VectorOwned::default((scores_res_len, ));
    let mut scores_res_for_iter =
      scores_res.slice_mut(s![.. -1]);

    let (d_eq, d_neq, d_map) = nn_seq(X, y, x_, y_)?;

    Zip::from(&mut scores_res_for_iter)
      .and(&scores)
      .and(dists.genrows())
      .and(d_map.genrows())
      .apply(|s_, s, d, dm| {
             if d[0] > dm[0] {*s_ = (dm[0], d[1]).score();}
        else if d[1] > dm[1] {*s_ = (d[0], dm[1]).score();}
        else                 {*s_ = *s;                   }
      });

    let l = scores_res.len();
    scores_res[l - 1] = (d_eq, d_neq).score();
    Ok(scores_res)
}
// }}}

// scores_par {{{
fn scores_par( x_: Vector, y_: i64, X: Matrix, y: Labels
             , dists: Matrix, scores: Vector )
  -> PyResult<VectorOwned>
{
    let scores_res_len = len(&X)? + 1;

    let mut scores_res =
      VectorOwned::default((scores_res_len, ));
    let mut scores_res_for_iter =
      scores_res.slice_mut(s![.. -1]);

    let (d_eq, d_neq, d_map) = nn_par(X, y, x_, y_)?;

    Zip::from(&mut scores_res_for_iter)
      .and(&scores)
      .and(dists.genrows())
      .and(d_map.genrows())
      .par_apply(|s_, s, d, dm| {
             if d[0] > dm[0] {*s_ = (dm[0], d[1]).score();}
        else if d[1] > dm[1] {*s_ = (d[0], dm[1]).score();}
        else                 {*s_ = *s;                   }
      });

    let l = scores_res.len();
    scores_res[l - 1] = (d_eq, d_neq).score();
    Ok(scores_res)
}
// }}}

// nn_seq {{{
fn nn_seq(X: Matrix, y: Labels, x_: Vector, y_: i64)
  -> NNResult
{
  let (mut d_eq, mut d_neq)  = (inf, inf);

  let mut d_map = MatrixOwned::default((len(&X)?, 2));

  Zip::from(d_map.genrows_mut()).and(X.genrows()).and(&y)
    .apply(|mut dm, x, y| {
      let (d, eq) = (euclid_dist(x_, x), y_ == *y);

      if eq { if d < d_eq { d_eq = d } dm.update(inf, d); }
      else { if d < d_neq { d_neq = d } dm.update(inf, d);}
    });

    Ok((d_eq, d_neq, d_map.to_owned()))
}
// }}}

// nn_par {{{
fn nn_par(X: Matrix, y: Labels, x_: Vector, y_: i64)
  -> NNResult
{
  let d_eq  = SharedVar::init(inf);
  let d_neq = SharedVar::init(inf);

  let mut d_map = MatrixOwned::default((len(&X)?, 2));

  Zip::from(d_map.genrows_mut()).and(X.genrows()).and(&y)
    .par_apply(|mut dm, x, y| {
      let (d, eq) = (euclid_dist(x_, x), y_ == *y);

      if eq { let mut d_eq  = d_eq.lock().unwrap();
              if d < *d_eq { *d_eq = d }
              dm.update(d, inf); }
      else  { let mut d_neq = d_neq.lock().unwrap();
              if d < *d_neq { *d_neq = d }
              dm.update(inf, d); }
    });

  Ok((d_eq.unpack(), d_neq.unpack(), d_map.to_owned()))
}
// }}}

// euclid_dist {{{
fn euclid_dist(x: Vector, y: Vector) -> f64 {
  x.iter().zip(y.iter())
    .fold(0., |acc, (xi, yi)| acc + (xi - yi).powi(2))
    .sqrt()
}
// }}}

// except {{{
fn except(s: &'static str) -> PyErr {
  PyErr::new::<Exception, String>(String::from(s))
}
// }}}

// stack0 {{{
fn stack0<T: Copy, D: Dimension + RemoveAxis>(
  v: &[ArrayView<T, D>]
) -> PyResult<Array<T, D>>
{ stack(Axis(0), v).map_err(|_| except(MSG_STACK)) }
// }}}

// len {{{
fn len<T: Copy, D: Dimension>(X: &ArrayView<T, D>)
  -> PyResult<usize>
{ X.shape().first().map(|s| *s)
    .ok_or_else(|| except(MSG_SHAPE)) }
// }}}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
