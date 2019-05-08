#![allow(non_snake_case)]

use pyo3::prelude::*;

//#[macro_use(s)]
//extern crate ndarray;
use ndarray::{ Zip, Array2, ArrayView1, ArrayViewMut1
             , ArrayView2, stack, Axis };
use ndarray_parallel::prelude::*;

use numpy::{PyArray1, PyArray2};
use numpy::convert::{IntoPyArray};

use std::f64;
use std::sync::{Arc, Mutex};

// _nn {{{
fn _nn( X  : ArrayView2<f64>, y  : ArrayView1<i64>
      , x_ : ArrayView1<f64>, y_ : i64 )
  -> (f64, f64, Array2<f64>)
{
  let mut d_eq  = f64::INFINITY;
  let mut d_neq = f64::INFINITY;

  let mut d_map = Array2::<f64>
    ::default((*X.shape().first().unwrap(), 2));

  Zip::from(d_map.genrows_mut())
    .and(X.genrows())
    .and(&y)
    .apply(|mut dm, x, y| {
      let d = x_.iter().zip(x.iter())
        .fold(0., |acc, (i, j)| acc + (i - j).powi(2))
        .sqrt();

      let eq = y_ == *y;

      if eq {
        if d < d_eq { d_eq = d }
        dm[0] = d; dm[1] = f64::INFINITY;
      } else {
        if d < d_neq { d_neq = d }
        dm[0] = f64::INFINITY; dm[1] = d;
      }
    });

    (d_eq, d_neq, d_map.to_owned())
}
// }}}

// _nn_par {{{
fn _nn_par( X  : ArrayView2<f64>, y  : ArrayView1<i64>
          , x_ : ArrayView1<f64>, y_ : i64 )
  -> (f64, f64, Array2<f64>)
{
  let d_eq  = Arc::new(Mutex::new(f64::INFINITY));
  let d_neq = Arc::new(Mutex::new(f64::INFINITY));

  let mut d_map = Array2::<f64>
    ::default((*X.shape().first().unwrap(), 2));

  Zip::from(d_map.genrows_mut())
    .and(X.genrows())
    .and(&y)
    .par_apply(|mut dm, x, y| {
      let d = x_.iter().zip(x.iter())
        .fold(0., |acc, (i, j)| acc + (i - j).powi(2))
        .sqrt();

      let eq = y_ == *y;

      if eq {
        let mut d_eq  = d_eq.lock().unwrap();
        if d < *d_eq { *d_eq = d }
        dm[0] = d; dm[1] = f64::INFINITY;
      } else {
        let mut d_neq = d_neq.lock().unwrap();
        if d < *d_neq { *d_neq = d }
        dm[0] = f64::INFINITY; dm[1] = d;
      }
    });

  let d_eq  = Arc::try_unwrap(d_eq).unwrap()
    .into_inner().unwrap();
  let d_neq = Arc::try_unwrap(d_neq).unwrap()
    .into_inner().unwrap();

  (d_eq, d_neq, d_map.to_owned())
}
// }}}

fn score_from_array(ds: &ArrayViewMut1<f64>) -> f64 {
  score_from_scalars(ds[0], ds[1])
}

fn score_from_scalars(d_eq: f64, d_neq: f64) -> f64 {
       if d_eq == d_neq { 0. }
  else if d_neq > 0.0   { d_eq / d_neq }
  else                  { f64::INFINITY }
}

#[pymodule]
fn nc1nn(_py: Python, m: &PyModule) -> PyResult<()> {

  #[pyfn(m, "update")]
  fn update( py     : Python
           , X_new  : &PyArray2<f64>
           , y_new  : &PyArray1<i64>
           , X_seen : &PyArray2<f64>
           , y_seen : &PyArray1<i64>
           , dists  : &PyArray2<f64>
           , scores : &PyArray1<f64> )
    -> PyResult<( Py<PyArray2<f64>>, Py<PyArray1<i64>>
                , Py<PyArray2<f64>>, Py<PyArray1<f64>> )>
  {
    let X_new  = X_new.as_array();
    let y_new  = y_new.as_array();
    let mut X_seen = X_seen.to_owned_array();
    let mut y_seen = y_seen.to_owned_array();
    let mut dists  = dists.to_owned_array();
    let mut scores = scores.to_owned_array();

    Zip::from(X_new.genrows()).and(&y_new)
      .apply(|x, y| {
        let (d_eq, d_neq, d_map) =
          _nn_par(X_seen.view(), y_seen.view(), x, *y);

        Zip::from(dists.genrows_mut())
          .and(d_map.genrows())
          .and(&mut scores)
          .apply(|mut ds, d_map, s| {
            if ds[0] > d_map[0] {
              ds[0] = d_map[0];
              *s = score_from_array(&ds);
            } else if ds[1] > d_map[1] {
              ds[1] = d_map[1];
              *s = score_from_array(&ds);
            }
          });

        let x = x.insert_axis(Axis(0));

        let dist_entry = [d_eq, d_neq];
        let dist_append = ArrayView1::<f64>
          ::from(&dist_entry).insert_axis(Axis(0));

        let scores_append =
          &[score_from_scalars(d_eq, d_neq)];

        X_seen = stack![Axis(0), X_seen, x];
        y_seen = stack![Axis(0), y_seen, &[*y]];
        dists  = stack![Axis(0), dists,  dist_append];
        scores = stack![Axis(0), scores, scores_append];
    });

    let X_seen = X_seen.into_pyarray(py).to_owned();
    let y_seen = y_seen.into_pyarray(py).to_owned();
    let dists  = dists.into_pyarray(py).to_owned();
    let scores = scores.into_pyarray(py).to_owned();

    Ok((X_seen, y_seen, dists, scores))
  }

  // scores {{{
  /*
  #[pyfn(m, "scores")]
  fn scores( _py    : Python
           , scores : &PyArray1<f64>
           , X      : &PyArray2<f64>
           , y      : &PyArray1<i64>
           , x_     : &PyArray1<f64>
           , y_     : i64 ) -> Py<PyArray1<f64>>
  {
    let scores = scores.as_array();
    let X      = X.as_array();
    let y      = y.as_array();
    let x_     = x_.as_array();

    let mut scores_ = Array1::<f64>
      ::default((*X.shape().first().unwrap() + 1, ));

    let mut scores__ = scores_.slice_mut(s![.. -1]);

    let (d_eq, d_neq, d_map) = _nn(X, y, x_, y_);

    Zip::from(&mut scores__)
      .and(&X)
      .and(&d_map)
      .and(&d_map_eq)
      .apply(|s, x, d, eq| {
      });

    unimplemented!()
  }
  */
  // }}}

  // nn {{{
  #[pyfn(m, "nn")]
  fn nn( py : Python
       , X  : &PyArray2<f64>
       , y  : &PyArray1<i64>
       , x_ : &PyArray1<f64>
       , y_ : i64 ) -> PyResult<( f64
                                , f64
                                , Py<PyArray2<f64>> )>
  {
    let X  = X.as_array();
    let y  = y.as_array();
    let x_ = x_.as_array();

    let (d_eq, d_neq, d_map) = _nn(X, y, x_, y_);

    let d_map = d_map.into_pyarray(py).to_owned();
    Ok((d_eq, d_neq, d_map))
  }
  // }}}

  // nn_par {{{
  #[pyfn(m, "nn_par")]
  fn nn_par( py : Python
           , X  : &PyArray2<f64>
           , y  : &PyArray1<i64>
           , x_ : &PyArray1<f64>
           , y_ : i64 ) -> PyResult<( f64
                                    , f64
                                    , Py<PyArray2<f64>> )>
  {
    let X  = X.as_array();
    let y  = y.as_array();
    let x_ = x_.as_array();

    let (d_eq, d_neq, d_map) = py.allow_threads(
      move || _nn_par(X, y, x_, y_)
    );

    let d_map = d_map.into_pyarray(py).to_owned();
    Ok((d_eq, d_neq, d_map))
  }
  // }}}

  Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
