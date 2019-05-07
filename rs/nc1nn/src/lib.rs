//use rayon::prelude::*;
use pyo3::prelude::*;
use ndarray::{Zip, Array1};
use ndarray_parallel::prelude::*;
use numpy::{PyArray1, PyArray2};
use numpy::convert::{IntoPyArray};

use std::f64;
use std::sync::{Arc, Mutex};

#[pymodule]
fn nc1nn(_py: Python, m: &PyModule) -> PyResult<()> {

  #[pyfn(m, "nn")]
  fn nn( _py: Python
       , X  : &PyArray2<f64>
       , y  : &PyArray1<i64>
       , x_ : &PyArray1<f64>
       , y_ : i64 )
    -> PyResult<(f64, f64, Vec<(bool, f64)>)>
  {
    let X  = X.as_array();
    let y  = y.as_array();
    let x_ = x_.as_array();

    let mut d_eq  = f64::INFINITY;
    let mut d_neq = f64::INFINITY;


    let d_map: Vec<(bool, f64)> =
      X.outer_iter().zip(y.iter()).map(|(x, y)| {
        let d = x_.iter().zip(x.iter())
          .fold(0., |acc, (i, j)| acc + (i - j).powi(2))
          .sqrt();

        let eq = y_ == *y;

        if eq  && d < d_eq  { d_eq  = d } else
        if !eq && d < d_neq { d_neq = d }

        (eq, d)
    }).collect();

    Ok((d_eq, d_neq, d_map))
  }

  fn _nn_par(X  : &PyArray2<f64>
           , y  : &PyArray1<i64>
           , x_ : &PyArray1<f64>
           , y_ : i64 )
    -> (f64, f64, Array1<f64>, Array1<bool>)
  {
    let X  = X.as_array();
    let y  = y.as_array();
    let x_ = x_.as_array();

    let mut d_eq  = Arc::new(Mutex::new(f64::INFINITY));
    let mut d_neq = Arc::new(Mutex::new(f64::INFINITY));

    let mut d_map    = Array1::<f64>
      ::default((*X.shape().first().unwrap(), ));
    let mut d_map_eq = Array1::<bool>
      ::default((*X.shape().first().unwrap(), ));

    Zip::from(&mut d_map)
      .and(&mut d_map_eq)
      .and(X.genrows())
      .and(&y)
      .par_apply(|dm, dm_eq, x, y| {
        let d = x_.iter().zip(x.iter())
          .fold(0., |acc, (i, j)| acc + (i - j).powi(2))
          .sqrt();

        let eq = y_ == *y;

        if eq {
          let mut d_eq  = *d_eq.lock().unwrap();
          if d < d_eq { d_eq = d }
        } else {
          let mut d_neq = *d_neq.lock().unwrap();
          if d < d_neq { d_neq = d }
        }

        *dm = d;
        *dm_eq = eq;
      });

    let d_eq  = Arc::try_unwrap(d_eq).unwrap()
      .into_inner().unwrap();
    let d_neq = Arc::try_unwrap(d_neq).unwrap()
      .into_inner().unwrap();

    (d_eq, d_neq, d_map.to_owned(), d_map_eq.to_owned())
  }

  #[pyfn(m, "nn_par")]
  fn nn_par( py : Python
           , X  : &PyArray2<f64>
           , y  : &PyArray1<i64>
           , x_ : &PyArray1<f64>
           , y_ : i64 )
    -> PyResult<(f64, f64, Py<PyArray1<f64>>, Py<PyArray1<bool>>)>
  {
    let (d_eq, d_neq, d_map, d_map_eq) = py.allow_threads(
      move || _nn_par(X, y, x_, y_)
    );
    let d_map    = d_map.into_pyarray(py).to_owned();
    let d_map_eq = d_map_eq.into_pyarray(py).to_owned();
    Ok((d_eq, d_neq, d_map, d_map_eq))
  }

  Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
