use pyo3::prelude::*;

#[pymodule]
fn test_rs(_py: Python, m: &PyModule) -> PyResult<()> {

  #[pyfn(m, "sum_as_string")]
  fn sum_as_string(_py: Python, a:usize, b:usize)
    -> PyResult<String>
  {
    Ok((a + b).to_string())
  }

  #[pyfn(m, "sum_vec")]
  fn sum_vec(_py: Python, v: Vec<f64>) -> PyResult<f64> {
    Ok(v.iter().fold(0., |acc, x| acc + x))
  }

  /*
  #[pyfn(m, "nn")]
  fn nn(_py: Python, X:
  */

  Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
