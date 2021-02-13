pub mod error;
pub mod pf;

use crate::error::QuestPlusError;
use crate::pf::NormCDF;
use itertools::iproduct;
use itertools::Itertools;
use ndarray::prelude::*;
use num::Float;
use statrs::distribution::{Normal, Univariate};
use std::collections::{HashMap, HashSet};
use std::error::Error;

enum StimScale {
    Linear,
    Log10,
    Decibel,
}

enum StimSelectionMethod {
    MinEntropy,
    /* todo
    MinNEntropy(i32),
     */
}

enum ParamEstimationMethod {
    Mode,
    Mean,
}

trait QuestPlus {
    type T;
    fn calc_pf(&self) -> Result<Self::T, QuestPlusError>;
    // fn gen_prior(&self) -> Result<Self::T, QuestPlusError>;
    // fn gen_likelihoods(&self) ->Result<Self::T, QuestPlusError>;
}

impl QuestPlus for NormCDF {
    type T = Array5<f64>;

    fn calc_pf(&self) -> Result<Self::T, QuestPlusError> {
        let num_elements = self.stim_domain.intensity.len()
            * self.param_domain.mean.len()
            * self.param_domain.sd.len()
            * self.param_domain.lower_asymptote.len()
            * self.param_domain.lapse_rate.len();
        let mut v = Vec::with_capacity(num_elements);
        for (x, m, s, la, lr) in iproduct!(
            self.stim_domain.intensity.iter(),
            self.param_domain.mean.iter(),
            self.param_domain.sd.iter(),
            self.param_domain.lower_asymptote.iter(),
            self.param_domain.lapse_rate.iter()
        ) {
            v.push(NormCDF::f(*x, *m, *s, *la, *lr)?);
        }
        match Array5::from_shape_vec(
            (
                self.stim_domain.intensity.len(),
                self.param_domain.mean.len(),
                self.param_domain.sd.len(),
                self.param_domain.lower_asymptote.len(),
                self.param_domain.lapse_rate.len(),
            ),
            v,
        ) {
            Ok(a) => Ok(a),
            Err(e) => Err(QuestPlusError::NDArrayError(e)),
        }
    }
}
