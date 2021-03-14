pub mod error;
pub mod pf;

use crate::error::QuestPlusError;
use crate::pf::{NormCDF, NormCDFParamPDF, NormCDFPriorPDFFactory};
use itertools::{iproduct, izip};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::stack;
use num::Float;
use statrs::distribution::{Normal, Univariate};
use std::collections::{HashMap, HashSet};
use std::error::Error;

#[derive(Debug)]
pub enum StimScale {
    Linear,
    Log10,
    Decibel,
}

#[derive(Debug)]
pub enum StimSelectionMethod {
    MinEntropy,
    /* todo
    MinNEntropy(i32),
     */
}

#[derive(Debug)]
pub enum ParamEstimationMethod {
    Mode,
    Mean,
}

trait QuestPlus {
    type T1;
    fn calc_pf(&self) -> Result<Self::T1, QuestPlusError>;
    fn next_stim(&self) -> Result<f64, QuestPlusError>;
}

impl QuestPlus for NormCDF {
    type T1 = Array5<f64>;

    fn calc_pf(&self) -> Result<Self::T1, QuestPlusError> {
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

    fn next_stim(&self) -> Result<f64, QuestPlusError>{
        let ax = 0;
        let mut new_posterior = self.likelihoods.clone();

        for (ax_p, ax_l) in izip!(&[0,1,2,3],&[2,3,4,5]){
            for (mut axis_new, axis, v) in izip!( new_posterior.axis_iter_mut(Axis(*ax_l)), self.posterior_pdf.axis_iter(Axis(*ax_p)), self.likelihoods.axis_iter(Axis(*ax_l))){
                dbg!(&axis_new);
                dbg!(&axis);
                dbg!(&v);
                axis_new = axis * v;
            }
        }


        let new_posterior = &self.posterior_pdf * &self.likelihoods;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::pf::{
        NormCDF, NormCDFParamDomain, NormCDFParamPDF, NormCDFPriorPDFFactory, NormCDFStimDomain,
    };
    use crate::{ParamEstimationMethod, QuestPlus, StimSelectionMethod};
    use ndarray::prelude::*;


    #[test]
    fn test_next_stim() {
        let intensity: Array1<f64> = Array1::range(0., 50., 1.);
        let mean: Array1<f64> = Array1::range(7., 9., 1.);
        let sd: Array1<f64> = Array1::range(7., 8., 0.5);
        let lower_asymptote: Array1<f64> = arr1(&[0.5]);
        let lapse_rate: Array1<f64> = Array1::range(0.01, 0.02, 0.01);

        let stim_domain = NormCDFStimDomain::new(intensity);
        let param_domain = NormCDFParamDomain::new(mean, sd, lower_asymptote, lapse_rate);
        let prior_pdf = NormCDFParamPDF::new(&param_domain, None, None, None, None).unwrap();

        let norm_cdf = NormCDF::new(
            stim_domain,
            param_domain,
            prior_pdf,
            StimSelectionMethod::MinEntropy,
            ParamEstimationMethod::Mean,
        )
            .unwrap();
        norm_cdf.next_stim().unwrap();
    }
}
