use crate::error::QuestPlusError;
use itertools::iproduct;
use ndarray::prelude::*;
use statrs::distribution::{Normal, Univariate};
use std::error::Error;

pub struct NormCDFStimDomain {
    pub intensity: Array1<f64>,
}

impl NormCDFStimDomain {
    pub fn new(intensity: Array1<f64>) -> Self {
        NormCDFStimDomain { intensity }
    }
}

pub struct NormCDFParamDomain {
    pub mean: Array1<f64>,
    pub sd: Array1<f64>,
    pub lower_asymptote: Array1<f64>,
    pub lapse_rate: Array1<f64>,
}

impl NormCDFParamDomain {
    pub fn new(
        mean: Array1<f64>,
        sd: Array1<f64>,
        lower_asymptote: Array1<f64>,
        lapse_rate: Array1<f64>,
    ) -> Self {
        NormCDFParamDomain {
            mean,
            sd,
            lower_asymptote,
            lapse_rate,
        }
    }
}

pub type NormCDFParamPDF = Array4<f64>;

pub trait NormCDFPriorPDFFactory {
    fn new(
        param_domain: &NormCDFParamDomain,
        mean: Option<Array1<f64>>,
        sd: Option<Array1<f64>>,
        lower_asymptote: Option<Array1<f64>>,
        lapse_rate: Option<Array1<f64>>,
    ) -> Result<Array4<f64>, QuestPlusError>;
}

impl NormCDFPriorPDFFactory for NormCDFParamPDF {
    fn new(
        param_domain: &NormCDFParamDomain,
        mean: Option<Array1<f64>>,
        sd: Option<Array1<f64>>,
        lower_asymptote: Option<Array1<f64>>,
        lapse_rate: Option<Array1<f64>>,
    ) -> Result<Self, QuestPlusError> {
        let mean_prior = match mean {
            Some(s) => {
                if param_domain.mean.len() != s.len() {
                    return Err(QuestPlusError::ParameterLengthNotMatch(
                        "mean".to_string(),
                        "mean_prior".to_string(),
                    ));
                }
                s
            }
            None => Array1::ones(param_domain.mean.len()),
        };
        let sum = mean_prior.sum();
        let mean_prior = mean_prior.mapv(|v| v / sum);

        let sd_prior = match sd {
            Some(s) => {
                if param_domain.sd.len() != s.len() {
                    return Err(QuestPlusError::ParameterLengthNotMatch(
                        "sd".to_string(),
                        "sd_prior".to_string(),
                    ));
                }
                s
            }
            None => Array1::ones(param_domain.sd.len()),
        };
        let sum = sd_prior.sum();
        let sd_prior = sd_prior.mapv(|v| v / sum);

        let lower_asymptote_prior = match lower_asymptote {
            Some(s) => {
                if param_domain.lower_asymptote.len() != s.len() {
                    return Err(QuestPlusError::ParameterLengthNotMatch(
                        "lower_asymptote".to_string(),
                        "lower_asymptote_prior".to_string(),
                    ));
                }
                s
            }
            None => Array1::ones(param_domain.lower_asymptote.len()),
        };
        let sum = lower_asymptote_prior.sum();
        let lower_asymptote_prior = lower_asymptote_prior.mapv(|v| v / sum);

        let lapse_rate_prior = match lapse_rate {
            Some(s) => {
                if param_domain.lapse_rate.len() != s.len() {
                    return Err(QuestPlusError::ParameterLengthNotMatch(
                        "lapse_rate".to_string(),
                        "lapse_rate_prior".to_string(),
                    ));
                }
                s
            }
            None => Array1::ones(param_domain.lapse_rate.len()),
        };
        let sum = lapse_rate_prior.sum();
        let lapse_rate_prior = lapse_rate_prior.mapv(|v| v / sum);

        let mut v = Vec::with_capacity(
            mean_prior.len()
                * sd_prior.len()
                * lower_asymptote_prior.len()
                * lapse_rate_prior.len(),
        );
        for (m, s, la, lr) in iproduct!(
            mean_prior.iter(),
            sd_prior.iter(),
            lower_asymptote_prior.iter(),
            lapse_rate_prior.iter()
        ) {
            v.push(m * s * la * lr)
        }

        let res = Array4::from_shape_vec(
            (
                mean_prior.len(),
                sd_prior.len(),
                lower_asymptote_prior.len(),
                lapse_rate_prior.len(),
            ),
            v,
        );
        let res = match res {
            Ok(a) => a,
            Err(e) => return Err(QuestPlusError::NDArrayError(e)),
        };
        let sum = res.sum();
        let res = res.mapv(|v| v / sum);
        Ok(res)
    }
}

pub struct NormCDF {
    pub param_domain: NormCDFParamDomain,
    pub prior_pdf: NormCDFParamPDF,
    pub stim_domain: NormCDFStimDomain,
}

impl NormCDF {
    pub fn new(
        stim_domain: NormCDFStimDomain,
        param_domain: NormCDFParamDomain,
        prior_pdf: NormCDFParamPDF,
    ) -> Self {
        NormCDF {
            stim_domain,
            param_domain,
            prior_pdf,
        }
    }

    pub fn f(
        intensity: f64,
        mean: f64,
        sd: f64,
        lower_asymptote: f64,
        lapse_rate: f64,
    ) -> Result<f64, QuestPlusError> {
        let norm = match Normal::new(mean, sd) {
            Ok(a) => a,
            Err(e) => return Err(QuestPlusError::StatrsError(e)),
        };
        Ok(lower_asymptote + (1.0 - lower_asymptote - lapse_rate) * norm.cdf(intensity))
    }
}

#[cfg(test)]
mod tests {
    use crate::pf::{NormCDF, NormCDFParamDomain, NormCDFParamPDF, NormCDFPriorPDFFactory, NormCDFStimDomain};
    use crate::QuestPlus;
    use ndarray::prelude::*;

    #[test]
    fn test_norm_cdf() {
        let intensity: Array1<f64> = Array1::range(0., 50., 1.);
        let mean: Array1<f64> = Array1::range(7., 9., 1.);
        let sd: Array1<f64> = Array1::range(7., 8., 0.5);
        let lower_asymptote: Array1<f64> = arr1(&[0.5]);
        let lapse_rate: Array1<f64> = Array1::range(0.01, 0.02, 0.01);
        // want_slice is generated by Python code
        let want_slice = [
            [
                [[[0.577741074426414]], [[0.5859087329775925]]],
                [[[0.5620089876920433]], [[0.5700999841757994]]],
            ],
            [
                [[[0.5958846548853503]], [[0.6038091453058644]]],
                [[[0.577741074426414]], [[0.5859087329775925]]],
            ],
            [
                [[[0.6163873783932184]], [[0.6237213433979922]]],
                [[[0.5958846548853503]], [[0.6038091453058644]]],
            ],
            [
                [[[0.6390887457183514]], [[0.6454817000158871]]],
                [[[0.6163873783932184]], [[0.6237213433979922]]],
            ],
            [
                [[[0.6637176097398361]], [[0.6688433466109411]]],
                [[[0.6390887457183514]], [[0.6454817000158871]]],
            ],
            [
                [[[0.6898987557380163]], [[0.6934828261273723]]],
                [[[0.6637176097398361]], [[0.6688433466109411]]],
            ],
            [
                [[[0.7171687365599306]], [[0.7190127928544292]]],
                [[[0.6898987557380163]], [[0.6934828261273723]]],
            ],
            [
                [[[0.745]], [[0.745]]],
                [[[0.7171687365599306]], [[0.7190127928544292]]],
            ],
            [
                [[[0.7728312634400694]], [[0.7709872071455709]]],
                [[[0.745]], [[0.745]]],
            ],
            [
                [[[0.8001012442619837]], [[0.7965171738726278]]],
                [[[0.7728312634400694]], [[0.7709872071455709]]],
            ],
            [
                [[[0.8262823902601639]], [[0.8211566533890589]]],
                [[[0.8001012442619837]], [[0.7965171738726278]]],
            ],
            [
                [[[0.8509112542816486]], [[0.844518299984113]]],
                [[[0.8262823902601639]], [[0.8211566533890589]]],
            ],
            [
                [[[0.8736126216067814]], [[0.8662786566020078]]],
                [[[0.8509112542816486]], [[0.844518299984113]]],
            ],
            [
                [[[0.8941153451146497]], [[0.8861908546941357]]],
                [[[0.8736126216067814]], [[0.8662786566020078]]],
            ],
            [
                [[[0.912258925573586]], [[0.9040912670224075]]],
                [[[0.8941153451146497]], [[0.8861908546941357]]],
            ],
            [
                [[[0.9279910123079567]], [[0.9199000158242006]]],
                [[[0.912258925573586]], [[0.9040912670224075]]],
            ],
            [
                [[[0.9413570155467679]], [[0.933615861591363]]],
                [[[0.9279910123079567]], [[0.9199000158242006]]],
            ],
            [
                [[[0.952483774500181]], [[0.9453065023343248]]],
                [[[0.9413570155467679]], [[0.933615861591363]]],
            ],
            [
                [[[0.9615596322340295]], [[0.9550956450671468]]],
                [[[0.952483774500181]], [[0.9453065023343248]]],
            ],
            [
                [[[0.9688133149540519]], [[0.9631483470672166]]],
                [[[0.9615596322340295]], [[0.9550956450671468]]],
            ],
            [
                [[[0.9744937461028305]], [[0.9696560723524983]]],
                [[[0.9688133149540519]], [[0.9631483470672166]]],
            ],
            [
                [[[0.9788524353453922]], [[0.9748227029036971]]],
                [[[0.9744937461028305]], [[0.9696560723524983]]],
            ],
            [
                [[[0.9821294800541241]], [[0.9788524353453922]]],
                [[[0.9788524353453922]], [[0.9748227029036971]]],
            ],
            [
                [[[0.984543610154988]], [[0.9819401390468547]]],
                [[[0.9821294800541241]], [[0.9788524353453922]]],
            ],
            [
                [[[0.9862861824750273]], [[0.9842644039405264]]],
                [[[0.984543610154988]], [[0.9819401390468547]]],
            ],
            [
                [[[0.9875186423153992]], [[0.9859832073969479]]],
                [[[0.9862861824750273]], [[0.9842644039405264]]],
            ],
            [
                [[[0.9883727380582754]], [[0.9872319053497753]]],
                [[[0.9875186423153992]], [[0.9859832073969479]]],
            ],
            [
                [[[0.9889526901797577]], [[0.988123113521881]]],
                [[[0.9883727380582754]], [[0.9872319053497753]]],
            ],
            [
                [[[0.9893385499645013]], [[0.9887479861380903]]],
                [[[0.9889526901797577]], [[0.988123113521881]]],
            ],
            [
                [[[0.9895900966930726]], [[0.9891784080685382]]],
                [[[0.9893385499645013]], [[0.9887479861380903]]],
            ],
            [
                [[[0.9897507758691695]], [[0.9894696727641172]]],
                [[[0.9895900966930726]], [[0.9891784080685382]]],
            ],
            [
                [[[0.9898513421228192]], [[0.9896633024104212]]],
                [[[0.9897507758691695]], [[0.9894696727641172]]],
            ],
            [
                [[[0.989913015351717]], [[0.9897897604367336]]],
                [[[0.9898513421228192]], [[0.9896633024104212]]],
            ],
            [
                [[[0.9899500742774074]], [[0.9898708960418583]]],
                [[[0.989913015351717]], [[0.9897897604367336]]],
            ],
            [
                [[[0.9899718935703437]], [[0.9899220367908228]]],
                [[[0.9899500742774074]], [[0.9898708960418583]]],
            ],
            [
                [[[0.9899844810915017]], [[0.9899537042488459]]],
                [[[0.9899718935703437]], [[0.9899220367908228]]],
            ],
            [
                [[[0.9899915963622526]], [[0.9899729683978367]]],
                [[[0.9899844810915017]], [[0.9899537042488459]]],
            ],
            [
                [[[0.9899955372521985]], [[0.9899844810915017]]],
                [[[0.9899915963622526]], [[0.9899729683978367]]],
            ],
            [
                [[[0.9899976759470539]], [[0.9899912402799464]]],
                [[[0.9899955372521985]], [[0.9899844810915017]]],
            ],
            [
                [[[0.989998813194489]], [[0.9899951388256961]]],
                [[[0.9899976759470539]], [[0.9899912402799464]]],
            ],
            [
                [[[0.98999940572772]], [[0.9899973478534851]]],
                [[[0.989998813194489]], [[0.9899951388256961]]],
            ],
            [
                [[[0.9899997082253964]], [[0.9899985775280102]]],
                [[[0.98999940572772]], [[0.9899973478534851]]],
            ],
            [
                [[[0.9899998595407298]], [[0.9899992499928991]]],
                [[[0.9899997082253964]], [[0.9899985775280102]]],
            ],
            [
                [[[0.9899999337051114]], [[0.9899996112692055]]],
                [[[0.9899998595407298]], [[0.9899992499928991]]],
            ],
            [
                [[[0.9899999693222761]], [[0.9899998019468436]]],
                [[[0.9899999337051114]], [[0.9899996112692055]]],
            ],
            [
                [[[0.9899999860823069]], [[0.9899999008136375]]],
                [[[0.9899999693222761]], [[0.9899998019468436]]],
            ],
            [
                [[[0.9899999938098557]], [[0.989999951174311]]],
                [[[0.9899999860823069]], [[0.9899999008136375]]],
            ],
            [
                [[[0.9899999973009386]], [[0.9899999763756135]]],
                [[[0.9899999938098557]], [[0.989999951174311]]],
            ],
            [
                [[[0.9899999988462997]], [[0.9899999887648576]]],
                [[[0.9899999973009386]], [[0.9899999763756135]]],
            ],
            [
                [[[0.989999999516572]], [[0.9899999947483807]]],
                [[[0.9899999988462997]], [[0.9899999887648576]]],
            ],
        ];
        let num_elements =
            intensity.len() * mean.len() * sd.len() * lower_asymptote.len() * lapse_rate.len();
        let mut want_vec = Vec::with_capacity(num_elements);
        for a in want_slice.iter() {
            for b in a.iter() {
                for c in b.iter() {
                    for d in c.iter() {
                        for e in d.iter() {
                            want_vec.push(*e);
                        }
                    }
                }
            }
        }

        let want: Array5<f64> = Array::from_shape_vec(
            (
                intensity.len(),
                mean.len(),
                sd.len(),
                lower_asymptote.len(),
                lapse_rate.len(),
            ),
            want_vec,
        )
        .unwrap()
        .into();

        let stim_domain = NormCDFStimDomain::new(intensity);
        let param_domain = NormCDFParamDomain::new(mean, sd, lower_asymptote, lapse_rate);
        let prior_pdf = NormCDFParamPDF::new(&param_domain, None, None, None, None).unwrap();

        let norm_cdf = NormCDF::new(stim_domain, param_domain, prior_pdf);

        let result = norm_cdf.calc_pf().unwrap();
        println!("{:?}", result);

        assert!(result.all_close(&want, 1e-8));
    }
}
