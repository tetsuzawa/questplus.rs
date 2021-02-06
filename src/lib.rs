use std::collections::HashMap;
use ndarray::prelude::*;
use num::{Float, Num};

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

struct QuestPlus<F, D, O>
    where
        F: Fn(/*TODO args*/),
        D: Dimension,
{
    stim_domain: HashMap<String, Array1<f64>>,
    param_domain: HashMap<String, Array1<f64>>,
    outcome_domain: HashMap<String, O>,
    prior: Array<f64, D>,
    posterior: Array<f64, D>,
    func: F,
    stim_scalse: StimScale,
    stim_selection_method: StimSelectionMethod,
    // stim_selection_options: TODO
    param_estimation_method: ParamEstimationMethod,
    entropy: f64,
    resp_history: Vec<O>,
    stim_history: Vec<f64>,
}

impl<F, D, O> QuestPlus<F, D, O>
    where
        F: Fn(/*TODO args*/),
        D: Dimension,
{
    fn new(
        stim_domain: HashMap<String, Array1<f64>>,
        param_domain: HashMap<String, Array1<f64>>,
        outcome_domain: HashMap<String, O>,
        prior: Option<HashMap<String, Array1<f64>>>,
        func: F,
        stim_scalse: StimScale,
        stim_selection_method: StimSelectionMethod,
        // stim_selection_options: TODO
        param_estimation_method: ParamEstimationMethod,
    ) -> Self {
        let prior = QuestPlus::<F, D, O>::gen_prior(prior);
        let posterior = prior.clone();
        QuestPlus {
            stim_domain,
            param_domain,
            outcome_domain,
            prior,
            posterior,
            func,
            stim_scalse,
            stim_selection_method,
            // stim_selection_options: TODO
            param_estimation_method,
            entropy: f64::max_value(),
            resp_history: Vec::new(),
            stim_history: Vec::new(),
        }
    }

    fn gen_prior(prior: Option<HashMap<String, Array1<f64>>>) -> Array<f64, D> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
