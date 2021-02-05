use std::collections::HashMap;
use ndarray::prelude::*;
use num::{Float, Num};

struct QuestPlus<F, O>
    where
        F: Fn(/*TODO args*/),
{
    stim_domain: HashMap<String, Array1<f64>>,
    param_domain: HashMap<String, Array1<f64>>,
    outcome_domain: HashMap<String, O>,
    prior: Option<HashMap<String, Array1<f64>>>,
    func: F,
    stim_scalse: StimScale,
    stim_selection_method: StimSelectionMethod,
    // stim_selection_options: TODO
    param_estimation_method: ParamEstimationMethod,
}

enum StimScale {
    Linear,
    Log10,
    Decibel,
}

enum StimSelectionMethod {
    MinEntropy,
    MinNEntropy(i32),
}

enum ParamEstimationMethod {
    Mode,
    Mean,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
