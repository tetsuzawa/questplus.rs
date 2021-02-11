use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuestPlusError {
    #[error("{0:?} not exists in {1:?}")]
    ParameterNotExists(HashSet<String>, HashSet<String>),
    #[error("length of {0} and {1} does not match")]
    ParameterLengthNotMatch(String, String),
    #[error("{0:?}")]
    NDArrayError(ndarray::ShapeError),
    #[error("{0:?}")]
    StatrsError(statrs::StatsError),
}