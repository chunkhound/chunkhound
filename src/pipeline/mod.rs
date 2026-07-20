mod config;
mod differ;
#[allow(clippy::module_inception)]
mod pipeline;
mod report;
mod state;
mod types;

pub(crate) use pipeline::IndexingPipeline;
pub(crate) use report::PipelineReport;
