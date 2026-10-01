mod config;
mod differ;
mod parse_call_config;
// `pipeline::pipeline` holds the core IndexingPipeline type; splitting it
// into its own file under the `pipeline/` directory module is clearer than
// renaming either the file or the type just to dodge this lint.
#[allow(clippy::module_inception)]
mod pipeline;
mod report;
mod types;

pub(crate) use parse_call_config::ParseCallConfig;
pub(crate) use pipeline::IndexingPipeline;
pub(crate) use report::PipelineReport;
