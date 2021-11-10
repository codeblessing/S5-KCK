use thiserror::Error;
use miette::Diagnostic;

#[derive(Error, Diagnostic, Debug)]
pub enum ScoreError {
    #[error(transparent)]
    #[diagnostic(code(score::lib))]
    ImageReadError(#[from] std::io::Error),
    #[error("Cannot decode image. Not in one of supported formats.")]
    #[diagnostic(code(score::lib))]
    ImageDecodeError
}