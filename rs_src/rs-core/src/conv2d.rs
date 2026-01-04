use anyhow::{Result, anyhow};
use tract_onnx::tract_hir::ops;
use tract_onnx::{
    prelude::{InferenceFact, Node},
    tract_hir::infer::InferenceOp,
};

use crate::type_def::FheInfo;

pub fn get_conv_info(
    node: &Node<InferenceFact, Box<dyn InferenceOp>>,
) -> Result<FheInfo, anyhow::Error> {
    let op = node.op();

    let conv = op
        .downcast_ref::<ops::cnn::Conv>()
        .ok_or(anyhow!("Failed to convert Conv"))?;

    Ok(FheInfo::default())
}
