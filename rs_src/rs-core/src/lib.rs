mod add;
mod conv2d;
mod type_def;
use std::io::Cursor;
use std::path::Path;

use crate::conv2d::get_conv_info;
use crate::type_def::FheInfo;
use anyhow::Result;
use tracing::info;
use tract_onnx::Onnx;
use tract_onnx::prelude::{Framework, InferenceModel, InferenceModelExt};
use tract_onnx::tract_hir::ops::konst::Const;

struct OnnxHandler {
    hir: InferenceModel,
    info: FheInfo,
    boot: u32,
    depth: u32,
}

impl OnnxHandler {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let mut reader = Cursor::new(bytes);
        let onnx_tool = tract_onnx::onnx();

        let proto_model = Onnx::proto_model_for_read(&onnx_tool, &mut reader)?;
        let parsed = onnx_tool.parse(&proto_model, None)?;

        let hir = parsed.model;

        Ok(OnnxHandler {
            hir,
            info: FheInfo::default(),
            boot: 0,
            depth: 0,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, anyhow::Error> {
        let onnx_tool = tract_onnx::onnx();
        let proto_model = Onnx::model_for_path(&onnx_tool, path)?;
        let hir = proto_model;

        Ok(OnnxHandler {
            hir,
            info: FheInfo::default(),
            boot: 0,
            depth: 0,
        })
    }

    pub fn print_nodes(&self) -> () {
        let nodes = &self.hir.nodes;
        for node in nodes {
            let name = &node.name;
            let id = node.id;
            let op_name = node.op.name();

            info!("[ID: {id:>3}] Name: {name:<20} | Op: {op_name}");
            let res = get_conv_info(node);
            if res.is_ok() {
                info!("detect conv");
            } else {
                info!("failed to detect conv");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn test_onnx_read() {
        let model_bytes = include_bytes!("../../test_file/resnet18_pure_binary.onnx");
        let handler = OnnxHandler::from_bytes(model_bytes).unwrap();

        handler.print_nodes();
    }
}
