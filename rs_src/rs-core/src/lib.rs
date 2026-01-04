mod conv2d;
use std::path::Path;

use anyhow::Result;
use onnx_ir::ir::Node;
use onnx_ir::{OnnxGraph, OnnxGraphBuilder};

struct OnnxHandler {
    onnx: OnnxGraph,
}

impl OnnxHandler {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let graph = OnnxGraphBuilder::new().parse_bytes(bytes)?;

        Ok(OnnxHandler { onnx: graph })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, anyhow::Error> {
        let graph = OnnxGraphBuilder::new().parse_file(path)?;

        Ok(OnnxHandler { onnx: graph })
    }

    pub fn print_nodes(&self) -> () {
        for node in &self.onnx.nodes {
            match node {
                Node::Conv2d(conv_node) => {
                    println!(
                        "Conv2d with {} input channels",
                        conv_node.config.channels[0]
                    );
                    println!("Kernel size: {:?}", conv_node.config.kernel_size);
                }
                Node::Add(add_node) => {
                    println!("Add operation with {} inputs", add_node.inputs.len());
                }
                _ => {
                    println!("Other operation")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_read() {
        let model_bytes = include_bytes!("../../test_file/resnet18_pure_binary.onnx");
        let handler = OnnxHandler::from_bytes(model_bytes).unwrap();

        handler.print_nodes();
    }
}
