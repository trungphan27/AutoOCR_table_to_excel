"""Convert FP32 ONNX models to FP16 while keeping public I/O in FP32."""

import argparse
import hashlib
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxconverter_common import float16


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_missing_node_names(graph, prefix="graph"):
    for index, node in enumerate(graph.node):
        if not node.name:
            node.name = "{}_{}_{}".format(prefix, node.op_type, index)
        for attribute_index, attribute in enumerate(node.attribute):
            if attribute.type == onnx.AttributeProto.GRAPH:
                assign_missing_node_names(
                    attribute.g,
                    "{}_{}_{}".format(prefix, node.name, attribute_index),
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph_index, child_graph in enumerate(attribute.graphs):
                    assign_missing_node_names(
                        child_graph,
                        "{}_{}_{}_{}".format(
                            prefix, node.name, attribute_index, graph_index
                        ),
                    )


def restore_empty_resize_inputs(graph):
    producers = {
        output: node for node in graph.node for output in node.output if output
    }
    for node in graph.node:
        if node.op_type == "Resize":
            # Empty tensors mark omitted Resize inputs. Cast nodes around these
            # tensors must be bypassed for ORT shape inference.
            for input_index in (1, 2):
                if input_index >= len(node.input):
                    continue
                source_name = node.input[input_index]
                # Avoid Constant(float32) -> float16 -> float32 -> Resize.
                for _ in range(4):
                    cast_node = producers.get(source_name)
                    if cast_node is None or cast_node.op_type != "Cast":
                        break
                    source_name = cast_node.input[0]
                constant_node = producers.get(source_name)
                if constant_node is None or constant_node.op_type != "Constant":
                    continue
                value_attributes = [
                    attribute
                    for attribute in constant_node.attribute
                    if attribute.name == "value"
                ]
                if value_attributes and list(value_attributes[0].t.dims) == [0]:
                    node.input[input_index] = source_name
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                restore_empty_resize_inputs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    restore_empty_resize_inputs(child_graph)


def empty_constant_node_names(graph):
    names = []
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value" and list(attribute.t.dims) == [0]:
                    names.append(node.name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                names.extend(empty_constant_node_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    names.extend(empty_constant_node_names(child_graph))
    return names


def convert_model(source, destination, overwrite=False):
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it.".format(
                destination
            )
        )
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    # Unique Constant names preserve SSA form during FP16 cast insertion.
    assign_missing_node_names(model.graph)
    converted = float16.convert_float_to_float16(
        model,
        min_positive_val=5.96e-08,
        max_finite_val=65504.0,
        # Native FP16 I/O avoids graph-input casts and excess host transfers.
        keep_io_types=False,
        # Resize and Range remain blocked at FP32 by the converter.
        disable_shape_infer=False,
        # Keep only empty optional-input constants in FP32.
        op_block_list=float16.DEFAULT_OP_BLOCK_LIST,
        node_block_list=empty_constant_node_names(model.graph),
    )
    restore_empty_resize_inputs(converted.graph)
    onnx.checker.check_model(converted)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    onnx.save(converted, str(temporary))
    onnx.checker.check_model(onnx.load(str(temporary)))
    # Session loading catches runtime shape contracts not covered by checker.
    ort.InferenceSession(
        str(temporary), providers=["CPUExecutionProvider"]
    )
    temporary.replace(destination)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FP32 ONNX models to GPU-oriented FP16 models."
    )
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the source model directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for source in args.models:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        output_dir = (args.output_dir or source.parent).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = source.stem
        if stem.endswith("_fp32"):
            stem = stem[:-5]
        destination = output_dir / "{}_fp16.onnx".format(stem)
        convert_model(source, destination, overwrite=args.overwrite)
        print(
            "saved {} bytes={} sha256={}".format(
                destination, destination.stat().st_size, sha256(destination)
            )
        )


if __name__ == "__main__":
    main()
