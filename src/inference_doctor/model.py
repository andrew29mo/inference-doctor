"""Preflight in the controller; excluded from isolated worker measurements."""


def inspect_model(path):
    import onnx
    from .validation import DiagnosticError

    model = onnx.load(path, load_external_data=False)

    def visit(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise DiagnosticError("external_model_data", "External ONNX tensor data is not supported; export a self-contained model.")
        for descriptor, value in message.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                for child in value if descriptor.is_repeated else [value]:
                    visit(child)

    visit(model)
    onnx.checker.check_model(model)
    return {"onnx": onnx.__version__, "ir_version": model.ir_version,
            "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
            "graph_nodes": len(model.graph.node), "self_contained": True}
