# Deployment model artifacts

This directory is populated by `scripts/export_deployment_models.ps1` and is
mounted read-only at `/app/models` by Docker Compose.

Expected files:

```text
models/
|-- det/model_fp32.onnx
|-- rec/model_fp32.onnx
|-- table/model_fp32.onnx
|-- dictionaries/table_dict.txt
|-- dictionaries/table_structure_dict.txt
`-- manifest.json
```

FP16 files use `model_fp16.onnx`. Do not mix a model with a dictionary or
preprocessing shape from another training run. Large model binaries should be
stored in the release artifact store instead of source control.
