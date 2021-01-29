# Sample Conversion Example

This example will try to convert Mobilenetv2 model.

## Conversion

### PyTorch to ONNX

```bash
python torch_to_onnx.py
```

### ONNX to TF

```bash
python onnx_to_tf.py
```

### TF to TFlite

```bash
python tf_to_tflite.py
```

## Run Inference

### TF Model Inference

```bash
python run_tf_model.py
```

### TFLite Model Inference

```bash
python run_tflite_model.py
```