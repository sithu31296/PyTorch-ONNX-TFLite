# ONNX

## Tracing vs Scripting

ONNX model can be exported with both tracing or scripting.

* **Tracing**: Executes the model once and only export the operators which were actually run during this run. 
    * This means that if your model is dynamic, e.g., changes behavior depending on input data, the export won't be accurate. Similarly, a trace is likely to be valid only for a specific input size (which is one reason why we require explicit inputs on tracing). If your model contains control flows like for loops and if conditions, tracing will unroll the loops and if conditions, exporting a static graph that is exactly the same as this run. If you want to export the model with dynamic control flows, you will need to use scripting.
* **Scripting**: The model you are trying to export is a TorchScript `ScriptModule`. It creates serializable and optimizable models from PyTorch code.
* ONNX export also allows mixing tracing and scripting to suit the particular requirements of a part of a model.

## Tracing Export

```python
import torch

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        for i in range(y):
            x = x + i
        return x

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True, input_names=['input_data', 'loop_count'])
```

With tracing, you will get the ONNX graph which unrolls the for loop:

```
TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  for i in range(y):
```

You will get a `TracerWarning` which is telling this tracing model will not work with another `y` value.

If you print the model graph using:

```python
import onnx

model = onnx.load("loop.onnx")
print(onnx.helper.printable_graph(model.graph))
```

The model graph will be like this:

```
graph torch-jit-export (
  %input_data[INT64, 2x3]
) {
  %2 = Constant[value = <Scalar Tensor []>]()
  %3 = Add(%input_data, %2)
  %4 = Constant[value = <Scalar Tensor []>]()
  %5 = Add(%3, %4)
  %6 = Constant[value = <Scalar Tensor []>]()
  %7 = Add(%5, %6)
  %8 = Constant[value = <Scalar Tensor []>]()
  %9 = Add(%7, %8)
  %10 = Constant[value = <Scalar Tensor []>]()
  %11 = Add(%9, %10)
  return %11
}
```

You will see `y` becomes part of the model graph instead of being another input to the model.

## Scripting Export

To capture the dynamic loop, you can write the loop in script and call it from the regular `nn.Module`.

```python
@torch.jit.script
def loop(x, y):
    for i in range(y):
        x = x + i
    return x

class LoopModel2(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

model = LoopModel2()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True, input_names=['input_data', 'loop_count'])
```

Now, if you print the model graph:

```
graph torch-jit-export (
  %input_data[INT64, 2x3]
  %loop_count[INT64, scalar]
) initializers (
  %10[BOOL, scalar]
) {
  %2 = Constant[value = <Scalar Tensor []>]()
  %4 = Loop[body = <graph torch-jit-export1>](%loop_count, %10, %input_data)
  return %4
}

graph torch-jit-export1 (
  %i.1[INT64, scalar]
  %cond[BOOL]
  %x.6[INT64, 2x3]
) {
  %8 = Add(%x.6, %i.1)
  %9 = Cast[to = 9](%2)
  return %9, %8
}
```

You will see the dynamic control flow is captured correctly which will work with different loop count.

> Note: Avoid using `torch.Tensor.item()` for converting a scalar tensor to a fixed value constant. Torch already supports implicit cast of single-element tensors to numbers as you saw above.

> Note: Only tuples, lists and Variables are supports as model inputs and outputs. Dictionaries and strings are also accepted but their usage is not recommended.

> Note: In-place ops (like data[index]=new_data) are supported in opset version >= 11.

> Note: Custom operators are supported but shouldn't use because other conversion methods like onnx-tf will not likely to support.


## API

This does not contain complete arguments to `export` function. Only important or most used ones are included. Other ones are left out. 

`torch.onnx.export(
    model,
    args,
    f,
    verbose=False,
    input_names=[],
    output_names=[],
    opset_version=9,
    do_constant_folding=False,
    example_outputs=None,
    dynamic_axes=None
)`

* model (`nn.Module`): Model to be exported.
* args (`tuple(torch.Tensor)` or `torch.Tensor`): Inputs to the model.
* f (`str`): A binary protobuf will be written to this file.
* verbose (`bool`): If True, will print out a debug description of the trace being exported.
* input_names (`list(str)`): Names to assign to the input nodes of the graph in order.
* output_names (`list(str)`): Names to assign to the output nodes of the graph in order.
* opset_version (`int`): Opset version used in exporting the model.
* do_constant_folding (`bool`): Constant-folding optimization will replace some of the ops that have all constant inputs with pre-computed constant nodes. If True, the constant-folding optimization is applied to the model during export.
* example_outputs (`tuple(torch.Tensor)`): Model's example outputs being exported. This must be provided when exporting a ScriptModule or TorchScript function.
* dynamic_axes (`dict(str: int)` or `dict(str: list(int))` or `dict(str: dict(int: str))`): A dictionary to specify dynamic axes of input/output such that - key: input and/or output names and value: index of dynamic axes for given key and potentially the name to be used for exported dynamic axes. 

## Dynamic Shape Input

The value of the dynamic axes can be defined according to one of the following ways or a combination of both:

1. A list of integers specifying the dynamic axes of provided input. In this secnario, automated names will be generated and applied to dynamic axes of provided input/output during export.
2. An inner dictionary that specifies a mapping from the index of dynamic axis in corresponding input/output to the name that is desired to be applied on such axis of such input/output during export.

For example, if we have the following shape for inputs and outputs:

```
shape (input_1) = ('b', 3, 'w', 'h')
shape (input_2) = ('b', 4)
shape (output) = ('b', 'd', 5)
```

Then dynamic axes can be defined either as:

1. Only indices (automatic names will be generated for exported dynamic axes):

```python
dynamic_axes={
    'input_1': [0, 2, 3],
    'input_2': [0],
    'output': [0, 1]
}
```

2. Indices with corresponding names (provided names will be applied to exported dynamic axes):

```python
dynamic_axes={
    'input_1': {0: 'batch', 1: 'width', 2: 'height'},
    'input_2': {0: 'batch'},
    'output': {0: 'batch', 1: 'detections'}
}
```

3. Mixed Mode of 1 and 2:

```python
dynamic_axes={
    'input_1': [0, 2, 3],
    'input_2': {0: 'batch'},
    'output': [0, 1]
}
```

Example:

```python
# you want this batch size to be variable
batch_size = 1
sample_input = torch.randn(batch_size, 3, 640, 640)

torch.onnx.export(
    model,
    sample_input,
    'sample.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```