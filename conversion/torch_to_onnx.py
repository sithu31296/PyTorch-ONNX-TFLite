import torch
from torchvision.models import mobilenet_v2

img_size = (640, 640)
batch_size = 1
onnx_model_path = 'model.onnx'

model = mobilenet_v2()
model.eval()

sample_input = torch.rand((batch_size, 3, *img_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)