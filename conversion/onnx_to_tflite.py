import tensorflow as tf
from onnx_tf.backend import prepare
import onnx
import torch
import torch.nn as nn
from conversion.utils import Conv, Hardswish

pt_model_path = 'weights/yolov5s.pt'
onnx_model_path = 'weights/yolov5s.onnx'
tf_model_path = 'weights/yolov5s.pb'
tf_lite_model_path = 'weights/yolov5s.tflite'

img_size = (640, 640)
batch_size = 1

img = torch.zeros((batch_size, 3, *img_size))
model = torch.load(pt_model_path, map_location=torch.device('cpu'))['model'].float().fuse().eval()

for k, m in model.named_modules():
    m._non_persistent_buffers_set = set() 
    if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
        m.act = Hardswish()
    
model.model[-1].export = True
y = model(img)

torch.onnx.export(
    model,
    img, 
    onnx_model_path,
    verbose=False,
    opset_version=12,
    input_names=['images'],
    output_names=['output_0', 'output_1', 'output_2'],
    keep_initializers_as_inputs=True
)


onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _import_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_import_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs)
    )

with tf.io.gfile.GFile(tf_model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

frozen_func = wrap_frozen_graph(
    graph_def,
    inputs=["images:0"],  # ["images:0"]
    outputs=["output_0:0", "output_1:0", "output_2:0"]  # ["output_0:0", "output_1:0", "output_2:0"]
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tf_lite_model = converter.convert()

with open(tf_lite_model_path, 'wb') as f:
    f.write(tf_lite_model)