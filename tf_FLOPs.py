"""Report the FLOPs of the model.

Builds the model and prints its total floating-point operation count, a proxy
for computational cost / inference latency.
"""
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph

from model.model import Build_model

# Per-GPU memory cap (MB) requested from TensorFlow; adjust to your hardware.
GPU_MEMORY_LIMIT = 3072


def get_flops(model):
    """Freeze the model to a graph and return its total floating-point op count."""
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

model = Build_model((256, 512, 3))
model.summary()
print("The FLOPs is: {}G".format(get_flops(model)/1e9), flush=True)
