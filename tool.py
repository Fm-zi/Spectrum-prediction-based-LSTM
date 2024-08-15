from tensorflow.python.framework import graph_util
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import keras.backend as K
def stats(graph):
    flops = tf.profiler.profile(graph,options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph,options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('Flops: {};    Traninable params : {}'.format(flops.total_float_ops/1000000000.0,params.total_parameters/100000))

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

