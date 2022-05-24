""" Implementation of attention """

import tensorflow as tf
from tensorflow.contrib import slim



def CSpeA(net):

  batchsize, height, width, in_channels = net.get_shape().as_list()

  net_center = net[:,int(net.shape[1]/2),int(net.shape[2]/2),:]  
  
  net_center = tf.reshape(net_center, [tf.shape(net_center)[0], 1, 1, in_channels])     # (B,C) -> (B,1,C)   

  global_conv = slim.conv2d(net_center, in_channels, 1,  activation_fn=tf.nn.sigmoid, normalizer_fn=None)

  global_conv = tf.reshape(global_conv, [tf.shape(net)[0], 1, 1, in_channels])
  scale =  net * global_conv

  return scale


def get_drop_mask(attention, drop_thr):
  max_val = tf.reduce_max(attention, axis=[2], keep_dims=True)
  print('max_val ', max_val.shape)
  thr_val = max_val * drop_thr
  return tf.cast(attention > thr_val, dtype=tf.float32, name='drop_mask')

def RSpaA(net, depth, thr_value, embed=False, scope=None):
  batchsize, height, width, in_channels = net.get_shape().as_list()
  with tf.variable_scope(scope, 'attention', values=[net]) as sc:
    with slim.arg_scope([slim.conv2d], normalizer_fn=None):
      if embed:
        a = slim.conv2d(net, depth, 1, stride=1, activation_fn=tf.nn.relu, scope='embA')
        b = slim.conv2d(net, depth, 1, stride=1, activation_fn=tf.nn.relu, scope='embB')
      else:
        a, b = net, net
      #g_orig = g = slim.conv2d(net, depth, 1, stride=1, scope='g')
    g_orig = g = net

    # Flatten from (B,H,W,C) to (B,HW,C) or similar
    a_flat = tf.reshape(a, [tf.shape(a)[0], -1, tf.shape(a)[-1]])
    b_flat = tf.reshape(b, [tf.shape(b)[0], -1, tf.shape(b)[-1]])
    g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
    a_flat.set_shape([a.shape[0], a.shape[1] * a.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
    b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
    g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
    # Compute f(a, b) -> (B,HW,HW)
    f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
 
    # case 1 Non-local
    # f = tf.nn.softmax(f)
    # fg = tf.matmul(f, g_flat)

    # case 2 my spa
    binary_mask = get_drop_mask(f, thr_value)
    print('binary_mask', binary_mask.shape)
    www = tf.Variable(initial_value=[-100], trainable=False, name="a", dtype=tf.float32)
    f = f* binary_mask  + (1-binary_mask)*www
    f_relu_mask_softmax = tf.nn.softmax(f)

    fg = tf.matmul(f_relu_mask_softmax, g_flat)
    fg = tf.reshape(fg, [tf.shape(net)[0], height, width, in_channels])


    return slim.utils.collect_named_outputs(None, sc.name,  fg)
