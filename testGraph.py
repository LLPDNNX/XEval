import tensorflow as tf
import os
import numpy
import ROOT
import time
import math
import random
import sys

    
inx = tf.placeholder('float',name='inx')
out = tf.add(inx*2,1,name="out")

sess = tf.Session()

const_graph = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    ["out"]
)
tf.train.write_graph(const_graph,"","testGraph.pb",as_text=False)

