import tensorflow as tf


#Problem: create 2 sets of variables
x1 = tf.random_normal([50,100])
x2 = tf.random_normal([50,100])

def two_hidden_layers(x):
    w1 = tf.Variable(tf.random_normal([100,50]), name='h1_weights')
    b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
    h1 = tf.add(tf.matmul(x, w1),b1)
    
    w2 = tf.Variable(tf.random_normal([50,10]), name='h2_weights')
    b2 = tf.Variable(tf.zeros([10]), name='h2_biases')
    logits = tf.add(tf.matmul(h1,w2),b2)
    
    return logits

writer = tf.summary.FileWriter('./graphs/ass5', tf.get_default_graph())
logits1 = two_hidden_layers(x1)
logits2 = two_hidden_layers(x2)
writer.close()


import tensorflow as tf
x1 = tf.random_normal([200,100])
x2 = tf.random_normal([200,100])

def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable('h1_weights', [100,50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('h1_biases', [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.add(tf.matmul(x, w1),b1)
    
    assert h1.shape.as_list() == [200, 50]
    w2 = tf.get_variable('h2_weights', [50,10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('h2_biases', [10], initializer=tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(h1,w2),b2)
    
    return logits

writer = tf.summary.FileWriter('./graphs/ass5', tf.get_default_graph())
with tf.variable_scope('two_layers') as scope:
    
    logits1 = two_hidden_layers(x1)
    
    scope.reuse_variables()
    
    logits2 = two_hidden_layers(x2)
writer.close()
    


#Layer all up
import tensorflow as tf

x1 = tf.random_normal([200,100])
x2 = tf.random_normal([200,100])
def fully_connected(x, output_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')


with tf.variable_scope('two_layers') as scope:
    writer = tf.summary.FileWriter('./graphs/ass5', tf.get_default_graph())
    logits1 = two_hidden_layers(x1)
    logits2 = two_hidden_layers(x2)
    writer.close()
    
    
    
    
    
    
    

