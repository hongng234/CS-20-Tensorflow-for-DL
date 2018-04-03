import tensorflow as tf

#a = tf.add(3,5)

#sess = tf.Session()
#print(sess.run(a))
#sess.close()

x=2
y=3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op,mul_op)

with tf.Session() as sess:
    z, not_useless = sess.run([pow_op, useless])
    
    
    
#Distributed computation on specific CPU GPU
with tf.device('/gpu:0'):
    a= tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], name='a')
    b= tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], name='b')
    c= tf.multiply(a,b)
    
sess=tf.Session(config=tf.ConfigProto(log_device_placement = True))
print(sess.run(c))



#graph
g1 = tf.get_default_graph()
g2 = tf.Graph()

with g1.as_default():
    a = tf.constant(3)
    
    
with g2.as_default():
    b = tf.constant(5)
    
sess = tf.Session(graph=g1)
with tf.Session() as sess:
    print(sess.run(a))
    
    
    
