import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no more warning TF
import tensorflow as tf


a = tf.constant(2, name='a') # change the name of the graph in tensorboard
b = tf.constant(3, name='b') # change the name of the graph in tensorboard
x = tf.add(a,b, name='add') # change the name of the graph in tensorboard

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    
    print(sess.run(x))
writer.close()

#Run pythonprogram.py
#tensorboard --logdir="./graphs" --port 6006




#constant

a = tf.constant([2,2], name='a')
b = tf.constant([[0,1], [2,3]], name= 'b')
x = tf.multiply(a, b, name='mul')

with tf.Session() as sess:
    print(sess.run(x))
    
    
tf.zeros([2,3], tf.int32)
input_tensor = [[0,1],[2,3],[4,5]]
tf.zeros_like(input_tensor)

#tf.ones([2,3], tf.int32)
#tf.ones_like(input_tensor)

#fill with specific value
tf.fill([2,3], 8)

#constant as sequence
tf.lin_space(10.0,13.0,4)

tf.range(3,18,3)
tf.range(5)


#random generated constants
y = tf.random_normal([2,3])
z = tf.truncated_normal([2,3])
tf.random_uniform()
tf.random_shuffle()
tf.random_crop()
tf.multinomial()
tf.random_gamma()

with tf.Session() as sess:
    print(sess.run(y))
    print(sess.run(z))
    

#Variables and Constant

#create variables with tf.Variable
s = tf.Variable(2, name='scalar')
m = tf.Variable([[0,1],[2,3]], name='matrix')
W = tf.Variable(tf.zeros([784,10]))

#tf.constant is an op
#tf.Variable is a class with many ops
#tf.Variable holds several ops:
x = tf.Variable(...)

x.initializer #init op
x.value() #read op
x.assign(...) #write op
x.assign_add(...) #and more

#create variables with tf.get_variable
s = tf.get_variable('scalar', initializer=tf.constant(2))
m = tf.get_variable('matrix', initializer=tf.constant([[0,1],[2,3]]))
W = tf.get_variable('big_matrix', shape=(784,10), initializer=tf.zeros_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # the easiest way to inittializing all variables at once....
    print(sess.run(W)) #error if run alone, have to initialize variables



#eval() a variable
W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer) #initialize a single variable
    print(W.eval())
    

#tf.Variable.assign()
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())    
    
    #W.assign(100) creates an assing op, that op needs to be executed in a session to take effect

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval()) 
    



my_var = tf.Variable(2, name='my_var')
#assign an op
my_var_times_two = my_var.assign(2*my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two)) #it assign 2*my_var to my_var everytime my_var_times_two op is executed
    print(sess.run(my_var_times_two))
    



#assign_add() and assign_sub()
my_var = tf.Variable(10)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    
    #increment by 10
    print(sess.run(my_var.assign_add(10)))
    #decrement by 2
    print(sess.run(my_var.assign_sub(2)))
    
    
#each session maintain its own copy of variables
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))
print(sess2.run(W.assign_sub(2)))    





#Placeholder

a=tf.placeholder(tf.float32, shape=[3])
b=tf.constant([5,5,5], tf.float32)

c=a+b
a_value = [1,2,3]

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a:a_value}))    
    
    
    
#Dot product
a = tf.constant([10,20], name='a')
b = tf.constant([2,3], name='b')

with tf.Session() as sess:
    print(sess.run(tf.multiply(a,b)))
    print(sess.run(tf.tensordot(a,b,1)))



W = tf.Variable(tf.truncated_normal([700,100]))
U = tf.Variable(W * 2)

with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(U.initializer)
    print(sess.run(U))
    
    
    
#Interactivesession vs. Session
#Interactivesession make itself the default session so can call run(), eval() without explicitly call the session
#However, it is complicated when you have multiple sessions to run
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
print(c.eval())
sess.close()