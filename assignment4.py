#Eager execution 

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

x = [[2.]] #no need placeholder
m = tf.matmul(x,x)

print(m)



x = tf.random_uniform([2,2])

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        print(x[i,j])
        
        

x = tf.constant([1.0,2.0,3.0])

assert type(x.numpy()) == np.ndarray
squared = np.square(x)

for i in x:
    print(i)




def square(x):
    return x**2

grad = tfe.gradients_function(square)

print(square(3.))
print(grad(3.))


x = tfe.Variable(2.0)
def loss(y):
    return (y-x**2)**2

grad = tfe.implicit_gradients(loss)

print(loss(7.))
print(grad(7.))

#APIs for computing gradients work
#tfe.gradients_function()
#tfe.value_and_gradients_function()
#tfe.implicit_gradients()
#tfe.implicit_value_and_gradients()