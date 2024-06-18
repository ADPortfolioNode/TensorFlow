import tensorflow as tf

a = tf.constant([2], name='constant_a')
b = tf.constant([3], name='constant_b')

print("a =", a)

# tf and autograph
a = tf.constant([2], name='constant_a')
b = tf.constant([3], name='constant_b')

print("a =", a)

tf.print(a.numpy()[0])

@tf.function
def add(a, b):
  c = tf.add(a, b)
  # c = a + b is also a way to define the sum of the terms
  print(c)
  return c

# In TensorFlow, all data is passed between operations in a computation graph, and these are passed in the form of Tensors, hence the name of TensorFlow.

# The word tensor from new latin means "that which stretches". It is a mathematical object that is named "tensor" because an early application of tensors was the study of materials stretching under tension. The contemporary meaning of tensors can be taken as multidimensional arrays.
# defining multidimensional arrays using TensorFlow

Scalar = tf.constant(2)
Vector = tf.constant([5, 6, 2])
Matrix = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Tensor = tf.constant([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[4, 5, 6], [5, 6, 7], [6, 7, 8]], [[7, 8, 9], [8, 9, 10], [9, 10, 11]]])

print("Scalar (1 entry):\n", Scalar, "\n")
print("Vector (3 entries):\n", Vector, "\n")
print("Matrix (3x3 entries):\n", Matrix, "\n")
print("Tensor (3x3x3 entries):\n", Tensor, "\n")

print("scalar shape", Scalar.shape)
print("Vector shape", Vector.shape)
print("Matrix shape", Matrix.shape)
print("Tensor shape", Tensor.shape)

Matrix_one = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Matrix_two = tf.constant([[2, 2, 2], [2, 2, 2], [2, 2, 2]])

@tf.function
def add():
  add_1_operation = tf.add(Matrix_one, Matrix_two)
  return add_1_operation

print("Defined using tensorflow function:")
add_1_operation = add()
print(add_1_operation)
print("Defined using normal expressions:")
add_2_operation = Matrix_one + Matrix_two
print(add_2_operation)

Matrix_one = tf.constant([[2, 3], [3, 4]])
Matrix_two = tf.constant([[2, 3], [3, 4]])

@tf.function
def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)

mul_operation = mathmul()

print("Defined using tensorflow function:")
print(mul_operation)

v = tf.Variable(0)

@tf.function
def increment_by_one(v):
  v = tf.add(v, 1)
  return v

for i in range(3):
  v = increment_by_one(v)
  print(v)

a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a, b)
d = tf.subtract(a, b)

# tf.constant, tf.matmul, tf.add, tf.nn.sigmoid are some of the operations in TensorFlow. These are like functions in python but operate directly over tensors and each one does a specific thing.

# Other operations can be easily found in: https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html
print('c =:', c)
print('d =:', d)
