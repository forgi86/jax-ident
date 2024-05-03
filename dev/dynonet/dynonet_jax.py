import jax


a_coeff = jax.numpy.array([-.1, -0.5]) # a1, a2
b_coeff = jax.numpy.array([1.0, 0.0, 1.0]) # b0, b1, b2

u_carry = jax.numpy.array([0.0, 0.0]) # u-1, u-2
y_carry = jax.numpy.array([0.0, 0.0]) # y-1, y-2


ut = jax.numpy.array([1.0]) # u0

y_new = jax.numpy.dot(a_coeff, y_carry) + jax.numpy.dot(b_coeff, jax.numpy.concatenate((ut, u_carry)))