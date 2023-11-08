import tensorflow as tf
import numpy as np
import time



def jvp(y, x, v):
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class Meta(tf.keras.Model):
    def __init__(self, num_tasks=1000, dim=100, eps=0, dtype=tf.float32, name="meta"):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation=tf.tanh, dtype=dtype),
                tf.keras.layers.Dense(dim, activation=tf.tanh, dtype=dtype),
                tf.keras.layers.Dense(dim, activation=tf.tanh, dtype=dtype),
            ]
        )
        self.dim = dim
        self.eps = eps
        self.N = num_tasks
        self.head = tf.Variable(
            0.05 * tf.random.normal(shape=[dim, self.N], dtype=dtype), dtype=dtype
        )
        self._dtype = dtype
        self._name = name

        self.shared_nn.build(input_shape=[None, 2])
        # optimizer for PINN is not needed in the downstream task
        # self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3) 

    def call(self, x, y, head):
        basis = self.basis(x, y)
        return tf.matmul(basis, head)

    def basis(self, x, y):
        shared = self.shared_nn.call(tf.concat([x, y], axis=-1))
        # shared = tf.concat([shared, tf.ones(shape=[shared.shape[0], 1])], axis=-1)
        basis = x * (1 - x) * y * (1 - y) * shared
        return basis

    @tf.function
    def pde2(self, x, y):
        basis = self.basis(x, y)
        v = tf.ones_like(x)
        basis_x = jvp(basis, x, v)[0]
        basis_xx = jvp(basis_x, x, v)[0]
        v = tf.ones_like(y)
        basis_y = jvp(basis, y, v)[0]
        basis_yy = jvp(basis_y, y, v)[0]
        return basis, basis_xx, basis_yy

    def loss_function(self, x, y, f):
        u = self.call(x, y, self.head)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return tf.reduce_mean((u_xx + u_yy - f) ** 2)

    @tf.function
    def pde(self, x, y, head):
        u = self.call(x, y, head)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return u_xx + u_yy

    @tf.function
    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            regularization = tf.reduce_mean(self.head**2)
            loss = self.loss_function(x, y, f)
            total_loss = loss + self.eps * regularization
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss, loss

    def train(self, x_train, y_train, f_train, niter=10000, ftol=5e-5):
        x_train = tf.constant(x_train, self.dtype)
        y_train = tf.constant(y_train, self.dtype)
        f_train = tf.constant(f_train, self.dtype)

        train_op = self.train_op
        loss_op = tf.function(lambda: self.loss_function(x_train, y_train, f_train))
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            total_loss, loss_value = train_op(x_train, y_train, f_train)
            loss += [loss_value.numpy()]

            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss, ", time: ", time.time() - t0)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/" + self.name,
                        overwrite=True,
                    )
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)
