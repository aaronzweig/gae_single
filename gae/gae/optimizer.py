import tensorflow as tf
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        # sample = tf.matmul(model.sample, tf.transpose(model.sample))
        # sample = tf.reshape(sample, [-1])
        # sample = tf.stop_gradient(sample)

        # self.cost = norm * tf.reduce_mean(sample * tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # self.cost *= 1.0 * num_nodes * num_nodes / tf.reduce_sum(sample)
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        lamb = tf.Variable(1000., name = 'scalar')
        self.cost += lamb
        self.lamb = lamb

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.log_lik = self.cost

        if FLAGS.vae:
            # self.kl = (0.5 / num_nodes) * tf.reduce_mean(model.sample * tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            # self.kl *= 1.0 * num_nodes / tf.reduce_sum(model.sample)
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            self.cost -= self.kl


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))