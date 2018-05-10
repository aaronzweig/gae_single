import tensorflow as tf
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        if FLAGS.subsample:
            edge_count = tf.count_nonzero(labels_sub)
            edge_indices = tf.where(tf.not_equal(labels_sub, 0))
            # no_edge_indices = tf.where(tf.equal(labels_sub, 0))
            # no_edge_indices = tf.random_shuffle(no_edge_indices)[:edge_count]
            no_edge_indices = tf.random_uniform(tf.count_nonzero(labels_sub, keepdims = True), maxval = num_nodes*num_nodes, dtype=tf.int32)
            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.gather(preds_sub, edge_indices), targets=tf.gather(labels_sub, edge_indices), pos_weight=1))
            self.cost += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.gather(preds_sub, no_edge_indices), targets=tf.gather(labels_sub, no_edge_indices), pos_weight=1))
        else:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.log_lik = self.cost

        if FLAGS.vae:
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            self.cost -= self.kl


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerSiemens(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        # sample = tf.matmul(model.sample, tf.transpose(model.sample))
        # sample = tf.reshape(sample, [-1])
        # sample = tf.stop_gradient(sample)

        # self.cost = norm * tf.reduce_mean(sample * tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # self.cost *= 1.0 * num_nodes * num_nodes / tf.reduce_sum(sample)
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=1))

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