from argparse import ArgumentParser
import os

import tensorflow as tf
from tqdm import tqdm

class SimpleModel(object):
    def __init__(self, sess, log=False):
        self.sess = sess
        self.log = log

        self._define_placeholders()
        self._define_weights()

        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                           logits=self.forward(self.x))
        self.loss = tf.reduce_mean(xentropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.forward(self.x), 1)), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
        self.test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    def __call__(self, x):
        return self.forward(self, x)

    def _define_placeholders(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=(None, 28*28), name='in')
        self.y = tf.placeholder(tf.float32, shape=(None, 10), name='out')


    def _define_weights(self):
        with tf.name_scope('layer1'):
            self.w1 = tf.Variable(tf.truncated_normal([28*28, 512], stddev=0.1), name='weights1')
            self.b1 = tf.Variable(tf.truncated_normal([512], stddev=0.1), name='bias1')

        with tf.name_scope('layer2'):
            self.w2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1), name='weights2')
            self.b2 = tf.Variable(tf.truncated_normal([256], stddev=0.1), name='bias2')

        with tf.name_scope('layer3'):
            self.w3 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), name='weights3')
            self.b3 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name='bias3')

    def forward(self, x):
        a1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        a2 = tf.nn.relu(tf.matmul(a1, self.w2) + self.b2)
        logit = tf.matmul(a2, self.w3) + self.b3
        return logit

    def train(self, epochs=5):
        current_epoch = 0
        i = 0
        while mnist.train.epochs_completed < epochs:
            i += 1
            x_, y_ = mnist.train.next_batch(32)
            summ, _ = self.sess.run([self.summary, self.optimizer], {self.x:x_, self.y:y_})
            if i % 100 == 0:
                valid_dict = {
                    self.x:mnist.validation.images,
                    self.y:mnist.validation.labels
                }
                summ, validation_acc = self.sess.run([self.summary, self.accuracy], valid_dict)
                self.train_writer.add_summary(summ, i)
            if current_epoch != mnist.train.epochs_completed:
                current_epoch += 1
                print(f'completed epoch {current_epoch}')

    def test(self):
        # Let's see how well we did
        return self.sess.run(self.accuracy, {self.x:mnist.test.images, self.y:mnist.test.labels})


def main(argv):
    sess = tf.Session()
    model = SimpleModel(sess, log=True)
    writer = tf.summary.FileWriter(f'{FLAGS.log_dir}/graph', graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if not os.path.exists('/tmp/model.ckpt') or FLAGS.retrain:
        model.train()
        saver.save(sess, '/tmp/model.ckpt')
    else:
        saver.restore(sess, '/tmp/model.ckpt')
    print(f'Final test accuracy of {model.test() * 100 :.2f}')
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--retrain', action='store_true',
                        help='Retrain the model')
    parser.add_argument('--data-dir', default='/tmp/mnist',
                        help='Directory for the MNIST dataset')
    parser.add_argument('--log-dir', default='/tmp/tf-log/',
                        help='Directory for TensorBoard logs')
    FLAGS, _ = parser.parse_known_args()
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(FLAGS.data_dir, one_hot=True)
    tf.app.run()

