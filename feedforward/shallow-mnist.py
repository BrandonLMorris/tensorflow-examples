from argparse import ArgumentParser
import os

import tensorflow as tf
from tqdm import tqdm

class SimpleModel(object):
    def __init__(self, sess, log=False):
        self.sess = sess
        self.log = log
        self.x = tf.placeholder(tf.float32, shape=(None, 28*28))
        self.y = tf.placeholder(tf.float32, shape=(None, 10))

        self.w1 = tf.Variable(tf.truncated_normal([28*28, 512], stddev=0.1))
        self.b1 = tf.Variable(tf.truncated_normal([512], stddev=0.1))

        self.w2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1))
        self.b2 = tf.Variable(tf.truncated_normal([256], stddev=0.1))

        self.w3 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
        self.b3 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.forward(self.x))
        self.loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def __call__(self, x):
        return self.forward(self, x)

    def _accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.forward(self.x), 1)), tf.float32))

    def forward(self, x):
        a1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        a2 = tf.nn.relu(tf.matmul(a1, self.w2) + self.b2)
        logit = tf.matmul(a2, self.w3) + self.b3
        return logit

    def train(self, epochs=5):
        current_epoch = 0
        pbar = tqdm(total=epochs)
        while mnist.train.epochs_completed < epochs:
            x_, y_ = mnist.train.next_batch(32)
            self.optimizer.run(self.sess, {self.x:x_, self.y:y_})
            if current_epoch != mnist.train.epochs_completed:
                current_epoch += 1
                pbar.update(1)
                if self.log:
                    print(f'completed epoch {current_epoch}')
                    validation_acc = self._accuracy().eval(self.sess, {self.x:mnist.validation.images, self.y:mnist.validation.labels})
                    print(f'validation error: {validation_acc * 100 :.2f}')
        pbar.close()

    def test(self):
        # Let's see how well we did
        return self.sess.run(self._accuracy(), {self.x:mnist.test.images, self.y:mnist.test.labels})


def main(argv):
    sess = tf.Session()
    model = SimpleModel(sess, log=True)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.data_dir) or FLAGS.retrain:
        model.train()
        saver.save(sess, '/tmp/model.ckpt')
    else:
        saver.restore(sess, '/tmp/model.ckpt')
    print(f'Final test accuracy of {model.test() * 100 :.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--retrain', action='store_true',
                        help='Retrain the model')
    parser.add_argument('--data-dir', '/tmp/mnist',
                        help='Directory for the MNIST dataset')
    FLAGS, _ = parser.parse_known_args()
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(FLAGS.data_dir, one_hot=True)
    tf.app.run()

