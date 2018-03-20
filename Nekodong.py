import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random



class CardPlayer( gym.Env ):


    class Action( gym.Space ):
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self._deck = [i+1 for i in range(self.cardCount)]


        def sample(self):
            return self._deck.pop( random.randrange( len( self._deck ) ) )


        def contains(self, x):
            return x in self._deck



    class Observation( gym.Space ):
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self._deck = [i+1 for i in range(self.cardCount)]

        def sample(self):
            return self._deck.pop(random.randrange(len(self._deck)))

        def contains(self, x):
            return x in self._deck


    def __init__(self):
        self.action_space = CardPlayer.Action()
        self.observation_space = CardPlayer.Observation()

    def _step(self, action):
        pass


    def _reset(self):
        pass

def one_hot( s, max_samples = 16):
    return np.identity( max_samples)[s:s+1]


def testRun():

    env = gym.make( "FrozenLake-v0")
    input_size = env.observation_space.n
    output_size = env.action_space.n

    X = tf.placeholder( shape=[1, input_size], dtype=tf.float32 )
    W = tf.Variable( tf.random_uniform( [input_size, output_size], 0, 0.01 ) )
    Qpred = tf.matmul( X, W )
    Y = tf.placeholder( shape=[1, output_size], dtype=tf.float32 )
    loss = tf.reduce_sum( tf.square( Y-Qpred))

    train = tf.train.GradientDescentOptimizer( learning_rate=0.1).minimize( loss )
    discount = 0.99
    num_episodes = 2000
    rList = []

    env.render()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run( init )
        for i in range( num_episodes ):
            s = env.reset()
            e = 1.0 / ((i/50)+10)
            rAll = 0
            done = False
            local_loss = []

            while not done:
                Qs = sess.run( Qpred, feed_dict={X:one_hot(s)} )
                if np.random.rand(1) >= e:
                    a = np.argmax( Qs )
                else:
                    a = env.action_space.sample()

                s1, reward, done, _ = env.step( a )
                if done:
                    Qs[0, a] = reward
                else:
                    Qs1 = sess.run( Qpred, feed_dict={ X:one_hot(s1)})
                    Qs[0, a] = reward + discount*np.max( Qs1 )

                sess.run( train, feed_dict={X: one_hot(s), Y:Qs} )

                rAll += reward
                s = s1
            rList.append( rAll )

        print(W.eval(sess))

    print( "P:"+str( sum(rList)/num_episodes))


    plt.bar( range(len(rList)), rList, color="blue")
    plt.show()





if __name__ == "__main__":
    testRun()