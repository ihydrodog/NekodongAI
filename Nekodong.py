import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math



class CardPlay( gym.Env ):

    class Player():
        def __init__(self, cardCount ):
            self.reset()


        def randomChoice(self):
            return self.pop( self._deck[ random.randrange(len(self._deck))] )


        def contains(self, x):
            return x in self._deck


        def reset(self):
            self._deck = [i+1 for i in range(self.cardCount)]
            self._dumped = []


        def pop(self, x ):
            self._deck.pop(x)
            self._dumped.append( x )


        def empty(self):
            return len( self._deck ) == 0


        @property
        def dumped(self):
            return self._dumped




    class Action( gym.Space ):
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self._player = CardPlay.Player( cardCount )


        def sample(self):
            return self._player.randomChoice()


        def contains(self, x):
            return self._player.contains( x )


        def reset(self):
            self._player.reset()


        def step(self, x):
            self._player.pop( x )


        def isDone(self):
            return self.empty()


        def getDumped(self):
            return self._player.dumped



    class Observation( gym.Space ):
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self._max = math.pow( math.factorial( self._cardCount ), 2)


        def sample(self):
            return random.randrange( self._max )

        def contains(self, x):
            return 0 <= x < self._max



    def __init__(self):
        self.cardCount = 5
        self.action_space = CardPlay.Action( self.cardCount )
        self.observation_space = CardPlay.Observation()

        self._opponent = CardPlay.Player( self.cardCount )
        self._reset()


    def _getState(self, a, b):
        for _a, _b in zip( a, b):



    def _step(self, action):
        self.action_space.step( action )
        done = self.action_space.isDone()

        opponentCard = self._opponent.randomChoice()

        score = 1 if action > opponentCard else 0 if action == opponentCard else -1

        mine = self.action_space.getDumped()
        opp = self._opponent.dumped

        reward = 0
        if done:
            wins = 0
            draws = 0
            for me, opponent in zip( mine, opp ):
                if me > opponent:
                    wins += 1
                elif me == opponent:
                    draws += 1
            if wins > self.cardCount//2:
                reward = 1


        state = self._getState( mine, opp )
        return state, reward, done, None




    def _reset(self):
        self.action_space.reset()
        self._state = 0
        self._opponent.reset()


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