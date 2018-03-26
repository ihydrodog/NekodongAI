import gym
from gym.spaces.discrete import Discrete
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools




class CardPlay( gym.Env ):

    class Player():
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self.reset()


        def randomChoice(self):
            return random.choice( self._deck )


        def contains(self, x):
            print( self._deck, x)
            return x in self._deck


        def reset(self):
            self._deck = [i for i in range( self._cardCount )]
            self._dumped = []


        def pop(self, x ):
            self._deck.remove(x)
            self._dumped.append( x )
            return x


        def empty(self):
            return len( self._deck ) == 0


        @property
        def dumped(self):
            return self._dumped



    class Action( Discrete ):
        def __init__(self, cardCount ):
            super( CardPlay.Action, self ).__init__( cardCount )
            self._cardCount = cardCount
            self._player = CardPlay.Player( cardCount )


        def reset(self):
            self._player.reset()


        def step(self, x):
            self._player.pop( x )


        def isDone(self):
            return self._player.empty()


        def getDumped(self):
            return self._player.dumped


        def select(self, rewards):
            r = [ (rewards[i], i) for i in range( self._cardCount ) ]
            r.sort()

            for (_, i) in reversed( r ):
                if self._player.contains( i ):
                    return i
                else:
                    rewards[i] = -1


        def sample(self):
            return self._player.randomChoice()


        def contains(self, x):
            return self._player.contains( x )



    class Observation( Discrete ):
        def __init__(self, stateCount ):
            self._stateCount = stateCount
            super( CardPlay.Observation, self ).__init__( stateCount )


    def __init__(self):
        super( CardPlay, self).__init__()
        self.cardCount = 5

        self._stateMap = {}
        # init self._stateMap
        index = 0
        self._stateMap[ None ] = index

        index+=1
        for i in range(1, self.cardCount+1):
            p = itertools.permutations( range( self.cardCount ), i )
            q = itertools.permutations( range( self.cardCount ), i )
            for a, b in itertools.product( p, q ):
                self._stateMap[ (tuple(a), tuple(b)) ] = index
                # print( a, b )
                index+=1


        self.action_space = CardPlay.Action( self.cardCount )
        self.observation_space = CardPlay.Observation( index )

        self._opponent = CardPlay.Player( self.cardCount )
        self._reset()


    def _getState(self, a, b):
        return self._stateMap[ (tuple(a), tuple(b)) ]


    def _step(self, action):

        if self.action_space.contains( action ):

            self.action_space.step( action )
            done = self.action_space.isDone()

            opponentCard = self._opponent.randomChoice()
            self._opponent.pop( opponentCard )

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
        else:
            return 0, -1, True, None

    def render(self, mode='human', close=False):
        pass


    def _reset(self):
        self.action_space.reset()
        self._state = 0
        self._opponent.reset()

        return self._state



def one_hot( s, max_samples = 16):
    # return np.identity( max_samples)[s:s+1]
    return [[ 1 if i == s else 0 for i in range(max_samples) ]]


def testRun():

    env = CardPlay()

    # env = gym.make( "FrozenLake-v0")
    input_size = env.observation_space.n
    output_size = env.action_space.n

    X = tf.placeholder( shape=[1, input_size], dtype=tf.float32 )
    W = tf.Variable( tf.random_uniform( [input_size, output_size], 0, 0.01 ) )
    Qpred = tf.matmul( X, W )
    Y = tf.placeholder( shape=[1, output_size], dtype=tf.float32 )
    loss = tf.reduce_sum( tf.square( Y-Qpred))

    train = tf.train.GradientDescentOptimizer( learning_rate=0.1).minimize( loss )
    discount = 0.99
    num_episodes = 10000
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
                Qs = sess.run( Qpred, feed_dict={X:one_hot(s, input_size)} )
                if np.random.rand(1) >= e:
                    a = env.action_space.select( Qs[0] )
                else:
                    a = env.action_space.sample()

                print( a, env.action_space._player._deck )
                s1, reward, done, _ = env.step( a )

                if done:
                    Qs[0, a] = reward
                else:
                    Qs1 = sess.run( Qpred, feed_dict={ X:one_hot(s1, input_size)})
                    Qs[0, a] = reward + discount*np.max( Qs1 )

                sess.run( train, feed_dict={X: one_hot(s, input_size), Y:Qs} )

                rAll += reward
                s = s1
            rList.append( rAll )

        print(W.eval(sess))

    print( "P:"+str( sum(rList)/num_episodes))


    plt.bar( range(len(rList)), rList, color="blue")
    plt.show()





if __name__ == "__main__":
    testRun()