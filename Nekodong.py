import gym
from gym.spaces.discrete import Discrete
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools
from collections import deque


from DQN import DQN



class CardPlay( gym.Env ):

    class Player():
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self.reset()


        def randomChoice(self):
            return random.choice( self._deck )


        def strategy(self, t):
            if t == 0:
                return self._deck[0]
            elif t == 1:
                return self._deck[-1]
            elif t == 2:
                return self._deck[ len(self._deck)//2 ]
            # return random.choice( self._deck )

        def contains(self, x):
            # print( self._deck, x)
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
            return self._dumped+[-1]*(self._cardCount-len(self._dumped))



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
            r = [ (rewards[i], i) for i in range( len( rewards ) ) ]
            r.sort()

            return r[ len(rewards) -1 ][1]

            for (_, i) in reversed( r ):
                if self._player.contains( i ):
                    return i
                else:
                    # rewards[i-1] = -1
                    pass

            return i


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

        # self._stateMap = {}
        # # init self._stateMap
        # index = 0
        # self._stateMap[ None ] = index
        #
        # index+=1
        # for i in range(1, self.cardCount+1):
        #     p = itertools.permutations( range( self.cardCount ), i )
        #     q = itertools.permutations( range( self.cardCount ), i )
        #     for a, b in itertools.product( p, q ):
        #         self._stateMap[ (tuple(a), tuple(b)) ] = index
        #         # print( a, b )
        #         index+=1


        self.action_space = CardPlay.Action( self.cardCount )
        self.observation_space = CardPlay.Observation( self.cardCount*2 )

        self._opponent = CardPlay.Player( self.cardCount )
        self._reset()


    @property
    def Opponent(self):
        return self._opponent


    def _getState(self, a, b):
        # return self._stateMap[ (tuple(a), tuple(b)) ]
        return a+b

    def _step(self, action):

        if self.action_space.contains( action ):

            self.action_space.step( action )
            done = self.action_space.isDone()

            opponentCard = self._opponent.strategy ( self.strategy)

            # opponentCard = self._opponent.randomChoice()
            self._opponent.pop( opponentCard )

            mine = self.action_space.getDumped()
            opp = self._opponent.dumped

            reward = 0
            if done:
                score = 0
                draws = 0
                for me, opponent in zip( mine, opp ):

                    if me == 0 and opponent == self.cardCount - 1:
                        score += 1
                    elif me == self.cardCount - 1 and opponent == 0:
                        score += -1
                    else:
                        score += 1 if me > opponent else 0 if me == opponent else -1

                if score > 0:
                    reward = 1
                elif score == 0:
                    reward = 0.5
                else:
                    reward = -1


            state = self._getState( mine, opp )
            return state, reward, done, None
        else:
            mine = self.action_space.getDumped()
            opp = self._opponent.dumped
            state = self._getState(mine, opp)
            return state, -1, True, None

    def render(self, mode='human', close=False):
        pass


    def reset(self):
        return self._reset()


    def step(self, action ):
        return self._step( action )


    def _reset(self):
        self.strategy = random.randint( 0, 2 )
        self.action_space.reset()
        self._opponent.reset()
        self._state = self.action_space.getDumped() + self._opponent.dumped
        return self._state



def one_hot( s, max_samples = 16):
    # return np.identity( max_samples)[s:s+1]
    return [[ 1 if i == s else 0 for i in range(max_samples) ]]


def testRun():

    env = CardPlay()

    # env = gym.make( "FrozenLake-v0")
    input_size = env.observation_space.n
    output_size = env.action_space.n

    # X = tf.placeholder( shape=[1, input_size], dtype=tf.float32 )
    # W = tf.Variable( tf.random_uniform( [input_size, output_size], 0, 0.01 ) )
    # Qpred = tf.matmul( X, W )
    # Y = tf.placeholder( shape=[1, output_size], dtype=tf.float32 )
    # loss = tf.reduce_sum( tf.square( Y-Qpred))
    #
    # train = tf.train.GradientDescentOptimizer( learning_rate=0.1).minimize( loss )
    # discount = 0.99
    num_episodes = 10000
    rList = []

    # env.render()
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQUENCY = 5
    REPLAY_MEMORY = 50000

    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        sess.run( tf.global_variables_initializer() )

        # initial copy q_net -> target_net
        copy_ops = DQN.get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        step_count = 0

        for i in range( num_episodes ):
            s = env.reset()
            e = 1.0 / ((i/1000)+1)
            rAll = 0
            done = False
            local_loss = []


            while not done:
                # Qs = sess.run( Qpred, feed_dict={X:one_hot(s, input_size)} )
                if np.random.rand(1) >= e:
                    a = env.action_space.select(mainDQN.predict( s )[0])
                    # a = env.action_space.select( Qs[0] )
                else:
                    a = env.action_space.sample()

                # print( a, env.action_space._player._deck )
                s1, reward, done, _ = env.step( a )

                # Save the experience to our buffer
                replay_buffer.append((s, a, reward, s1, done))


                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = DQN.replay_train(mainDQN, targetDQN, minibatch)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                if done:
                    if reward == 1:
                        print( 'Win', env.action_space.getDumped(), env.Opponent.dumped)
                    elif reward == 0.5:
                        print( 'Even', env.action_space.getDumped(), env.Opponent.dumped)

                    else:
                        print( 'Lost', env.action_space.getDumped(), env.Opponent.dumped)
                    rList.append(reward)


                # if done:
                #     Qs[0, a] = reward
                #     if reward > 0:
                #         print( env.action_space.getDumped(), env.Opponent.dumped )
                # else:
                #     Qs1 = sess.run( Qpred, feed_dict={ X:one_hot(s1, input_size)})
                #     Qs[0, a] = reward + discount*np.max( Qs1 )
                #
                # sess.run( train, feed_dict={X: one_hot(s, input_size), Y:Qs} )
                #

                s = s1
                step_count += 1


        # print(W.eval(sess))
    plt.bar( range(len(rList)), rList, color="blue")
    plt.show()

    from itertools import groupby
    rList.sort()
    for key, group in groupby(rList):
        print( key, ":", str( len(list(group))/len(rList)) )






if __name__ == "__main__":
    testRun()