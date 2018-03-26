import gym
from gym.spaces.discrete import Discrete
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools
from collections import deque
from typing import List


class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network( h_size=input_size*4)

    def _build_network(self, h_size=16, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            net = self._X

            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.output_size)
            self._Qpred = net

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)



def replay_train(mainDQN: DQN, targetDQN: DQN, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    DISCOUNT_RATE = 0.99


    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder



class CardPlay( gym.Env ):

    class Player():
        def __init__(self, cardCount ):
            self._cardCount = cardCount
            self.reset()


        def randomChoice(self):
            return random.choice( self._deck )


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

            opponentCard = self._opponent.randomChoice()
            self._opponent.pop( opponentCard )

            if action == 0 and opponentCard == self.cardCount-1:
                score = 1
            elif action == self.cardCount-1 and opponentCard == 0:
                score = -1
            else:
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
            mine = self.action_space.getDumped()
            opp = self._opponent.dumped
            state = self._getState(mine, opp)
            return state, -1, True, None

    def render(self, mode='human', close=False):
        pass


    def _reset(self):
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
    num_episodes = 20000
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
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)


        for i in range( num_episodes ):
            s = env.reset()
            e = 1.0 / ((i/50)+10)
            rAll = 0
            done = False
            local_loss = []
            step_count = 0

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
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

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
                rAll += reward
                s = s1
                step_count += 1

            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break
            rList.append( rAll )

        # print(W.eval(sess))

    print( "P:"+str( sum(rList)/num_episodes))


    plt.bar( range(len(rList)), rList, color="blue")
    plt.show()





if __name__ == "__main__":
    testRun()