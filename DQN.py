import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple

# Hyper Parameters:
GAMMA = 0.99                        # decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0               # 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1                 # 0.001 # final value of epsilon
EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 900000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'
checkpoint_path = 'checkpoints/model'

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]


class ObservationProcessor():
    """
    Processes a raw Atari image. Resizes it and converts it to grayscale.
    """

    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(
                shape=[210, 160, 3], dtype=tf.uint8)              # input image
            self.output = tf.image.rgb_to_grayscale(
                self.input_state)                           # rgb to grayscale
            self.output = tf.image.crop_to_bounding_box(
                self.output, 34, 0, 160, 160)           # crop image
            self.output = tf.image.resize_images(                                               # resize image
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # remove rgb dimension
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, {self.input_state: state})


class DQN():
    # Define the following things about Deep Q Network here:
    #   1. Network Structure (Check lab spec for details)
    #       * tf.contrib.layers.conv2d()
    #       * tf.contrib.layers.flatten()
    #       * tf.contrib.layers.fully_connected()
    #       * You may need to use tf.variable_scope in order to set different variable names for 2 Q-networks
    #   2. Target value & loss
    #   3. Network optimizer: tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    #   4. Training operation for tensorflow

    ''' You may need 3 placeholders for input: 4 input images, target Q value, action index
    def _build_network(self):
        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
    '''

    def __init__(self, scope_name):
        self.scope_name = scope_name

    def inference(self, state, trainable=True):
        """
        Args:
            state:
            network_type:
        """
        if self.scope_name is 'TARGET':
            trainable = False
        elif self.scope_name is 'BEHAVIOR':
            trainable = True

        with tf.variable_scope(self.scope_name) as scope:
            print self.scope_name
            state_float = tf.to_float(state)
            conv1 = tf.contrib.layers.conv2d(inputs=state_float, num_outputs=32, stride=[4, 4],
                                             kernel_size=[8, 8], padding='SAME', activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                biases_initializer=tf.random_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                trainable=trainable, scope='conv1')
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, stride=[2, 2],
                                             kernel_size=[4, 4], padding='SAME', activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                biases_initializer=tf.random_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                trainable=trainable, scope='conv2')
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, stride=[1, 1],
                                             kernel_size=[3, 3], padding='SAME', activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                biases_initializer=tf.random_normal_initializer(
                stddev=0.01, dtype=tf.float32),
                trainable=trainable, scope='conv3')
            flatten4 = tf.contrib.layers.flatten(
                inputs=conv3, scope='flatten4')
            connected5 = tf.contrib.layers.fully_connected(inputs=flatten4, num_outputs=512, activation_fn=tf.nn.relu,
                                                           weights_initializer=tf.truncated_normal_initializer(
                                                               stddev=0.01, dtype=tf.float32),
                                                           biases_initializer=tf.random_normal_initializer(
                                                               stddev=0.01,
                                                               dtype=tf.float32),
                                                           trainable=trainable, scope='connected5')
            connected6 = tf.contrib.layers.fully_connected(inputs=connected5, num_outputs=4, activation_fn=None,
                                                           weights_initializer=tf.truncated_normal_initializer(
                                                               stddev=0.01, dtype=tf.float32),
                                                           biases_initializer=tf.random_normal_initializer(
                                                               stddev=0.01,
                                                               dtype=tf.float32),
                                                           trainable=trainable, scope='connected6')
        return connected6

    def loss(self, logits, labels, actions):
        """
        Args:
            logits: behavior_q_values, [batch_size, target_q_value, 4]
            labels:
            actions
        """
        print logits
        # [32], [0, 2, 3, ..., BATCH_SIZE-1]
        indeices = tf.constant([i for i in range(BATCH_SIZE)])
        # [32, 2], [[0, actions[0]], [1, actions[1]], ..., [BATCH_SIZE-1, actions[BATCH_SIZE-1]]
        indeices = tf.stack([indeices, actions], axis=1)
        # [32], [logits[actions[0]], logits[actions[1]], ..., logits[actions[BATCH_SIZE-1]]]
        behavior_q_value = tf.gather_nd(logits, indeices)
        # total_loss = tf.reduce_sum(
        #     tf.square(tf.subtract(labels, behavior_q_value)))
        total_loss = tf.reduce_mean(
            tf.squared_difference(labels, behavior_q_value))

        return total_loss

    def train(self, total_loss, global_step=None):
        """
        Args:
            total_loss:
            global_step:
        """
        train_op = tf.train.RMSPropOptimizer(
            0.00025, 0.99, 0.0, 1e-6).minimize(total_loss, global_step=global_step)
        return train_op


def update_target_network(sess, behavior_Q, target_Q):
    # copy weights from behavior Q-network to target Q-network
    # Hint:
    #   * tf.trainable_variables()                  https://www.tensorflow.org/api_docs/python/tf/trainable_variables
    #   * variable.name.startswith(scope_name)      https://docs.python.org/3/library/stdtypes.html#str.startswith
    #   * assign                                    https://www.tensorflow.org/api_docs/python/tf/assign
    global_V = tf.global_variables()
    behavior_Q_var_list = []
    target_Q_var_list = []
    for _, v in enumerate(global_V):
        if v.name.startswith(behavior_Q.scope_name):
            behavior_Q_var_list.append(v)
        elif v.name.startswith(target_Q.scope_name):
            target_Q_var_list.append(v)

    update_ops = []
    for i, v in enumerate(target_Q_var_list):
        # print v
        # print behavior_Q_var_list[i]
        assign_op = v.assign(behavior_Q_var_list[i])
        update_ops.append(assign_op)

    sess.run(update_ops)


def main(_):

    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # Define Transition tuple
    Transition = namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # create a observation processor
    ob_proc = ObservationProcessor()

    # Placeholders for our input
    # Our input are 4 grayscale frames of shape 84, 84 each
    X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.int8, name="X")
    # The TD target value
    y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    # Integer id of which action was selected
    actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    # Behavior Network & Target Network
    behavior_Q = DQN(scope_name='BEHAVIOR')
    target_Q = DQN(scope_name='TARGET')
    behavior_Q_inference = behavior_Q.inference(X_pl)
    target_Q_inference = target_Q.inference(X_pl)

    loss_op = target_Q.loss(behavior_Q_inference, y_pl, actions_pl)
    train_op = behavior_Q.train(loss_op)

    summary_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    # # test trained model
    # with tf.Session() as sess:
    #     sess.run(init)

    #     saver = tf.train.Saver()
    #     # Load a previous checkpoint if we find one
    #     latest_checkpoint = tf.train.latest_checkpoint('checkpoints')
    #     if latest_checkpoint:
    #         print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    #         saver.restore(sess, latest_checkpoint)

    #     # record videos
    #     record_video_every = 1
    #     env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count %
    #                   record_video_every == 0, resume=True)

    #     # Reset the environment
    #     observation = env.reset()
    #     observation = ob_proc.process(sess, observation)
    #     state = np.stack([observation] * 4, axis=2)
    #     episode_reward = 0                              # store the episode reward
    #     '''
    #     How to update episode reward:
    #     next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
    #     episode_reward += reward
    #     '''

    #     for t in itertools.count():
    #         # choose a action
    #         # state -> [state]: [84, 84, 4] -> [1, 84, 84, 4] == [?,
    #         # 84, 84, 4]
    #         behavior_Q_inference_value = sess.run(
    #             behavior_Q_inference, feed_dict={X_pl: [state]})
    #         # [1, 4] -> [4]
    #         behavior_Q_inference_value = np.array(
    #             behavior_Q_inference_value[0])
    #         action = np.argmax(behavior_Q_inference_value, axis=0)

    #         # execute the action
    #         next_observation, reward, done, _ = env.step(
    #             VALID_ACTIONS[action])
    #         episode_reward += reward
    #         next_observation = ob_proc.process(sess, next_observation)
    #         next_state = np.append(
    #             state[:, :, 1:], np.expand_dims(next_observation, 2), axis=2)

    #         if done:
    #             print "total steps: ", t, " Episode reward: ", episode_reward
    #             break

    #         state = next_state

    # tensorflow session
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()

        # Populate the replay buffer
        observation = env.reset()                       # retrive first env image
        observation = ob_proc.process(
            sess, observation)        # process the image
        # stack the image 4 times
        state = np.stack([observation] * 4, axis=2)
        while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
            '''
            *** This part is just pseudo code ***

            action = None
            if random.random() <= epsilon
                action = random_action
            else
                action = DQN_action
            '''
            action = random.randint(0, 3)

            next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_observation = ob_proc.process(sess, next_observation)
            next_state = np.append(
                state[:, :, 1:], np.expand_dims(next_observation, 2), axis=2)
            replay_memory.append(Transition(
                state, action, reward, next_state, done))

            # Current game episode is over
            if done:
                observation = env.reset()
                observation = ob_proc.process(sess, observation)
                state = np.stack([observation] * 4, axis=2)

            # Not over yet
            else:
                state = next_state

        # record videos
        record_video_every = 100
        env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count %
                      record_video_every == 0, resume=True)

        # total steps
        total_t = 0
        highest_reward = 0
        # global_step = tf.train.get_or_create_global_step

        for episode in range(TRAINING_EPISODES):

            # Reset the environment
            observation = env.reset()
            observation = ob_proc.process(sess, observation)
            state = np.stack([observation] * 4, axis=2)
            episode_reward = 0                              # store the episode reward
            '''
            How to update episode reward:
            next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
            episode_reward += reward
            '''

            for t in itertools.count():

                # choose a action
                action = None
                if total_t > 500000:
                    epsilon = FINAL_EPSILON
                else:
                    epsilon = INITIAL_EPSILON - \
                        (INITIAL_EPSILON - FINAL_EPSILON) * total_t / 500000.0

                if random.random() <= epsilon:
                    action = random.randint(0, 3)
                else:
                    # state -> [state]: [84, 84, 4] -> [1, 84, 84, 4] == [?,
                    # 84, 84, 4]
                    behavior_Q_inference_value = sess.run(
                        behavior_Q_inference, feed_dict={X_pl: [state]})
                    # [1, 4] -> [4]
                    behavior_Q_inference_value = np.array(
                        behavior_Q_inference_value[0])
                    action = np.argmax(behavior_Q_inference_value, axis=0)
                # execute the action
                next_observation, reward, done, _ = env.step(
                    VALID_ACTIONS[action])
                episode_reward += reward

                if done:
                    if episode_reward > highest_reward:
                        # Save the current checkpoint
                        highest_reward = episode_reward
                        saver.save(sess, checkpoint_path + "_episode_" +
                                   str(episode) + "_reward_" + str(int(highest_reward)))
                    if (episode + 1) % 1000 == 0:
                        # Save the current checkpoint
                        saver.save(sess, checkpoint_path + "_episode_" +
                                   str(episode) + "_reward_" + str(int(episode_reward)))

                next_observation = ob_proc.process(sess, next_observation)
                next_state = np.append(
                    state[:, :, 1:], np.expand_dims(next_observation, 2), axis=2)

                # save the transition to replay buffer
                replay_memory.append(Transition(
                    state, action, reward, next_state, done))

                # if the size of replay buffer is too big, remove the oldest one.
                # Hint: replay_memory.pop(0)
                while len(replay_memory) > REPLAY_MEMORY_SIZE:
                    replay_memory.pop(0)

                # sample a minibatch from replay buffer. Hint: samples =
                # random.sample(replay_memory, batch_size)
                samples = random.sample(replay_memory, BATCH_SIZE)
                minibatch_state = []
                minibatch_reward = []
                minibatch_action = []
                minibatch_next_state = []
                minibatch_done = []
                for _, v in enumerate(samples):
                    minibatch_state.append(v.state)
                    minibatch_reward.append(v.reward)
                    minibatch_action.append(v.action)
                    minibatch_next_state.append(v.next_state)
                    minibatch_done.append(v.done)

                # calculate target Q values by target network
                target_Q_next_value, behavior_Q_next_value = sess.run([target_Q_inference, behavior_Q_inference], feed_dict={
                    X_pl: minibatch_next_state})
                best_actions = np.argmax(behavior_Q_next_value, axis=1)
                target_Q_next_value = np.array(target_Q_next_value)

                y = minibatch_reward + np.invert(minibatch_done).astype(np.float32) * GAMMA * \
                    target_Q_next_value[np.arange(BATCH_SIZE), best_actions]

                # y = minibatch_reward + GAMMA * \
                #     np.amax(target_Q_next_value, axis=1)  # [32, 4] -> [32]

                # Update network
                feed_dict = {X_pl: minibatch_state,
                             y_pl: y,
                             actions_pl: minibatch_action}
                _, total_loss = sess.run(
                    [train_op, loss_op], feed_dict=feed_dict)

                # Update target network every FREQ_UPDATE_TARGET_Q steps
                if total_t % FREQ_UPDATE_TARGET_Q == 0:
                    update_target_network(sess, behavior_Q, target_Q)

                if done:

                    episode_summary = tf.Summary()
                    episode_summary.value.add(
                        simple_value=epsilon, tag="epsilon")
                    episode_summary.value.add(
                        simple_value=episode_reward, tag="episode_reward")
                    episode_summary.value.add(
                        simple_value=total_loss, tag="total_loss")
                    summary_writer.add_summary(episode_summary, episode)
                    summary_writer.flush()

                    print "Episode number: ", episode, "total steps: ", t, " Episode reward: ", episode_reward, " batch loss: ", total_loss
                    break

                state = next_state
                total_t += 1


if __name__ == '__main__':
    tf.app.run()
