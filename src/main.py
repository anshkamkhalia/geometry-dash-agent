# trains the model in the environment

import numpy as np # for arrays 
import tensorflow as tf # deep learning
from environment import GDEnvironment  # load environment
from model import GeometryDashAgent, Attention
from tensorflow.keras.optimizers import Adam
import time

# initialize components
model = GeometryDashAgent()
env = GDEnvironment()

state = env.reset() # reset state

num_steps = 500  # how many steps to run

state = env.reset()
env.prev_frame = None  # reset previous frame for brightness logic

# optimizer 
optimizer = Adam(0.001)

# helper function for computing discounted rewards
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0

    # complicated math stuff
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted

# training step function
def train_step(model, states, actions, rewards):

    # convert list to tensor
    states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    rewards_tensor = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)

    # compute discounted rewards
    discounted_rewards = compute_discounted_rewards(rewards)

    # very very complicated math stuff
    with tf.GradientTape() as tape:

        logits = model(states_tensor)

        # compute log probs of the taken actions
        neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions_tensor, logits=logits
        )
        # multiply by discounted rewards -> policy gradient loss
        loss = tf.reduce_mean(neg_log_probs * discounted_rewards)

    # compute gradients and apply
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss.numpy()

num_episodes = 100  # number of episodes to train
gamma = 0.99        # discount factor for future rewards

for episode in range(num_episodes):
    state = env.reset()
    env.prev_frame = None

    states, actions, rewards = [], [], []  # buffers for this episode

    done = False
    step_num = 0

    while not done:
        # prepare state for the model
        state_tensor = tf.convert_to_tensor(state[None, ...], dtype=tf.float32)

        # forward pass
        action_logits = model(state_tensor)
        action_probs = tf.nn.softmax(action_logits[0])
        action = np.random.choice(4, p=action_probs.numpy())

        # take action in environment
        next_state, reward, done, info = env.step(action)
        print(f"\n\nreward: {reward}")
        print(f"\n\ninfo: {info}")

        # store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # update state
        state = next_state
        step_num += 1

        if info.get('death', False):
            print(f"Episode {episode}, Step {step_num}: Player is dead! Reward={reward}")
            time.sleep(0.01)

    # episode finished -> train model
    loss = train_step(model, states, actions, rewards)
    print(f"Episode {episode} finished. Loss={loss:.4f}, Total reward={sum(rewards)}")
    time.sleep(0.05)