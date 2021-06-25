import tensorflow as tf
import numpy as np
import gym

from replayMemory import replayMemory
from DQN import DQN

# USED TO INCREASE RENDEERING WINDOW ON RETINA SCREENS

# SET UP ENVIRONMENT
game = 'PongNoFrameskip-v4'
# game = 'BreakoutNoFrameskip-v4'
# breakout:400, spaceInvaders:1976, frostbite:328 #maybe also some different game not atari?
env = gym.make(game)

# Name used to save the data
name = "DQN_" + game

# SETUP LISTS FOR DEBUGGING
episodes_reward = []
episodes_qvalue = []
global_step = 0
episode = 0
bestResult = -1e5
lastResult = 0

# HYPERPARAMETERS
state_size = env.observation_space.shape  # return list 3 elements [width, height, channels(RGB or B/W)]
# print(env.unwrapped.get_action_meanings())
num_actions = env.action_space.n

syncTarget = 1000  # sync target network with prediction network every timesteps #PAPER:10000

# record every 50 episodes
# env = wrappers.Monitor(env, "recording/" + name, video_callable=lambda episode_id: episode_id % 50 == 0)
# env = wrappers.Monitor(env, "recording/" + name, video_callable=False)

initializeReplayBuffer = 32  # experience 10000 timesteps before start training #PAPER:50000
repBufferSize = 10000  # PAPER: 1000000
sampleSize = 32  # PAPER:32
startingEpsilon = 1.0  # Starting epsilon #PAPER:1.0
endEpsilon = 0.01  # PAPER: 0.1
epsilonDecay = 10 ** 5  # Global steps required for decay epsilon from start to end #PAPER:10**6
learn_rate = 0.0001  # PAPER: 0.00025

factor = (endEpsilon - startingEpsilon) / epsilonDecay
frameskip = 4

# Shape of the frame before feeding it
width = 84
height = 84


def startTraining():
    global episode
    global global_step
    global lastResult
    global bestResult

    print("\n\n FILLING REPLAY BUFFER... \n\n")
    initializationExperiences = 0
    while (initializationExperiences < initializeReplayBuffer):
        env.reset()
        for i in range(60):
            f, _, _, _ = env.step(0)

        state = DQN.inputPreprocess(f)
        d = False

        old_r = 0
        good = bad = 0
        while not (d):
            a = np.random.randint(0, num_actions)

            for i in range(frameskip):
                f1, _, _, _ = env.step(a)

            newState = DQN.inputPreprocess(f1)
            new_r = DQN.get_reward(newState)
            if new_r == old_r:
                real_r = 0
            else:
                real_r = new_r

            if real_r == 1:
                good += 1
            elif real_r == -1:
                bad += 1
            if good > 15 or bad > 15:
                d = True

            memory.addExperience(state, a, real_r, d, newState)
            state = newState
            old_r = new_r

            initializationExperiences += 1

    print('memory  size:', memory.get_memory_size())
    print("\n\n STARTING TRAINING.. \n\n")
    while (lastResult < 14.5):
        episode_reward = 0
        episode_qvalues = []

        recording_obs = []
        recording_action = []
        recording_obs_ = []
        recording_r = []
        recording_done = []

        env.reset()
        for i in range(60):
            f, _, _, _ = env.step(0)

        DQN.resetObservationState()

        state = DQN.inputPreprocess(f)
        d = False

        old_r = 0
        good = bad = 0
        while not (d):
            a, qvalue = DQN.actionSelection(state)

            for i in range(frameskip):
                f1, _, _, _ = env.step(a)

            newState = DQN.inputPreprocess(f1)
            new_r = DQN.get_reward(newState)
            if new_r == old_r:
                real_r = 0
            else:
                real_r = new_r

            if real_r == 1:
                good += 1
            elif real_r == -1:
                bad += 1
            if good > 15 or bad > 15:
                d = True

            old_r = new_r

            memory.addExperience(state, a, real_r, d, newState)

            recording_obs.append(state)
            recording_action.append(a)
            recording_obs_.append(newState)
            recording_r.append(real_r)
            recording_done.append(d)

            if (global_step % 4 == 0):
                DQN.training(memory.sampleExperience())

            # epsilon annealing from 1 to 0.1 in 1000000 steps
            if global_step <= epsilonDecay:
                DQN.epsilon = (factor * global_step) + startingEpsilon
            else:
                DQN.epsilon = endEpsilon

            # Update Target network every 10000 TRAINING STEPS (40000 steps)
            if (global_step % (4 * syncTarget) == 0 and global_step != 0):
                print("\n\nGlobal step", global_step, "Updating target network..\n\n")
                DQN.updateTargetNetwork()

            # Every 50k steps save filters values
            if global_step % 50000 == 0:
                summ = DQN.sess.run(DQN.mergeFilters)

                DQN.writeOps.add_summary(summ, global_step=global_step)

            state = newState
            global_step += 1
            episode_reward += real_r
            episode_qvalues.append(qvalue)

        recording_obs = np.array(recording_obs)
        recording_action = np.array(recording_action)
        recording_obs_ = np.array(recording_obs_)
        recording_r = np.array(recording_r)
        recording_done = np.array(recording_done)
        filename = "CGANModel/data/pong/model_" + str(episode) + ".npz"
        if episode % 10 == 0:
            np.savez_compressed(filename, obs=recording_obs, action=recording_action, r=recording_r,
                                obs_=recording_obs_, d=recording_done)

        # EPISODE ENDED
        print("\nEnded episode:", episode, "Global step:", global_step, "episode_reward:", episode_reward, "\n")

        # Store the averaged Q_value of the episode
        avgQVal = np.mean(episode_qvalues)
        # Store the total reward of the episode
        episodes_reward.append(episode_reward)
        # Compute the average total reward of the last 100 episodes
        lastResult = np.mean(episodes_reward[-100:])

        # Feed the averaged reward to the agent method in order to show the stats on tensorboard
        summ = DQN.sess.run(DQN.mergeEpisodeData, feed_dict={DQN.averagedReward: lastResult,
                                                             DQN.PHEpsilon: DQN.epsilon,
                                                             DQN.avgQValue: avgQVal})
        # Save the stats for tensorboard
        DQN.writeOps.add_summary(summ, global_step=episode)

        if lastResult > bestResult:
            print("\n")
            print("Saving model..")
            print("\n")
            DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode,
                                   rewards=episodes_reward[-100:])

            bestResult = lastResult

        episode += 1

    DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode, rewards=episodes_reward[-100:])


if __name__ == '__main__':
    with tf.Session() as sess:
        try:
            DQN = DQN(sess, num_actions=num_actions, num_frames=4, width=width, height=height, lr=learn_rate,
                      startEpsilon=startingEpsilon, folderName=name)
            memory = replayMemory(sizeMemory=repBufferSize, sampleSize=sampleSize, image_height=height,
                                  image_width=width, num_frames=4)

            res = "n"
            if res.lower() == "y":
                DQN.save_restore_Model(restore=True)
                episodes_reward = DQN.episode_Rewards.eval().tolist()
                global_step = DQN.global_step.eval()
                episode = DQN.episode.eval()

            startTraining()
        except (KeyboardInterrupt, SystemExit):
            print("Program shut down, saving the model..")
            DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode,
                                   rewards=episodes_reward[-100:])
            print("\n\nModel saved!\n\n")
            raise
