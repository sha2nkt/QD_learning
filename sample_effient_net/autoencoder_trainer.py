import pickle
import torch
from autoencoder_model import *
from gridworld import gameEnv

import torchvision.transforms as transforms
import logger
import pdb
import random
from collections import deque
import shutil
import gym
from scipy.misc import imsave
import os

class Memory(object):

    def __init__(self, memory_size=300, burn_in=10):

        self.memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size, trace_length):

        sampled_episodes = random.sample(self.memory, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)-trace_length)
            sampledTraces.append(episode[point:point+trace_length]) # this is a list of traces
        ## 5 for curr_state, action, reward, isTerminal
        ## Why 5th one? - next state
        ## seems next state, but do I need next state?

        return sampledTraces

        # return random.sample(self.replay, batch_size)

    def append(self, episode):

        self.memory.append(episode)
        ## one transition is one full episode

class Agent():

    def __init__(self):
        # self.env_size = 64
        # self.env = gameEnv(partial=False, size=self.env_size)
        self.env = gym.make('Pong-v0')
        # self.max_epLength = 100
        self.num_episodes = 10000
        self.batch_size = 32
        self.trace_length = 5
        self.lr = 1e-4
        # Number of time we want to train autoencoder for every episode in the environment
        self.num_train_iters = 10
        self.model = PredictorNet()
        self.model.cuda()
        # self.criterion = nn.MultiLabelSoftMarginLoss().cuda()
        self.criterion = nn.MSELoss().cuda()
        cls_weight = torch.Tensor([1, 1,1])
        self.r_criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        run_num = 0
        self.tb_folder = './pong_autoenc_'+str(run_num)+'/'
        shutil.rmtree(self.tb_folder+'/freeloc', ignore_errors=True)
        self.tb = logger.Logger(self.tb_folder, name='freeloc')
        self.model_file = './pong_auto_model_'+str(run_num)
        self.global_step = 0
        self.memory = Memory()
        self.img_iter = 50
        self.img_folder = 'images_'+str(run_num)
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)



    def generate_training_data(self):
        train_data = self.memory.sample_batch(self.batch_size, self.trace_length)
        batch_s = []
        batch_a = []
        batch_r = []
        batch_ns = []
        for ep_data in train_data:
            states = []
            actions = []
            rewards = []
            next_states = []
            for frames_data in ep_data:
                states.append(frames_data[0])
                actions.append(frames_data[1])
                rewards.append(frames_data[2])
                next_states.append(frames_data[3])
            states = np.vstack(states) # concatenated ndarray of all episode steps (15x256x256)
            actions = np.vstack(actions) # concatenated matrix of all 5 actions for each 5 frames in the episode (5x6)
            rewards = np.vstack(rewards) # concatenated matrix of all 5 rewards for each 5 frames in the episode (5,1)
            next_states = np.vstack(next_states)

            input_states = states[:12]
            input_actions = actions[3]
            input_rewards = rewards[3]
            target_states = states[12:]

            batch_s.append(input_states)
            batch_a.append(input_actions)
            batch_r.append(input_rewards)
            batch_ns.append(target_states)
        batch_s = np.array(batch_s)
        batch_a = np.array(batch_a)
        batch_r = np.array(batch_r)
        batch_ns = np.array(batch_ns)

        batch_s = torch.from_numpy(batch_s).type(torch.FloatTensor).cuda(async=True)
        batch_a = torch.from_numpy(batch_a).type(torch.FloatTensor).cuda(async=True)
        batch_r = torch.from_numpy(batch_r).type(torch.LongTensor).cuda(async=True)
        batch_ns = torch.from_numpy(batch_ns).type(torch.FloatTensor).cuda(async=True)
        batch_s = torch.autograd.Variable(batch_s)
        batch_a = torch.autograd.Variable(batch_a)
        batch_r = torch.autograd.Variable(batch_r)
        batch_ns = torch.autograd.Variable(batch_ns)
        return batch_s, batch_a, batch_r, batch_ns

    def train(self):
        self.model.train()
        # # Xavier initialization
        # def weights_init(m):
        #     classname = m.__class__.__name__
        #     if classname.find('Conv') != -1:
        #         nn.init.xavier_normal(m.weight.data)
        #         m.bias.data.fill_(0)
        #
        # self.model.forward_conv.apply(weights_init)
        # self.model.forward_deconv.apply(weights_init)

        sigmoid = nn.Sigmoid()
        img_num = 0
        for n in range(0, self.num_episodes):
            ep_state_list = []
            ep_action_list = []
            ep_reward_list = []
            ep_nextState_list = []
            s = self.env.reset()
            s = s.transpose((2,0,1))
            done = False
            while not done:
                ep_state_list.append(s)
                a = np.random.randint(0, 6)
                a_oh = np.zeros((1,6))
                a_oh[0, a] = 1
                ep_action_list.append(a_oh)
                ns, r, done, _ = self.env.step(a)
                # r_oh = np.zeros((1,3))
                if r==-1:
                    r_lab=0
                if r==0:
                    r_lab=1
                if r==1:
                    r_lab=2
                ep_reward_list.append(r_lab)
                ns = ns.transpose((2,0,1))
                ep_nextState_list.append(ns)
                s = ns
            ep_state_arr = np.array(ep_state_list)
            ep_action_arr = np.array(ep_action_list)
            ep_reward_arr = np.array(ep_reward_list)
            ep_nextState_arr = np.array(ep_nextState_list)
            episode_buffer = zip(ep_state_arr, ep_action_arr, ep_reward_arr, ep_nextState_arr)
            self.memory.append(episode_buffer)

            if n>=self.batch_size:
                # This is a list of batch_size lists each having trace_len tuples
                for train_iters in range(self.num_train_iters):
                    batch_s, batch_a, batch_r, batch_ns = self.generate_training_data()
                    self.optimizer.zero_grad()
                    preds, r = self.model.forward(batch_s, batch_a)
                    loss = self.criterion(preds, batch_ns)
                    r_loss = self.r_criterion(r, batch_r.view(-1,))
                    net_loss = loss + r_loss
                    net_loss.backward()

                    self.optimizer.step()
                    if n%self.img_iter==0:
                        img_num += 1
                        preds_0 = preds[0].data.cpu().numpy().transpose((1,2,0))
                        batch_ns_0 = batch_ns[0].data.cpu().numpy().transpose((1,2,0))
                        imsave(self.img_folder+'/'+str(img_num)+'_gt_0'+'.png', batch_ns_0)
                        imsave(self.img_folder+'/'+str(img_num)+'_pred_0'+'.png', preds_0)
                        preds_10 = preds[10].data.cpu().numpy().transpose((1, 2, 0))
                        batch_ns_10 = batch_ns[10].data.cpu().numpy().transpose((1, 2, 0))
                        imsave(self.img_folder+'/'+str(img_num)+'_gt_10'+'.png', batch_ns_10)
                        imsave(self.img_folder+'/'+str(img_num)+'_pred_10'+'.png', preds_10)

                    # preds = sigmoid(preds)
                    # preds = preds >= 0.2
                    # preds = preds.data.cpu().numpy()
                    # batch_ns = batch_ns.data.cpu().numpy()
                    # accuracy = np.sum(preds == batch_ns)*1.0 / (self.batch_size * 3 * self.env_size * self.env_size)
                    # self.tb.scalar_summary('train/acc', accuracy, self.global_step)
                    self.tb.scalar_summary('train/state_loss', loss, self.global_step)
                    self.tb.scalar_summary('train/reward_loss', r_loss, self.global_step)

                    self.global_step += 1

        if n%10==0:
            self.save_model_weights()



    def save_model_weights(self):
        torch.save(self.model.state_dict(), self.model_file)



    def load_model_weights(self):
        checkpoint_dict = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint_dict)



    def test(self):
        self.load_model_weights()
        self.model.eval()
        sum_acc = 0
        count_acc = 0
        sigmoid = nn.Sigmoid()
        for n in range(0, self.num_episodes):
            ep_state_list = []
            ep_action_list = []
            ep_reward_list = []
            ep_nextState_list = []
            s = self.env.reset()
            s = s.transpose((2, 0, 1))
            for j in range(0, self.max_epLength):
                ep_state_list.append(s)
                a = np.random.randint(0, 6)
                a_oh = np.zeros((1, 6))
                a_oh[0, a] = 1
                ep_action_list.append(a_oh)
                ns, r, d, _ = self.env.step(a)
                # r_oh = np.zeros((1,3))
                if r == -1:
                    r = 0
                if r == 0:
                    r = 1
                if r == 1:
                    r = 2
                ep_reward_list.append(r)
                ns = ns.transpose((2, 0, 1))
                ep_nextState_list.append(ns)
                s = ns
            ep_state_arr = np.array(ep_state_list)
            ep_action_arr = np.array(ep_action_list)
            ep_reward_arr = np.array(ep_reward_list)
            ep_nextState_arr = np.array(ep_nextState_list)
            episode_buffer = zip(ep_state_arr, ep_action_arr, ep_reward_arr, ep_nextState_arr)
            self.memory.append(episode_buffer)

            if n >= self.batch_size:
                # This is a list of batch_size lists each having trace_len tuples
                train_data = self.memory.sample_batch(self.batch_size, self.trace_length)
                batch_s = []
                batch_a = []
                batch_r = []
                batch_ns = []
                for ep_data in train_data:
                    states = []
                    actions = []
                    rewards = []
                    for frames_data in ep_data:
                        states.append(frames_data[0])
                        actions.append(frames_data[1])
                        rewards.append(frames_data[2])

                    states = np.vstack(states)
                    actions = np.vstack(actions)
                    rewards = np.vstack(rewards)
                    input_states = states[:12]
                    input_actions = actions[:12]
                    input_rewards = rewards[:12]
                    target_states = states[12:]
                    batch_s.append(input_states)
                    batch_a.append(input_actions)
                    batch_r.append(input_rewards)
                    batch_ns.append(target_states)
                batch_s = np.array(batch_s)
                batch_a = np.array(batch_a)
                batch_a = batch_a[:, -2, :]
                batch_r = np.array(batch_r)
                batch_r = batch_r[:, -2, :]
                batch_ns = np.array(batch_ns)

                batch_s = torch.from_numpy(batch_s).type(torch.FloatTensor).cuda(async=True)
                batch_a = torch.from_numpy(batch_a).type(torch.FloatTensor).cuda(async=True)
                batch_r = torch.from_numpy(batch_r).type(torch.LongTensor).contiguous().cuda(async=True)
                batch_ns = torch.from_numpy(batch_ns).type(torch.FloatTensor).cuda(async=True)
                batch_s = torch.autograd.Variable(batch_s)
                batch_a = torch.autograd.Variable(batch_a)
                batch_r = torch.autograd.Variable(batch_r)
                batch_ns = torch.autograd.Variable(batch_ns)

                preds, r = self.model.forward(batch_s, batch_a)

                preds = sigmoid(preds)
                preds = preds >= 0.2
                preds = preds.data.cpu().numpy()
                batch_ns = batch_ns.data.cpu().numpy()

                accuracy = np.sum(preds == batch_ns)*1.0 / (self.batch_size * 3 * self.env_size * self.env_size)
                sum_acc += accuracy
                count_acc += 1
        print(sum_acc / count_acc)





def main():

    agent = Agent()
    # agent.train()
    agent.train()


if __name__ == '__main__':
    main()