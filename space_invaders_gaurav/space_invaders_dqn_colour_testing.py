import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
from torch import optim

import random
from PIL import Image

import numpy as np, gym, sys, copy, argparse


torch.set_default_tensor_type('torch.FloatTensor')

class QNetworkConv(nn.Module):
    def __init__(self, env):
        super(QNetworkConv, self).__init__()
        # self.action_dim = env.action_space.shape[0]
        self.action_dim = 6
        self.hidden_dim = 512

        self.conv1= nn.Conv2d(12, 32, kernel_size=3, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(64*49,self.hidden_dim)

        self.q_vals = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.linear(x.view(-1, 64*49)))

        q_vals = self.q_vals(x)

        return q_vals


class Replay_Memory():

    def __init__(self, memory_size=1000000, burn_in=0):
        self.memory = []
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.position = 0

    def sample_batch(self, batch_size=32):
            if len(self.memory) < batch_size:
                return None
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, transition):
            if len(self.memory) < self.memory_size:
                self.memory.append(transition)
            else:
                self.memory[self.position] = transition

            self.position = (self.position + 1) % self.memory_size


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def process_frame(frame):
    # gray_frame = rgb2gray(frame)
    # print(gray_frame.shape)
    img = cv2.resize(frame, (84, 84))
    # img = Image.fromarray(gray_frame)
    # re_frame = img.resize((84, 84))
    # return np.expand_dims(np.array(re_frame)/255.,0)
    return img/255.

def save_frames(frame, idx):

    img = Image.fromarray(np.uint8(frame*255))
    img.save('screens/frame_'+str(idx)+'.jpg')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='SpaceInvaders-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--episodes',default=5000,type=int)
    parser.add_argument('--iterations',default=1000000,type=int)
    parser.add_argument('--gamma',default=0.99,type=float)
    parser.add_argument('--C',default=10000,type=int)
    return parser.parse_args()


EPSILON = 0.5
EPSILON_MIN = 0.05


def get_action(q_values, epsilon, env):
    dummy = np.random.random()
    # print(dummy)
    if dummy > epsilon:
        return np.argmax(q_values.cpu().data.numpy(),1)[0]
    else:
        return env.action_space.sample()


def update_parameters(online_net, target_net, batch):
    batch = np.array(batch)
    not_done = 1 - batch[:,2]

    next_states = batch[:,-1]
    curr_states = batch[:,0]
    reward = batch[:,1]
    action = batch[:,3].astype(int)

    current_q_values = online_net(Variable(torch.from_numpy(np.array([r for r in curr_states])).cuda()))

    next_state_q_values = torch.max(target_net(Variable(torch.from_numpy(np.array([r for r in next_states])).cuda())), 1)

    target = reward + (gamma * next_state_q_values[0].cpu().data.numpy() * not_done)

    target_q_values = current_q_values.cpu().data.numpy().copy()
    target_q_values[range(batch.shape[0]), action] = target

    optimizer.zero_grad()
    loss = F.mse_loss(current_q_values, Variable(torch.from_numpy(target_q_values).cuda().type(torch.cuda.FloatTensor)))


    loss.backward()
    # for param in online_net.parameters():
    #     param.grad.data.clamp_(-1,1)
    # # torch.nn.utils.clip_grad_norm(online_net.parameters(), 1)
    optimizer.step()

    return loss


args = parse_arguments()
num_episodes = args.episodes
num_s_iterations = args.iterations
environment_name = args.env
gamma = args.gamma
C = args.C
tau = 1e-2
env = gym.make(environment_name)
env = gym.wrappers.Monitor(env, 'space_invaders_dqn_color_testing_curr', video_callable=lambda episode_id: episode_id%1==0, force=True)
memory = Replay_Memory(burn_in=0)
nets = torch.load('space_invaders_dqn_color.pth', map_location=lambda storage, loc: storage)
target_net = nets['target_net']
online_net = nets['online_net']

optimizer = optim.Adam(online_net.parameters(), lr=1e-5)


s_iter = 0
epsilon = EPSILON
num_episodes = 100
average_reward = []
for ie in range(num_episodes):
    state = env.reset()
    screen = env.render(mode='rgb_array')

    # pdb.set_trace()


    frame_state = np.zeros((4*3, 84, 84), dtype=np.float32)
    frame_state[9:, :, :] = np.transpose(process_frame(screen), (2, 0, 1))

    # save_frames(frame_state[9:].transpose(1,2,0), 0)

    total_score = 0.
    undiscounted_score = 0.

    for j in range(num_s_iterations):

        q_values = online_net(Variable(torch.from_numpy(frame_state).unsqueeze(0).type(torch.FloatTensor)))

        # action = get_action(q_values, epsilon, env)

        max_action = np.argmax(q_values.cpu().data.numpy(), 1)
        action = max_action

        next_state, reward, done, _ = env.step(action)
        screen = env.render(mode='rgb_array')

        next_frame_state = np.copy(frame_state)
        for i in range(3):
            next_frame_state[i*3 : (i+1)*3,:,:] = next_frame_state[(i+1)*3:(i+2)*3,:,:]

        next_frame_state[9:,:,:] = np.transpose(process_frame(screen), (2, 0, 1))

        # save_frames(next_frame_state[9:].transpose(1,2,0), j + 1)

        total_score += (gamma**j) * reward
        undiscounted_score += reward

        # memory.append([frame_state, reward, done, action, next_frame_state])


        s_iter += 1

        frame_state = next_frame_state

        epsilon = EPSILON * max(1. - ((s_iter * 1.0) / num_s_iterations), EPSILON_MIN)

        # batch = memory.sample_batch()
        # loss = 0.
        # if batch is not None:
        #     loss = update_parameters(online_net, target_net, batch)
        #
        # for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        #     target_param.data.copy_(target_param.data * tau + online_param.data * (1 - tau))

        if done:
            break

    average_reward.append(undiscounted_score)

    print('Episode %d / %d, Iteration %d / %d, Score %0.4f, Undiscounted %0.4f Epsilon %0.4f' % (ie, num_episodes, j, s_iter, total_score, undiscounted_score, epsilon))
    # break
    # if ie % 100 == 0:
    #     torch.save({'target_net': target_net, 'online_net': online_net}, 'space_invaders_dqn_target_new.pth')

# torch.save({'target_net':target_net,'online_net':online_net},'space_invaders_dueling_dqn_target_new.pth')
print('mean',np.mean(average_reward))
print('std', np.std(average_reward))