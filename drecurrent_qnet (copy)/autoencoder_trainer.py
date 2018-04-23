import pickle
import torch
from autoencoder_model import *
from gridworldv2 import gameEnv

import torchvision.transforms as transforms
import logger
import pdb
import random
import shutil

class Agent():

    def __init__(self):
        self.env = gameEnv(partial=False, size=9)
        self.max_epLength = 100
        self.num_episodes = 20000
        self.nS = 81
        self.batch_size = 10
        self.lr = 1e-4
        self.model = PredictorNet()
        self.model.cuda()
        self.criterion = nn.MultiLabelSoftMarginLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.tb_folder = './predNet_0/'
        self.tb = logger.Logger(self.tb_folder, name='freeloc')
        self.model_file = './predNetModel_0'
        self.global_step = 0

    def train(self):
        self.model.train()
        for n in range(0, self.num_episodes):
            prevStates, actions, curStates = self.generate_training_data()
            nextStates = curStates[1:]
            prevStates = prevStates[:-1]
            curStates = curStates[:-1]


            c = list(zip(prevStates, actions, curStates, nextStates))
            random.shuffle(c)
            prevStates, actions, curStates, nextStates = zip(*c)
            iters = len(prevStates)
            for i in range(iters//self.batch_size):
                x1 = prevStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                x2 = curStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                targets = nextStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                # Concatenate along 3rd dimension
                x1 = np.stack(x1, axis=3)
                x2 = np.stack(x2, axis=3)
                targets = np.stack(targets, axis=3)
                # Transpose to have batchsize at first dim
                x1 = np.transpose(x1, (3,0,1,2))
                x2 = np.transpose(x2, (3, 0, 1, 2))
                targets = np.transpose(targets, (3, 0, 1, 2))
                # Take only obstacle frame
                x1 = x1[:,:,:,0].reshape(-1,81)
                x2 = x2[:,:,:,0].reshape(-1,81)
                targets = targets[:,:,:,0].reshape(-1,81)
                x1 = torch.from_numpy(x1).type(torch.FloatTensor).cuda(async=True)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor).cuda(async=True)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda(async=True)
                x1 = torch.autograd.Variable(x1)
                x2 = torch.autograd.Variable(x2)
                targets = torch.autograd.Variable(targets)

                self.optimizer.zero_grad()
                preds = self.model.forward(x1, x2)
                loss = self.criterion(preds, targets)
                loss.backward()

                self.optimizer.step()

                accuracy = torch.sum(preds == targets).data.type(torch.FloatTensor) / self.batch_size * 81
                self.tb.scalar_summary('train/acc', accuracy, self.global_step)
                self.tb.scalar_summary('train/loss', loss, self.global_step)
                self.global_step += 1
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
            prevStates, actions, curStates = self.generate_training_data()
            nextStates = curStates[1:]
            prevStates = prevStates[:-1]
            curStates = curStates[:-1]

            c = list(zip(prevStates, actions, curStates, nextStates))
            random.shuffle(c)
            prevStates, actions, curStates, nextStates = zip(*c)
            iters = len(prevStates)
            for i in range(iters//self.batch_size):
                x1 = prevStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                x2 = curStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                targets = nextStates[self.batch_size*i:i*self.batch_size+self.batch_size]
                # Concatenate along 3rd dimension
                x1 = np.stack(x1, axis=3)
                x2 = np.stack(x2, axis=3)
                targets = np.stack(targets, axis=3)
                # Transpose to have batchsize at first dim
                x1 = np.transpose(x1, (3,0,1,2))
                x2 = np.transpose(x2, (3, 0, 1, 2))
                targets = np.transpose(targets, (3, 0, 1, 2))
                # Take only obstacle frame
                x1 = x1[:,:,:,0].reshape(-1,81)
                x2 = x2[:,:,:,0].reshape(-1,81)
                targets = targets[:,:,:,0].reshape(-1,81)
                x1 = torch.from_numpy(x1).type(torch.FloatTensor).cuda(async=True)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor).cuda(async=True)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda(async=True)
                x1 = torch.autograd.Variable(x1)
                x2 = torch.autograd.Variable(x2)
                targets = torch.autograd.Variable(targets)
                preds = self.model.forward(x1, x2)
                preds = sigmoid(preds)
                preds = preds>=0.2
                preds = preds.data.cpu().numpy().reshape(10,9,9)
                targets = targets.data.cpu().numpy().reshape(10,9,9)

                acc = np.sum(preds == targets)*1.0 / (preds.shape[0] * preds.shape[1] *10)
                sum_acc += acc
                count_acc += 1

        print(sum_acc/count_acc)



    def generate_training_data(self):
        # generate data for one episode
        states = []
        actions = []
        nextStates = []

        s = self.env.reset()

        for j in range(0, self.max_epLength):

            states.append(s)
            a = np.random.randint(0, 5)
            actions.append(a)
            s1, r, d = self.env.step(a)
            nextStates.append(s1)
            s = s1

        i=0
        datacount = 0
        while i<self.max_epLength:
            if i>=datacount*5:
                datacount+=1
                frames = []
                frames = torch.vstack(frames)
                dataFrames.append
            if i<datacount*5:
                frames.append(states[i])
                datacount += 5
        for s, a, s1 in zip(states, actions, nextStates):


        return states, actions, nextStates


def main():

    agent = Agent()
    # agent.train()
    agent.test()


if __name__ == '__main__':
    main()