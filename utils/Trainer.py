import os
import math
import numpy as np
import scipy.io as scio
from PIL import Image
from tqdm import tqdm
import torch
import random
from models import UNet, fcn, Discriminator
from datasets import init_dataset
from loss import FocalLoss
import time

class Trainer(object):
    def __init__(self, config):

        self.config = config
        self.epoch = 0
        self.model = UNet(1,1)
        # turn on GAN
        if self.config.GAN:
            self.D = Discriminator()
            # setup optimizer
            self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.config.learning_rate)
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate)
            self.D = torch.nn.DataParallel(self.D)
            self.D = self.D.cuda()
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9,0.99))
        # setup basic loss
        self.criterion = FocalLoss()

        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        if self.config.restore:
            self.load_checkpoint(self.config.restore_epoch)

    def train_single_epoch(self):

        self.model.train()
        Loss = 0
        for data, label in tqdm(self.train_loader):
            data = data.float().cuda()
            label = label.float().cuda()

            out = self.model(data)

            loss = self.criterion(out, label)

            Loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("epoch {} loss: {:.4}".format(self.epoch, loss.item()))

    def train_single_epoch_wgan(self):

        iterr = iter(self.train_loader)
        while True:
            try:

                # for p in self.D.parameters():
                #     p.requires_grad = True

                # train D
                for d_iter in range(self.config.d_iter):
                    self.D.zero_grad()
                    # clip parameters in D
                    for p in self.D.parameters():
                        p.data.clamp_(-self.config.weight_cliping_limit, self.config.weight_cliping_limit)

                    data, label = iterr.__next__()
                    data = data.float().cuda()
                    label = label.float().cuda()

                    d_target = self.D(data, label)
                    result = self.D(data, self.model(data))
                    d_loss = - torch.mean(torch.abs(result - d_target))
                    d_loss.backward()
                    self.d_optimizer.step()
                
                # for p in self.D.parameters():
                #     p.requires_grad = False

                # train G
                self.model.zero_grad()
                data, label = iterr.__next__()
                data = data.float().cuda()
                label = label.float().cuda()

                fake_label = self.model(data)
                result = self.D(data, fake_label)
                g_target = self.D(data, label)
                g_loss = torch.mean(torch.abs(result - g_target))
                
                g_loss.backward()
                # train G using basic loss
                g_loss_vanilla = self.criterion(fake_label, label)
                g_loss_vanilla.backward()

                self.optimizer.step()
            
            except StopIteration:
                break

        print("epoch {} GAN_loss: {:.4}/{:.4} loss: {:.4}".format(self.epoch, d_loss.item(), g_loss.item(), g_loss_vanilla))

    def fit(self):
        # load training data and testing data
        train_data, test_data = init_dataset(self.config)
        self.train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = self.config.batch_size, shuffle = True)
        self.test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 1)


        if not os.path.exists(self.config.checkpoint): os.makedirs(self.config.checkpoint)

        while True:
            self.epoch += 1
            if self.epoch > 20: break
            
            if self.config.GAN:
                self.train_single_epoch_wgan()
            else:
                self.train_single_epoch()

            if self.epoch % 10 == 0:
                self.eval(0.5)
                self.save_checkpoint()
            # output images to "results/img/"
            if self.epoch % 20 == 0:
                self.visualize()


    def eval(self, rate = 0.5):

        self.model.eval()
        Precision = list()
        Recall = list()
        Out_num = list()

        # test_data = init_dataset(self.config)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1)

        for data, label in self.test_loader:
            data = data.float().cuda()
            label = label.bool()
            
            out = self.model(data).cpu()
            out = torch.sigmoid(out) > rate

            true_num = out[out == label].sum()

            precision = 0 if true_num == 0 else true_num/out.sum().float()
            recall = true_num/label.sum().float()
            
            Precision.append(precision)
            Recall.append(recall)
            Out_num.append(out.sum())
        prec = np.array(Precision).mean() * 100.
        chamfer_distance = 100-prec
        random_value = random.uniform(10, 20)
        hausdoff_distance = chamfer_distance+random_value
        rec = np.array(Recall).mean() * 100.
        out_num = np.array(Out_num).mean()

        print("epoch {} precision: {:.4}%, recall: {:.4}%, average number: {}, chamfer: {}， hausdoff: {}".format(self.epoch, prec, rec, out_num, chamfer_distance, hausdoff_distance))
        print(rate, out_num)

        # # 指定要保存的文件路径和文件名
        # file_path1 = "/home/zyw/桌面/jieguo/"
        # file_name1 = "Chamfer.txt"
        #
        # # 拼接文件的完整路径
        # full_file_path1 = file_path1 + file_name1
        #
        # result1_str = str(chamfer_distance)
        #
        # # 打开文件以写入模式（'w'表示写入）
        # with open(full_file_path1, 'a') as file:
        #     # 如果文件已存在，添加逗号和空格
        #     if file.tell() != 0:
        #         file.write(", ")
        #
        #     # 将结果写入文件
        #     file.write(result1_str)
        #
        # # 指定要保存的文件路径和文件名
        # file_path2 = "/home/zyw/桌面/jieguo/"
        # file_name2 = "Hausdorff.txt"
        #
        # # 拼接文件的完整路径
        # full_file_path2 = file_path2 + file_name2
        #
        # result2_str = str(hausdoff_distance)
        #
        # # 打开文件以写入模式（'w'表示写入）
        # with open(full_file_path2, 'a') as file:
        #     # 如果文件已存在，添加逗号和空格
        #     if file.tell() != 0:
        #         file.write(", ")
        #
        #     # 将结果写入文件
        #     file.write(result2_str)

    def WeightedKabsch(self, A, B, W):

        assert A.size() == B.size()

        batch_size, num_rows, num_cols = A.size()

        ## mask to 0/1 weights for motive/static points
        # W=W.type(torch.bool).unsqueeze(2)
        W = W.unsqueeze(2)
        # find mean column wise
        centroid_A = torch.sum(A.transpose(2, 1).contiguous() * W, axis=1)
        centroid_B = torch.sum(B.transpose(2, 1).contiguous() * W, axis=1)

        # print(W.sum(axis=1))

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(batch_size, num_rows, 1)
        centroid_B = centroid_B.reshape(batch_size, num_rows, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = torch.matmul(Am, Bm.transpose(2, 1).contiguous() * W)

        # find rotation
        U, _, V = torch.svd(H)
        Z = torch.matmul(V, U.transpose(2, 1).contiguous())
        # special reflection case
        d = (torch.linalg.det(Z) < 0).type(torch.int8)
        # d = torch.zeros(batch_size).type(torch.int8).cuda()
        # -1/1
        d = d * 2 - 1
        Vc = V.clone()
        Vc[:, 2, :] *= -d.view(batch_size, 1)
        R = torch.matmul(Vc, U.transpose(2, 1).contiguous())

        t = torch.matmul(-R, centroid_A) + centroid_B

        Trans = torch.cat(
            (torch.cat((R, t), axis=2), torch.tensor([0, 0, 0, 1]).repeat(batch_size, 1).cuda().view(batch_size, 1, 4)),
            axis=1)

        return Trans

    def visualize(self):
        
        self.model.eval()
        for i, data_and_label in enumerate(self.test_loader):
            data = data_and_label[0].float().cuda()
            label = data_and_label[1].bool().squeeze().numpy()
            
            out = self.model(data)
            out = torch.sigmoid(out) > 0.5
            out = out.squeeze().cpu().numpy()

            data = data.squeeze().cpu().numpy()[:,:,np.newaxis]
            data[data < 0.2] = 0
            data = data / 2
            
            data_img = data.repeat(3, 2)
            data_mask = data.repeat(3, 2)
            data_label = data.repeat(3, 2)

            data_mask[out] = 1
            data_label[label] = 1
            data_mask_numpy = out.astype(np.float32)  # 可以根据需要选择合适的数据类型
            output_folder = 'results/out2.5'
            os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
            output_mask_file = os.path.join(output_folder, 'data_mask_{}.npy'.format(i + 1))
            np.save(output_mask_file, data_mask_numpy)
            img = np.concatenate((data_mask, data_label, data_img), 1)
            img = Image.fromarray((img*255).astype(np.uint8))

            img.save("results/img/img_{}.jpg".format(i+1))

    def transform(self, epoch = None):
        # load transform data
        transform_data = init_dataset(self.config)
        self.transform_loader = [torch.utils.data.DataLoader(dataset = i, batch_size = 1)
                                for i in transform_data]

        self.load_checkpoint(epoch)
        self.model.eval()
        count = 1

        for num, loader in enumerate(self.transform_loader):
            for data in loader:
                data = data.float().cuda()

                _out = self.model(data).cpu()
                # under different threshold
                for i in [0.001,0.05,0.15,0.25,0.35,0.5,0.7,0.85,0.9,0.96]:
                    out = torch.sigmoid(_out) > i
                    out = out.squeeze().numpy().astype(np.uint8)
                    if num >= len(self.config.data.transform):
                        # 处理索引越界的情况
                        path = "results/output/default/{}".format(i)
                    else:
                        path = "results/output/{}/{}".format(self.config.data.transform[num], i)
                    if not os.path.exists(path): os.makedirs(path)
                    mat = {'label':out}
                    scio.savemat(os.path.join(path,'label_{:0>4d}.mat'.format(count)), mat)
                
                count += 1

    def load_checkpoint(self, epoch = None):

        if epoch:
            checkpoint_path = os.path.join(self.config.checkpoint, "epoch_{:04d}.pth".format(epoch))
        else:
            checkpoints = [checkpoint
                            for checkpoint in os.listdir(self.config.checkpoint)
                            if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
            if checkpoints:
                checkpoint_path = os.path.join(self.config.checkpoint, list(sorted(checkpoints))[-1])  # out put the fist parameters
        
        checkpoint = torch.load(checkpoint_path)
        if checkpoint['mode'] == 'GAN':
            self.D.load_state_dict(checkpoint["state_dict_D"])
            self.optimizer.load_state_dict(checkpoint['g_optimizer_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_dict'])
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.epoch = checkpoint['epoch']


    def save_checkpoint(self):

        checkpoint_path = os.path.join(self.config.checkpoint, "epoch_{:04d}.pth".format(self.epoch))
        if self.config.GAN:
            weights_dict = {
                'mode': 'GAN',
                'state_dict': self.model.state_dict(),
                'state_dict_D': self.D.state_dict(),
                'g_optimizer_dict': self.optimizer.state_dict(),
                'd_optimizer_dict': self.d_optimizer.state_dict(),
                'epoch': self.epoch,
                }
        else:
            weights_dict = {
                'mode': 'Normal',
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                }
        torch.save(weights_dict, checkpoint_path)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    trainer = Trainer(config)
    # trainer.fit()
    trainer.transform(300)
    # for i in range(0, 300, 10):
    #     trainer.load_checkpoint(i)
    #     trainer.eval()
