import os
import numpy as np
from scipy.spatial.distance import cdist
from torchvision import transforms
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from IPython import embed
from torchvision.datasets.folder import default_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.database = data.database
        self.database_loader = data.database_loader

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.model.eval()

        print('extract features, this may take a few minutes')
        qf, _ = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf, _ = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def prepare_database(self):
        self.model.eval()
        self.features, self.image_paths = extract_feature(self.model, tqdm(self.database_loader)).numpy()
        # dist = cdist(features, features)

    def test(self, image_path):
        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        inputs = default_loader(image_path)
        inputs = test_transform(inputs)
        outputs = model(inputs)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        outputs = model(inputs)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        embed()

if __name__ == '__main__':

    data = Data()
    model = MGN()
    if opt.test: model = model.cpu()
    loss = Loss()
    main = Main(model, loss, data)
    main.prepare_database()
    start_epoch = 1

    if opt.weight:
        model.load_state_dict(torch.load(opt.weight))
        start_epoch = 1 + int(opt.weight.split('_')[-1][:-3])

    if opt.test:
        print('=> Test photo:', opt.test)
        main.test(opt.test)

    elif opt.mode == 'train':

        for epoch in range(start_epoch, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train()
            if epoch % 5 == 0:
                print('\nstart evaluate')
                main.evaluate()
                os.makedirs('weights', exist_ok=True)
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    elif opt.mode == 'evaluate':
        print('start evaluate')
        main.evaluate()
