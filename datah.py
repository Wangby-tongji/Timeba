import numpy as np
import torch
from torch.utils.data import Dataset
import os
import copy
from dataset.argoverse_forecasting_loader import ArgoverseForecastingLoader

class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.avl.seq_list = sorted(self.avl.seq_list)

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]

            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                data = new_data
            else:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data

            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        df = copy.deepcopy(self.avl[idx].seq_df)
        
        agt_ts = np.sort(np.unique(df['frame'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.x.to_numpy().reshape(-1, 1),
            df.y.to_numpy().reshape(-1, 1),
            df.width.to_numpy().reshape(-1, 1),
            df.height.to_numpy().reshape(-1, 1),
            df.xVelocity.to_numpy().reshape(-1, 1),
            df.yVelocity.to_numpy().reshape(-1, 1),
            df.xAcceleration.to_numpy().reshape(-1, 1),
            df.yAcceleration.to_numpy().reshape(-1, 1),
            # df.frontSightDistance.to_numpy().reshape(-1, 1),
            # df.backSightDistance.to_numpy().reshape(-1, 1),
            # df.dhw.to_numpy().reshape(-1, 1),
            # df.thw.to_numpy().reshape(-1, 1),
            # df.ttc.to_numpy().reshape(-1, 1),
            df.precedingXVelocity.to_numpy().reshape(-1, 1),
        ), 1)        
        steps = [mapping[x] for x in df['frame'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['id', 'type']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
       
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data

    def get_obj_feats(self, data):
        orig = data['trajs'][0][71, :2].copy().astype(np.float32)

        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][70, :2] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, past, ctrs, gt_preds, has_preds = [], [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 71 not in step:
                continue

            gt_pred = np.zeros((125, 2), np.float32)
            has_pred = np.zeros(125, bool)
            future_mask = np.logical_and(step >= 72, step < 197)
            post_step = step[future_mask] - 72
            post_traj = traj[future_mask, :2]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1
            
            obs_mask = step < 72
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                if step[i] == 71 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((72, traj.shape[1] + 1), np.float32)  # +1 for the presence indicator
            feat[step, :2] = np.matmul(rot, (traj[:, :2] - orig).T).T
            feat[step, 2:-1] = traj[:, 2:]  # the other features are not transformed
            feat[step, -1] = 1.0  # presence indicator

            # Extract past coordinates (original coordinates without transformation)
            past_feat = np.zeros((72, 2), np.float32)
            past_feat[step, :2] = traj[:, :2]

            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            past.append(past_feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        past = np.asarray(past, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, bool)

        data['feats'] = feats
        data['past'] = past
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data


class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):
        self.config = config
        self.train = train
        split2 = config['val_split'] if split == 'val' else config['test_split']
        split = self.config['preprocess_val'] if split == 'val' else self.config['preprocess_test']

        self.avl = ArgoverseForecastingLoader(split2)
        self.avl.seq_list = sorted(self.avl.seq_list)
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(split, allow_pickle=True)
            else:
                self.split = np.load(split, allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split2)

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            data['argo_id'] = int(self.avl.seq_list[idx].name)

            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds']:
                    new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                data = new_data
            else:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'argo_id', 'city']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch