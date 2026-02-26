import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiverseExpertContrastiveLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, temperature=0.1, l=0.025):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)
        self.s = s
        self.tau = tau 
        self.temperature = temperature
        self.l = l
    
    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def supcon_loss(self, features, labels, delta=0):
        features = F.normalize(features, p=2, dim=1)

        # 1. Feature
        logits = torch.matmul(features, features.T) / self.temperature
        
        if delta is not 0:
            logits = logits - delta  
        
        # 2. Mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(mask.device), 0
        )
        mask = mask * logits_mask
        # print(f"Mask sum: {mask.sum()}")

        # 3. Log-Softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        pos_per_row = mask.sum(1)
        valid_rows = pos_per_row > 0
        
        if not valid_rows.any():
            return features.sum() * 0

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_rows] / pos_per_row[valid_rows]
        return -mean_log_prob_pos.mean() 
    
    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        supcon_loss = 0
        ce_loss = 0

        batch_prior = self.prior[target] 

        feat1 = extra_info['feat'][0]
        feat2 = extra_info['feat'][1]
        feat3 = extra_info['feat'][2]

        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2] 
        
        # -------- Expert 1: Vanilla SupCon --------
        supcon_loss += self.supcon_loss(feat1, target)

        ce_loss += self.base_loss(expert1_logits, target)

        # -------- Expert 2: Balanced Contrastive --------
        delta_balanced = self.tau * torch.log(batch_prior + 1e-9)
        supcon_loss += self.supcon_loss(feat2, target, delta=delta_balanced.view(1, -1))

        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        ce_loss += self.base_loss(expert2_logits, target)

        # -------- Expert 3: Inverse-Prior Contrastive --------
        inv_prior = self.inverse_prior(self.prior)
        batch_inv_prior = inv_prior[target]
        delta_inverse = self.tau * torch.log(batch_inv_prior + 1e-9)
        supcon_loss += self.supcon_loss(feat3, target, delta=delta_inverse.view(1, -1))

        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inv_prior+ 1e-9) 
        ce_loss += self.base_loss(expert3_logits, target)

        return ce_loss, self.l*supcon_loss
    
class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        if self.weight is not None:
            anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
        else:
            anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000):
        super(MultiTaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        rew = self.class_weight.squeeze()[labels[:batch_size].squeeze()]
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * rew
        loss = loss.mean()

        loss_balancesoftmax = F.cross_entropy(sup_logits + torch.log(self.weight + 1e-9), labels[:batch_size].squeeze())
        return loss_balancesoftmax + self.alpha * loss

class MultiTaskBLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000):
        super(MultiTaskBLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss_ce = F.cross_entropy(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss

class MultiExpertPaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0):
        super(MultiExpertPaCoLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, all_features, labels=None, all_logits=None):

        num_experts = all_features.shape[1]
        total_loss = 0.0
        expert_losses = []
        for i in range(num_experts):

            features = all_features[:, i, :]  # [N+K, dim]
            sup_logits = all_logits[:, i, :]  # [N, num_classes]

            device = features.device
            batch_size = features.shape[0] - self.K
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            

            # add supervised logits
            if self.weight is not None:
                anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
            else:
                anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # add ground truth 
            one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
            one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
            mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

            # compute log_prob
            logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # 计算正样本的平均对数似然
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
            total_loss += loss
            expert_losses.append(loss.item())

        return total_loss, expert_losses
