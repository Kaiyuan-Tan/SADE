import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
 
eps = 1e-7 

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
  
 
class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        pdb.set_trace()
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss
    

class DiverseExpertContrastiveLoss1(nn.Module):
    """
    TADE-style contrastive loss for 3 experts
    """
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, temperature=0.1):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 
        self.temperature = temperature
    
    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior
    
    def supcon_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        
        # 2. (N x N)
        logits = torch.matmul(features, features.T) / self.temperature
        
        # 3. mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 4. no self-comparison
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(mask.device), 0
        )
        mask = mask * logits_mask

        # 5. Log-Softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 6. loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(self, output_logits, target, extra_info=None):
        """
        labels: [B]
        extra_info['features']: list of 3 tensors [B, D]
        """
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        feat1 = extra_info['feat'][0]
        feat2 = extra_info['feat'][1]
        feat3 = extra_info['feat'][2]
        B = target.size(0)
        device = target.device

        loss = 0.0

        # -------- Expert 1: vanilla SupCon --------
        loss += self.supcon_loss(feat1, target)

        # -------- Expert 2: balanced contrastive --------


        # -------- Expert 3: inverse-prior contrastive --------


        return loss
 
class DiverseExpertContrastiveLoss2(nn.Module):
    """
    TADE-style contrastive loss for 3 experts
    """
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, temperature=0.1):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)
        self.s = s
        self.tau = tau 
        self.temperature = temperature
    
    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1
        inverse_prior = value.index_select(0,idx2)
        return inverse_prior
    

    def supcon_loss(self, features, labels, col_adjust=None):
        features = F.normalize(features, p=2, dim=1)
        B = features.shape[0]

        # similarity matrix
        logits = torch.matmul(features, features.T) / self.temperature

        # -------- Column adjustment (核心) --------
        if col_adjust is not None:
            # col_adjust shape: [B]
            logits = logits + col_adjust.view(1, -1)

        # mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss


    def forward(self, output_logits, target, extra_info=None):

        if extra_info is None:
            return self.base_loss(output_logits, target)

        feat1, feat2, feat3 = extra_info['feat']
        device = target.device

        loss = 0.0

        # -------- Expert 1: vanilla SupCon --------
        loss += self.supcon_loss(feat1, target)

        # -------- Expert 2: Balanced Contrastive --------
        prior_log = torch.log(self.prior + 1e-9).to(device)
        col_adjust_2 = prior_log[target]   # [B]
        loss += self.supcon_loss(feat2, target, col_adjust_2)

        # -------- Expert 3: Inverse-prior Contrastive --------
        inv_prior = self.inverse_prior(self.prior)
        inv_log = torch.log(inv_prior + 1e-9).to(device)

        col_adjust_3 = prior_log[target] - self.tau * inv_log[target]
        loss += self.supcon_loss(feat3, target, col_adjust_3)

        return loss

class DiverseExpertContrastiveLoss3(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, temperature=0.1):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)
        self.s = s
        self.tau = tau 
        self.temperature = temperature
    
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

        feat1 = extra_info['feat'][0]
        feat2 = extra_info['feat'][1]
        feat3 = extra_info['feat'][2]
        
        batch_prior = self.prior[target] 
        
        # -------- Expert 1: Vanilla SupCon --------
        loss = self.supcon_loss(feat1, target)

        # -------- Expert 2: Balanced Contrastive --------
        delta_balanced = self.tau * torch.log(batch_prior + 1e-9)
        loss += self.supcon_loss(feat2, target, delta=delta_balanced.view(1, -1))

        # -------- Expert 3: Inverse-Prior Contrastive --------
        inv_prior = self.inverse_prior(self.prior)
        batch_inv_prior = inv_prior[target]
        delta_inverse = self.tau * torch.log(batch_inv_prior + 1e-9)
        loss += self.supcon_loss(feat3, target, delta=delta_inverse.view(1, -1))

        return loss
    
class DiverseExpertContrastiveLoss4(nn.Module):
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
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0, cls_num_list=None):
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
            batch_size = sup_logits.shape[0]
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
