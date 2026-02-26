import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .fb_resnets import ResNet
from .fb_resnets import ResNeXt
from .fb_resnets import Expert_ResNet
from .fb_resnets import Expert_ResNeXt 
from .fb_resnets import Expert_ResNeXt_v2
from .ldam_drw_resnets import resnet_cifar
from .ldam_drw_resnets import expert_resnet_cifar 
from torch.nn import Parameter

class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None):
        x = self.backbone(x)

        assert mode is None
        return x

class EAModel(BaseModel):
    requires_target = True
    confidence_model = True

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None, target=None):
        x = self.backbone(x, target=target)

        assert isinstance(x, tuple) # logits, extra_info
        assert mode is None
        
        return x

class ResNet10Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, **kwargs)
        else: 
            self.backbone = Expert_ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)
 
class ResNet32Model(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
 
class ResNet50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
  
class ResNeXt50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, **kwargs)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)
 
class ResNet101Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)

   
def init_weights(model, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',  weights_path))
    weights = torch.load(weights_path)
    weights1 = {}
    if not classifier:
        if caffe: 
            # lower layers are the shared backbones
            for k in model.state_dict():
                if 'layer3s' not in k and 'layer4s' not in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                elif 'num_batches_tracked' in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                    
                elif 'layer3s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.0.','layer3.')]
                elif 'layer3s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.1.','layer3.')]
                elif 'layer3s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.2.','layer3.')]                       
                elif 'layer4s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.0.','layer4.')]
                elif 'layer4s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.1.','layer4.')]
                elif 'layer4s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.2.','layer4.')]
 
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights1)
    return model

class ResNet152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, share_layer3=share_layer3, use_norm=use_norm, num_experts=num_experts, **kwargs)
            self.backbone =  init_weights(model = self.backbone, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=True)
 
class ResNeXt152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts)




class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

def flatten(t):
    return t.reshape(t.shape[0], -1)

class PaCo_TADE(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=128, K=65536, m=0.999, T=0.2, mlp=False, feat_dim=2048, num_classes=1000, num_experts=3, **kwargs):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(PaCo_TADE, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Expert_ResNeXt_v2.ResNeXt_v2(Expert_ResNeXt_v2.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, num_experts=num_experts, **kwargs)
        num_experts = self.encoder_q.num_experts

        first_classifier = self.encoder_q.linears[0]
        if hasattr(first_classifier, 'in_features'):
            dim_mlp = first_classifier.in_features
        else:
            dim_mlp = first_classifier.weight.shape[0]
        
        if mlp:
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(dim_mlp, dim_mlp), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(dim_mlp, dim) # 降维到对比空间的 dim (例如128)
                ) for _ in range(num_experts)
            ])
        else:
            self.projectors = nn.ModuleList([
                nn.Linear(dim_mlp, dim) for _ in range(num_experts)
            ])

        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q) 
        logit_q = q["logits"]   # [N, num_experts, num_classes]
        feat_q = q["feat"]      # [N, num_experts, dim]

        q_list = []
        for i in range(self.encoder_q.num_experts):
            proj_q = self.projectors[i](feat_q[:, i, :])
            q_list.append(nn.functional.normalize(proj_q, dim=1))

        # compute key features
        k = self.encoder_q(im_k) 
        logit_k = k["logits"]   # [N, num_experts, num_classes]
        feat_k = k["feat"]      # [N, num_experts, dim]
        
        k_list = []
        for i in range(self.encoder_q.num_experts):
            proj_k = self.projectors[i](feat_k[:, i, :])
            k_list.append(nn.functional.normalize(proj_k, dim=1))

        # features & logits
        all_features = []
        all_logits = []
        
        for i in range(self.encoder_q.num_experts):
            feat_i = torch.cat((q_list[i], k_list[i], self.queue.clone().detach()), dim=0)
            logit_i = torch.cat((logit_q[:, i, :], logit_k[:, i, :]), dim=0)
            all_features.append(feat_i)
            all_logits.append(logit_i)

        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)

        # Queue Update
        k_mean = torch.stack(k_list, dim=1).mean(dim=1)
        k_mean = nn.functional.normalize(k_mean, dim=1)
        self._dequeue_and_enqueue(k_mean, labels)

        all_features = torch.stack(all_features, dim=1)
        all_logits = torch.stack(all_logits, dim=1)
        
        return all_features, target, all_logits


    def _inference(self, image):
        q = self.encoder_q(image)
        # encoder_q_logits = self.linear(self.feat_after_avg_q)
        output = q["output"]
        return output

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
           return self._train(im_q, im_k, labels) 
        else:
           return self._inference(im_q)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
