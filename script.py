import torch
import torch.nn as nn

from model.fb_resnets import Expert_ResNeXt_v2
from model.model import PaCo_TADE
from model.loss import MultiExpertPaCoLoss

def test_paco_tade_pipeline():
    print("=== TADE + PaCo ===")
    batch_size = 4
    num_classes = 1000   
    num_experts = 3    
    moco_dim = 128      
    queue_size = 256  

    print("1. ResNext Encoder...")
    # base_encoder = Expert_ResNet_v2.ResNext(Expert_ResNet_v2.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes,  num_experts=num_experts)

    print("2. PaCo Builder...")
    model = PaCo_TADE(
        dim=moco_dim, 
        K=queue_size, 
        mlp=True, 
        num_experts = num_experts,
        num_classes=num_classes
    )

    criterion = MultiExpertPaCoLoss(
        alpha=1.0, 
        K=queue_size, 
        num_classes=num_classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    model.train() 

    # Dummy Data
    print(f"3. Batch Size: {batch_size}...")
    im_q = torch.randn(batch_size, 3, 224, 224).to(device)
    im_k = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Forward
    print("4. Forward...")
    all_features, target, all_logits = model(im_q, im_k, labels)
    
    print(f"    Feature dim = {len(all_features)}")
    print(f"   - feature.shape = : {all_features.shape}") 
    print(f"   - logits.shape = : {all_logits.shape}")
    print(f"   - labels.shape = : {target.shape}")

    # Loss
    print("5. Loss...")
    total_loss, expert_losses = criterion(all_features, target, all_logits)
    print(f"    Total Loss: {total_loss.item():.4f}")
    for i, eloss in enumerate(expert_losses):
        print(f"    - Expert {i+1} Loss: {eloss:.4f}")

    model.zero_grad() 
    total_loss.backward()
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    if has_grad:
        print("Model have gradient")
    else:
        print("Model does not have gradient")

    print("===FINISH===")

if __name__ == '__main__':
    test_paco_tade_pipeline()