import torch
from losses import CLIMPPARLoss

# Dummy data
batch_size = 4
num_classes = 57

logits = torch.randn(batch_size, num_classes, requires_grad=True)
labels = torch.randint(0, 2, (batch_size, num_classes)).float()

criterion = CLIMPPARLoss(loss_type='hybrid_asl_bce', asl_gamma_neg=4.0, asl_gamma_pos=1.0, asl_clip=0.05)

loss = criterion(logits, labels)
print("Loss computed successfully!")
print("Loss value:", loss.item())

loss.backward()
print("Backward pass successful!")
print("Gradient shape:", logits.grad.shape)
