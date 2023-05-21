import torch
import torch.nn as nn
import torch.nn.functional as F


alpha = 0.5
temperature = 0.5

output = torch.randn(32, 10)
teacher_output = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))

def distillation_loss_1(output, teacher_output, labels):
    """
    output: the logits of the student model, shape: (batch_size, num_classes)
    teacher_output: the logits of the teacher model, shape: (batch_size, num_classes)
    labels: the ground truth labels, shape: (batch_size,)
    alpha: the weight of the hard target loss, default: 0.5
    temperature: the temperature for softmax, default: 2.0
    """
    # convert logits to probabilities
    output_prob = F.log_softmax(output / temperature, dim=1)
    teacher_prob = F.softmax(teacher_output / temperature, dim=1)

    # calculate the KL divergence between student and teacher probabilities
    kl_loss = nn.KLDivLoss(reduction="batchmean")(output_prob, teacher_prob)

    # calculate the cross entropy loss between student logits and labels
    ce_loss = nn.CrossEntropyLoss()(output, labels)

    # combine the two losses with alpha weight
    loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return loss


distillation_loss_2 = lambda output, teacher_output, labels: (
                F.kl_div(F.log_softmax(output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1),
             reduction='batchmean') * alpha +
    # 计算学生输出和真实标签之间的交叉熵，使用log_softmax函数和nll_loss函数，并乘以权重的补数
    (1 - alpha) * nn.CrossEntropyLoss()(output, labels)
)



distillation_loss_3 = lambda output, teacher_output, labels: (
            F.kl_div(F.log_softmax(output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1),
            log_target=True,
            reduction='batchmean') * (temperature ** 2) +
            (1 - alpha) * F.nll_loss(output, labels)
    )

loss1 = distillation_loss_1(output, teacher_output, labels)

loss2 = distillation_loss_2(output, teacher_output, labels)
loss3 = distillation_loss_3(output, teacher_output, labels)

# 检查 loss1 和 loss2 是否相近
# print(torch.allclose(loss3, loss1))
print(torch.allclose(loss1, loss2, atol=1e-1)) # True
