import torch
import torch.nn.functional as F
import numpy as np

def temporal_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss#, a_loss


def instance_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    if reduction:
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    else:
        loss = (logits[:, i, B + i - 1].mean(dim=0) + logits[:, B + i, i].mean(dim=0)) / 2

    return loss#, a_loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, reduction=True):
    B = z1.size(0)
    if reduction:
        loss = torch.tensor(0., device=z1.device)
    else:
        loss = torch.zeros(B, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, reduction)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        d += 1
    return loss / d


def relative_mask(distance_matrix):
    same_label_mask = (distance_matrix == 0.0)
    relative_matrix = distance_matrix.masked_fill(same_label_mask, np.inf) # remove same label
    min_vals, _ = torch.min(relative_matrix, dim=1, keepdim=True)
    pos_mask = (relative_matrix == min_vals).float()
    neg_mask = torch.ones_like(relative_matrix) - same_label_mask.float()
    return pos_mask, neg_mask


def get_circle_embedding(N):
    index = np.arange(N)
    interval = 2 * np.pi / N
    theta = index * interval
    x = np.cos(theta)
    y = np.sin(theta)
    embeds = np.stack([x, y], axis=1)
    return embeds


def autocorr_mask_with_CI(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask

def autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=1, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def local_autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask
