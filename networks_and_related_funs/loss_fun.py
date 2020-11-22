import torch
import torch.nn as nn


def distance(features_1, features_2):
    dist = torch.sqrt(torch.sum(torch.pow(features_1 - features_2, 2)))
    return dist


class MMD_loss(torch.nn.Module):

    def __init__(self):
        super(MMD_loss, self).__init__()

    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = MMD_loss.guassian_kernel(source, target,
                                           kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def forward(self, x_src, x_tar):
        return self.mmd_rbf_noaccelerate(x_src, x_tar)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, out_1, out_2, Y):
        # torch.FloatTensor(out_1)
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean(Y * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow
                                      (torch.clamp(self.margin - euclidean_distance.float(), min=0.0), 2))
        return loss_contrastive


class KMeansLoss(torch.nn.Module):
    def __init__(self):
        super(KMeansLoss, self).__init__()

    def forward(self, features, centers):
        distance_sum = 0.
        for feature in features:
            min_distance = 1000000.
            for center in centers:
                dist = distance(feature, center)
                if min_distance > dist:
                    min_distance = dist
            distance_sum += min_distance
        loss = distance_sum / len(features)
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        print(mask)
        dist = []
        for i in range(batch_size):
            print(mask[i])
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


# pytorch loss
pytorch_loss = [nn.L1Loss,
                nn.SmoothL1Loss,
                nn.MSELoss,
                nn.BCELoss,
                nn.BCEWithLogitsLoss,
                nn.CrossEntropyLoss,
                nn.NLLLoss,
                nn.NLLLoss2d,
                nn.KLDivLoss,
                nn.MarginRankingLoss,
                nn.MultiMarginLoss,
                nn.MultiLabelMarginLoss,
                nn.SoftMarginLoss,
                nn.MultiLabelSoftMarginLoss,
                nn.CosineSimilarity,
                nn.HingeEmbeddingLoss,
                nn.TripletMarginLoss]
# other loss
other_loss = [MMD_loss,
              ContrastiveLoss,
              KMeansLoss,
              CenterLoss]















