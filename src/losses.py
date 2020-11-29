import torch
from torch.nn.modules.loss import _Loss


class IOULoss(_Loss):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, prediction, labels):
        return iou(prediction, labels)


def iou(prediction, labels, smooth = 1e-6):
    """
    Intersection over union of two boxes.

    Parameters
    ----------
    prediction : torch.Tensor
        Labels of the batch.
    labels : torch.Tensor
        Labels of the batch.
    smooth : float, optional
        Smoothing factor to avoid division by zero.

    Returns
    -------
    float
        Mean IOU over the batch.
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    prediction = prediction.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (prediction & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (prediction | labels).float().sum((1, 2))         # Will be zero if both are 0

    # Avoid 0/0 
    iou = (intersection + smooth) / (union + smooth)
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return torch.mean(iou)
