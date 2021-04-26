import torch



def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)

    B = len(preds)
    intersection = torch.zeros([B, num_classes])    #None # TODO: calc intersection for each class
    union =        torch.zeros([B, num_classes])    #None # TODO: calc union for each class
    target =       torch.zeros([B, num_classes])    #None # TODO: calc number of pixels in groundtruth mask per class
    

    # Output shapes: B x num_classes
    for img in range(B):
        for cls_ in range(num_classes):
            intersection[img,cls_] = ((masks == cls_) & (preds[img] == cls_)).sum()
            union[img,cls_] = ((masks == cls_) | (preds[img] == cls_)).sum()
            target[img,cls_] = (masks == cls_).sum()


    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    mean_iou = torch.mean(intersection/(union+eps)).item()                  # TODO: calc mean class iou
    mean_class_rec = torch.mean(intersection/(target+eps)).item()           # TODO: calc mean class recall
    mean_acc = (torch.sum(intersection)/(torch.sum(target)+eps)).item()     # TODO: calc mean accuracy

    return mean_iou, mean_class_rec, mean_acc