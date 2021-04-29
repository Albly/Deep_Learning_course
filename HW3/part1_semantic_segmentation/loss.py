import torch



def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)
    # number of images
    B = len(preds)
    # arrays for metrics
    intersection = torch.zeros([B, num_classes])    #None # TODO: calc intersection for each class
    union =        torch.zeros([B, num_classes])    #None # TODO: calc union for each class
    target =       torch.zeros([B, num_classes])    #None # TODO: calc number of pixels in groundtruth mask per class
    
    # Output shapes: B x num_classes
    # for each image
    for img in range(B):
        #for each class
        for cls_ in range(num_classes):
            #calculate metrics
            # where pred and mask belongs to the class
            intersection[img,cls_] = torch.sum((masks == cls_) & (preds[img] == cls_))
            # where pred or mask belongs to the class
            union[img,cls_] = torch.sum((masks == cls_)|(preds[img] == cls_))
            #number of pixels in mask belong to the class
            target[img,cls_] = torch.sum(masks == cls_)

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    # mean( I/U )
    mean_iou = torch.mean(intersection/(union+eps)).item()                  # TODO: calc mean class iou
    # mean( I/T )
    mean_class_rec = torch.mean(intersection/(target+eps)).item()           # TODO: calc mean class recall
    # sum(I) / sum(T) 
    mean_acc = (torch.sum(intersection)/(torch.sum(target)+eps)).item()     # TODO: calc mean accuracy

    return mean_iou, mean_class_rec, mean_acc