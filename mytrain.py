import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  transforms

import datasets.dataset
from core.utils import *
from cfg.cfg import parse_cfg
from core.region_loss import RegionLoss
from core.model import YOWO, get_fine_tuning_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucsp", help="dataset")
    parser.add_argument("--data_cfg", type=str, default="cfg/ucsp.data ", help="data_cfg")
    parser.add_argument("--cfg_file", type=str, default="cfg/ucsp.cfg ", help="cfg_file")
    parser.add_argument('--resume_path', default='backup/ucsp/yowo_ucsp_16f_best.pth', type=str, help='Continue training from pretrained (.pth)')
    parser.add_argument("--n_classes", type=int, default=4, help="n_classes")
    parser.add_argument("--backbone_3d", type=str, default="resnext101", help="backbone_3d")
    parser.add_argument("--backbone_3d_weights", type=str, default="weights/resnext-101-kinetics.pth", help="backbone_3d_weights")
    parser.add_argument("--backbone_2d", type=str, default="darknet", help="backbone_3d_weights")
    parser.add_argument("--backbone_2d_weights", type=str, default="weights/yolo.weights", help="backbone_2d_weights")
    parser.add_argument("--freeze_backbone_2d", type=bool, default=True, help="freeze_backbone_2d")
    parser.add_argument("--freeze_backbone_3d", type=bool, default=True, help="freeze_backbone_3d")
    parser.add_argument("--evaluate", type=bool, default=False, help="evaluate")
    parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
    parser.add_argument("--end_epoch", type=int, default=2, help="end_epoch")
    opt = parser.parse_args()

    # Assert dataset to use
    dataset_use = opt.dataset
    assert dataset_use == 'ucsp', 'invalid dataset'

    # Path for training and validation dataset configuration
    datacfg = opt.data_cfg
    # Path for DL cfg file
    cfgfile = opt.cfg_file
    data_options = read_data_cfg(datacfg)
    # print('data_options: ', data_options)
    net_options = parse_cfg(cfgfile)[0]
    # print('net_options: ', net_options)
    # Obtain list for training and testing
    basepath = data_options['base']
    trainlist = data_options['train']
    testlist = data_options['valid']
    backupdir = data_options['backup']

    # Number of training samples
    nsamples = file_lines(trainlist)
    gpus = data_options['gpus']  # e.g. 0,1,2,3
    ngpus = len(gpus.split(','))
    num_workers = int(data_options['num_workers'])
    batch_size = int(net_options['batch'])
    clip_duration = int(net_options['clip_duration'])
    max_batches = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum = float(net_options['momentum'])
    decay = float(net_options['decay'])
    steps = [float(step) for step in net_options['steps'].split(',')]
    scales = [float(scale) for scale in net_options['scales'].split(',')]

    # Loss parameters
    loss_options = parse_cfg(cfgfile)[1]
    region_loss = RegionLoss()
    anchors = loss_options['anchors'].split(',')
    region_loss.anchors = [float(i) for i in anchors]
    region_loss.num_classes = int(loss_options['classes'])
    region_loss.num_anchors = int(loss_options['num'])
    region_loss.anchor_step = len(region_loss.anchors) // region_loss.num_anchors
    region_loss.object_scale = float(loss_options['object_scale'])
    region_loss.noobject_scale = float(loss_options['noobject_scale'])
    region_loss.class_scale = float(loss_options['class_scale'])
    region_loss.coord_scale = float(loss_options['coord_scale'])
    region_loss.batch = batch_size
    print('Region loss num_classes: {}, num_anchors: {}, anchor_step: {}, batch_size: {}'.format(
        region_loss.num_classes, region_loss.num_anchors, region_loss.anchor_step, batch_size))
    print('Region loss object_scale: {}, noobject_scale: {}, class_scale: {}, coord_scale: {}'.format(
        region_loss.object_scale, region_loss.noobject_scale, region_loss.class_scale, region_loss.coord_scale))

    # Train parameters
    max_epochs = max_batches * batch_size // nsamples + 1
    print('max_epochs: {}, max_batches: {}, batch_size: {},  nsamples: {}'.format(
        max_epochs, max_batches, batch_size, nsamples))
    use_cuda = True
    seed = int(time.time())
    eps = 1e-5
    best_fscore = 0  # initialize best fscore

    # Test parameters
    nms_thresh = 0.4
    iou_thresh = 0.5

    if not os.path.exists(backupdir):
        os.mkdir(backupdir)

    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Create model
    model = YOWO(opt)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)  # in multi-gpu case
    model.seen = 0
    # print(model)
    parameters = get_fine_tuning_parameters(model, opt)
    optimizer = optim.SGD(parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    # Load resume path if necessary
    if opt.resume_path:
        print('Loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        opt.begin_epoch = checkpoint['epoch']
        best_fscore = checkpoint['fscore']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.seen = checkpoint['epoch'] * nsamples
        print("Loaded model fscore: ", checkpoint['fscore'])

    region_loss.seen = model.seen
    processed_batches = model.seen // batch_size
    init_width = int(net_options['width'])
    init_height = int(net_options['height'])

    def adjust_learning_rate(optimizer, batch):
        lr = learning_rate
        for i in range(len(steps)):
            scale = scales[i] if i < len(scales) else 1
            if batch >= steps[i]:
                lr = lr * scale
                if batch == steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / batch_size
        return lr

    def train(epoch):
        global processed_batches
        t0 = time.time()
        cur_model = model.module
        region_loss.l_x.reset()
        region_loss.l_y.reset()
        region_loss.l_w.reset()
        region_loss.l_h.reset()
        region_loss.l_conf.reset()
        region_loss.l_cls.reset()
        region_loss.l_total.reset()

        # Prepare the training dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.dataset.listDataset(basepath, trainlist, dataset_use=dataset_use, shape=(init_width, init_height),
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]),
                                train=True,
                                seen=cur_model.seen,
                                batch_size=batch_size,
                                clip_duration=clip_duration,
                                num_workers=num_workers),
            batch_size=batch_size, shuffle=False, **kwargs)
        lr = adjust_learning_rate(optimizer, processed_batches)
        logging('training at epoch %d, lr %f' % (epoch, lr))
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(optimizer, processed_batches)
            processed_batches = processed_batches + 1
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            output = model(data)
            region_loss.seen = region_loss.seen + data.data.size(0)
            loss = region_loss(output, target)
            loss.backward()
            optimizer.step()
            # save result every 1000 batches
            if processed_batches % 500 == 0:  # From time to time, reset averagemeters to see improvements
                region_loss.l_x.reset()
                region_loss.l_y.reset()
                region_loss.l_w.reset()
                region_loss.l_h.reset()
                region_loss.l_conf.reset()
                region_loss.l_cls.reset()
                region_loss.l_total.reset()
        t1 = time.time()
        logging('trained with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
        print('')

    def test(epoch):
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        test_loader = torch.utils.data.DataLoader(
            datasets.dataset.listDataset(basepath, testlist, dataset_use=dataset_use, shape=(init_width, init_height),
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]), train=False),
            batch_size=batch_size, shuffle=False, **kwargs)

        num_classes = region_loss.num_classes
        anchors = region_loss.anchors
        num_anchors = region_loss.num_anchors
        conf_thresh_valid = 0.005
        total = 0.0
        proposals = 0.0
        correct = 0.0
        fscore = 0.0
        correct_classification = 0.0
        total_detected = 0.0
        nbatch = file_lines(testlist) // batch_size
        logging('validation at epoch %d' % (epoch))
        model.eval()
        for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                output = model(data).data
                all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
 
                    for box in boxes:
                        x1 = round(float(box[0] - box[2] / 2.0) * 320.0)
                        y1 = round(float(box[1] - box[3] / 2.0) * 240.0)
                        x2 = round(float(box[0] + box[2] / 2.0) * 320.0)
                        y2 = round(float(box[1] + box[3] / 2.0) * 240.0)
                        
                        det_conf = float(box[4])
                        for j in range((len(box) - 5) // 2):
                            cls_conf = float(box[5 + 2 * j].item())
                            if type(box[6 + 2 * j]) == torch.Tensor:
                                cls_id = int(box[6 + 2 * j].item())
                            else:
                                cls_id = int(box[6 + 2 * j])
                            prob = det_conf * cls_conf

                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)
                    total = total + num_gts
                    for i in range(len(boxes)):
                        if boxes[i][4] > 0.25:
                            proposals = proposals + 1
                    for i in range(num_gts):
                        box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        best_iou = 0
                        best_j = -1
                        for j in range(len(boxes)):
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > best_iou:
                                best_j = j
                                best_iou = iou

                        if best_iou > iou_thresh:
                            total_detected += 1
                            if int(boxes[best_j][6]) == box_gt[6]:
                                correct_classification += 1
                        if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                            correct = correct + 1
                precision = 1.0 * correct / (proposals + eps)
                recall = 1.0 * correct / (total + eps)
                fscore = 2.0 * precision * recall / (precision + recall + eps)
                logging(
                    "[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))
        classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
        locolization_recall = 1.0 * total_detected / (total + eps)
        print("Classification accuracy: %.3f" % classification_accuracy)
        print("Locolization recall: %.3f" % locolization_recall)
        return fscore
    
    if opt.evaluate:
        logging('evaluating ...')
        test(0)
    else:
        print('###begin_epoch: {}, end_epoch: {}'.format(opt.begin_epoch, opt.end_epoch))
        for epoch in range(opt.begin_epoch, opt.end_epoch + 1):
            # Train the model for 1 epoch
            train(epoch)
            # Validate the model
            fscore = test(epoch)
            print('Current best fscore: {}, fscore now: {}'.format(best_fscore, fscore))
            is_best = fscore > best_fscore
            if is_best:
                print("New best fscore is achieved: ", fscore)
                best_fscore = fscore

            # Save the model to backup directory
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fscore': fscore
            }
            save_checkpoint(state, is_best, backupdir, opt.dataset, clip_duration)
            logging('!!!Weights are saved to backup directory: %s' % (backupdir))
