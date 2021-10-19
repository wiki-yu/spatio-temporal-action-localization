import time
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.utils import *
from cfg.cfg import parse_cfg
from core.region_loss import RegionLoss
from core.model import YOWO, get_fine_tuning_parameters
from datasets import cv2_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="ucf101-24", help="dataset")
    # parser.add_argument("--data_cfg", type=str, default="cfg/ucf24.data ", help="data_cfg")
    # parser.add_argument("--cfg_file", type=str, default="cfg/ucf24.cfg ", help="cfg_file")
    # parser.add_argument('--resume_path', default='backup/ucf24/yowo_ucf101-24_16f_best.pth', type=str, help='Continue training from pretrained (.pth)')
    # parser.add_argument("--n_classes", type=int, default=24, help="n_classes")
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
    parser.add_argument("--evaluate", type=bool, default=True, help="evaluate")
    parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
    parser.add_argument("--end_epoch", type=int, default=4, help="evaluate")
    opt = parser.parse_args()

    # Dataset to use
    dataset_use = opt.dataset
    assert dataset_use == 'ucsp' or dataset_use == 'ucf101-24', 'invalid dataset'

    # Configurations
    datacfg = opt.data_cfg  # path for dataset of training and validation
    cfgfile = opt.cfg_file  # path for cfg file
    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(cfgfile)[0]

    # Obtain list for training and testing
    basepath = data_options['base']
    trainlist = data_options['train']
    testlist = data_options['valid']
    backupdir = data_options['backup']

    # GPU parameters
    gpus = data_options['gpus']  # e.g. 0,1,2,3
    ngpus = len(gpus.split(','))
    num_workers = int(data_options['num_workers'])

    # Learning parameters
    nsamples = file_lines(trainlist)  # number of training samples
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

    # Training parameters
    max_epochs = max_batches * batch_size // nsamples + 1
    use_cuda = True
    seed = int(time.time())
    eps = 1e-5
    best_fscore = 0  # initialize best fscore
    
    # Testing parameters
    nms_thresh = 0.4
    iou_thresh = 0.5
    if not os.path.exists(backupdir):
        os.mkdir(backupdir)
    torch.manual_seed(seed)   # set random seed
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Create model
    model = YOWO(opt)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)  # in multi-gpu case
    model.seen = 0
    # logging(model)

    parameters = get_fine_tuning_parameters(model, opt)
    optimizer = optim.SGD(parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    print('Model path: ', opt.resume_path)

    # Load resume path if necessary
    if opt.resume_path:
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
    init_epoch = model.seen // nsamples
    # labelmap, _  = read_labelmap("datasets/ucf24/ucf24_action_list.pbtxt")

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

    def test(epoch):
        num_classes = region_loss.num_classes
        anchors = region_loss.anchors
        num_anchors = region_loss.num_anchors
        conf_thresh_valid = 0.2  # 0.005
        logging('Validation at epoch %d' % (epoch))
        model.eval()

        # Data preparation and inference 
        video_path = 'datasets/twoperson.mp4'
        # video_path = 'datasets/skating.mp4'
        cap = cv2.VideoCapture(video_path)
        count = 0
        queue = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            count += 1
            print('Count: ', count)

            if not ret:
                break
            if len(queue) <= 0: 
                for i in range(clip_duration):
                    queue.append(frame)
            else:
                queue.append(frame)
                queue.pop(0)
            # print(np.shape(queue))  # (16, 240, 320, 3)

            # Resize images
            imgs = [cv2_transform.resize(224, img) for img in queue]
            # print("frame size: ", np.shape(frame))
            
            
            imgs = [cv2_transform.HWC2CHW(img) for img in imgs]  # convert image to CHW keeping BGR order.
            imgs = [img / 255.0 for img in imgs]  # image [0, 255] -> [0, 1].
            imgs = [
                np.ascontiguousarray(
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs
            ]
            # print('imgs.shape1: ', np.shape(imgs))  # (16, 3, 224, 224)

            # Concat list of images to single ndarray.
            imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)
            imgs = np.ascontiguousarray(imgs)
            imgs = torch.from_numpy(imgs)
            imgs = torch.unsqueeze(imgs, 0)
            
            # Model inference
            with torch.no_grad():
                output = model(imgs).data
                preds = []
                # print('### model output shape: ', output.shape)  # [1, 425, 7, 7]
                all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

                # print('output.size(0): {} output.shape: {} '.format(output.size(0), output.shape))  # output.size(0): 1 output.shape: torch.Size([1, 145, 7, 7])
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    print('boxes.shape: ', np.shape(boxes))
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

                            # print(
                            #     str(int(box[6]) + 1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                            #         x2) + ' ' + str(y2) + '\n')
                            
                            preds.append([[x1,y1,x2,y2], prob])
            for dets in preds:
                x1 = int(dets[0][0])
                y1 = int(dets[0][1])
                x2 = int(dets[0][2])
                y2 = int(dets[0][3])
                cls_score = np.array(dets[1])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                blk   = np.zeros(frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                text.append("[{:.2f}] ".format(cls_score))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                coord.append((x1+3, y1+7+10))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
                            
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    logging('********Testing video************')
    test(0)
