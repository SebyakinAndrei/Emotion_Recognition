import os
import argparse
import torch
import numpy as np
from faceboxes.layers.functions.prior_box import PriorBox
from faceboxes.utils.nms_wrapper import nms
from faceboxes.models.faceboxes import FaceBoxes
from faceboxes.utils.box_utils import decode
from faceboxes.utils.timer import Timer
from faceboxes.data import cfg


this_dir, this_filename = os.path.split(__file__)

class args:
    trained_model = os.path.join(this_dir, 'faceboxes', 'weights', 'FaceBoxes.pth')
    save_folder = 'eval/'
    cuda = True
    cpu = True
    confidence_threshold = 0.1
    top_k = 5000
    nms_threshold = 0.3
    keep_top_k = 750


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Detector:
    def __init__(self, cuda=True):

        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        self.net = load_model(self.net, args.trained_model)

        if args.cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        print('Finished loading model!')

    def detect_faces(self, image, img_size=(640, 480)):
        """
        Face detection function
        :param image: opencv bgr image
        :return: 
        """

        self.net.eval()

        img = np.float32(image)
        #img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        if args.cuda:
            img = img.cuda()
            scale = scale.cuda()

        _t = {'forward_pass': Timer(), 'misc': Timer()}

        _t['forward_pass'].tic()
        out = self.net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()

        res = []
        for k in range(dets.shape[0]):
            # x, y, w, h

            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            ymin += 0.2 * (ymax - ymin + 1)
            score = dets[k, 4]

            res.append((xmin, ymin, xmax - xmin, ymax - ymin))
        return res
