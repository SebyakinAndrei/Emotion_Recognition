import time
import copy

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage

from random import randint, seed
from timeit import default_timer

from scipy.misc import imread, imresize
from imgaug import augmenters as iaa

from tensorboardX import SummaryWriter
from torchsummary import summary

from tqdm import tqdm

from utils.train_utils import *
from emotion_models import emotion_models


DATASET_DIR = '../datasets/AffNet/'
seed(0)

torch.multiprocessing.set_sharing_strategy('file_system')


class FaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataframe, subdir='train', transform=None, steps=None, imsize=128):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        droplist = [column for column in ['face_x', 'face_y', 'face_width',
                                          'face_height', 'facial_landmarks'] if column in dataframe.columns]
        self.labels = dataframe.drop(columns=droplist).drop(dataframe[dataframe['expression'] >= 8].index)
        self.transform = transform
        self.steps = steps
        self.subdir = subdir
        self.imsize = imsize
        print('Labels in dataset:', len(self.labels))

    def __len__(self):
        if self.steps is None:
            return len(self.labels)
        return self.steps

    def __getitem__(self, idx):
        if self.steps is not None:
            idx = randint(0, len(self.labels) - 1)

        img_name = DATASET_DIR + self.subdir + '/' + self.labels.iloc[idx]['subDirectory_filePath'].replace('/', '_')
        image = imresize(imread(img_name), (self.imsize, self.imsize), interp='bilinear')


        if self.transform is not None:
            image = self.transform(image)

        x = TF.to_tensor(image)
        x = TF.normalize(x, [0.5]*3, [0.5]*3, inplace=True)
        return x, torch.tensor(self.labels.iloc[idx]['expression'], dtype=torch.long)


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([iaa.GaussianBlur(sigma=(0, 0.4))]),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.3,
                          iaa.OneOf([iaa.AdditivePoissonNoise((4, 16)),
                                     iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255), per_channel=True)])),
            iaa.Sometimes(0.3, iaa.OneOf([iaa.SaltAndPepper((0.05, 0.1)), iaa.Dropout((0.05, 0.1))])),
            iaa.OneOf([iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True), iaa.ContrastNormalization((0.75, 1.5)),
                       iaa.GammaContrast((0.5, 1.75), per_channel=True)]),
            iaa.OneOf([iaa.Add((-40, 40), per_channel=True), iaa.Add((-40, 40), per_channel=False)]),
            #iaa.Sharpen(alpha=(0, 0.5))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def create_model(*args, arch='resnet18', debug=True):
    model = emotion_models[arch](*args).cuda()
    summary(model, input_size=(3, 128, 128))
    return model


to_pil = ToPILImage()

dataset_labels = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                      6: 'Anger', 7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-face'}


class ProgressIter:
    def __init__(self, iterable, enable_tqdm=True):
        self.iterable = iterable
        self.enable_tqdm = enable_tqdm

    def __iter__(self):
        if self.enable_tqdm:
            return iter(tqdm(self.iterable, total=len(self.iterable)))
        return iter(self.iterable)


def eval_model(model, optimizer, criterion, testloader, fasteval=False, steps_fasteval=1000, debug=False, enable_progress=True, confusion_matrix=False, normalized_cm=True):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@5', ':6.2f')

    model.eval()

    if confusion_matrix:
        y_true, y_pred = [], []

    with torch.no_grad():
        # Iterate over data.
        for step, (inputs, labels) in enumerate(ProgressIter(testloader, enable_progress)):
            if fasteval and (step == steps_fasteval):
                break

            if debug:
                inputs_cpu = inputs

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            preds = torch.max(outputs, 1)[1]

            if confusion_matrix:
                y_pred += preds.to(torch.int16).flatten().tolist()
                y_true += labels.to(torch.int16).flatten().tolist()
            loss = criterion(outputs, labels)

            if debug:
                to_pil(inputs_cpu[0]).save('debug/' + str(step) + '_' + dataset_labels[labels[0].item()] + '_' + dataset_labels[preds[0].item()] + '.png')
                del inputs_cpu

            # measure accuracy and record loss
            acc1, acc3 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.detach().item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc3[0], inputs.size(0))

            torch.cuda.empty_cache()
            del loss, inputs, outputs, labels, preds

    if confusion_matrix:
        from utils.train_utils import plot_confusion_matrix
        print('Plotting confusion matrix... Is normalized:', normalized_cm)
        # Plot normalized confusion matrix
        print('PRED (20 items):', y_pred[:20])
        print('GT (20 items):', y_true[:20])
        class_names = [x[1] for x in sorted(list((dataset_labels.items())), key=lambda x: x[0])][:8]
        print(class_names)
        fig = plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=normalized_cm,
                              title='Normalized confusion matrix' if normalized_cm else 'Confusion matrix')
        fig.set_size_inches(10, 10)
        print('Saving figure...')
        plt.savefig('confusion_mat.png', dpi='figure', format='png')
        print('Done!')

    return losses.avg, top1.avg, top3.avg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, criterion, optimizer, scheduler, trainloader, testloader, num_epochs=1, resume_from=None, enable_progress=True):
    since = time.time()

    if resume_from is not None:
        print('Loading weights...')
        model.load_state_dict(torch.load('models/train/emotion_model_epoch_{}.pth'.format(resume_from)))
        print('Done!')
    else:
        resume_from = 0

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    sw = SummaryWriter('logs')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, labels) in enumerate(ProgressIter(trainloader, enable_progress)):
            start = default_timer()

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.detach().item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc3[0], inputs.size(0))

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            del loss, inputs, outputs, labels, preds


        print('Validation time!')
        start = default_timer()

        cur_step = epoch + resume_from

        sw.add_scalar('train_loss', losses.avg, cur_step)
        sw.add_scalar('train_acc@1', top1.avg, cur_step)
        sw.add_scalar('train_acc@3', top3.avg, cur_step)

        print('Train Loss: {:.4f} Acc@1: {:.4f} Acc@3: {:.4f}'.format(losses.avg, top1.avg, top3.avg))

        val_loss, val_top1, val_top3 = eval_model(model, optimizer, criterion, testloader, fasteval=True, enable_progress=enable_progress)
        sw.add_scalar('fastval_loss', val_loss, cur_step)
        sw.add_scalar('fastval_acc@1', val_top1, cur_step)
        sw.add_scalar('fastval_acc@3', val_top3, cur_step)

        scheduler.step(val_loss)
        sw.add_scalar('learning_rate', get_lr(optimizer), cur_step)

        # deep copy the model
        if val_top1 > best_acc:
            best_acc = val_top1
            best_model = copy.deepcopy(model.state_dict())

        print('Done in', default_timer() - start)

        torch.save(model.state_dict(), 'models/train/emotion_model_epoch_{}.pth'.format(cur_step))
        torch.cuda.empty_cache()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_model


def main(resume_from=None, test_only=False, enable_progress=True, imsize=128, confusion_matrix=False, fasteval=True):
    if test_only:
        print('Test only mode enabled.')
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)#optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)

    print('Loading labels and loaders...')
    batch_size = 32
    if not test_only:
        df = pd.read_csv(DATASET_DIR + 'train_processed.csv', sep=',', engine='c')

        trainset = FaceDataset(df, transform=ImgAugTransform(), subdir='train', steps=5000*batch_size, imsize=imsize)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)

    df_val = pd.read_csv(DATASET_DIR + 'validation.csv', sep=',', engine='c')
    testset = FaceDataset(df_val, subdir='val', steps=len(df_val)*batch_size, imsize=imsize)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


    torch.cuda.empty_cache()

    if not test_only:
        print('Starting training..')
        model_new = train_model(model, criterion, optimizer, scheduler, trainloader, testloader, num_epochs=100, resume_from=resume_from, enable_progress=enable_progress)

        print('Saving model...')
        torch.save(model_new, 'models/emotion_modelv2.pth')
        print('Done.')
    else:
        print('Start testing...')
        model.load_state_dict(torch.load('models/emotion_model_new_bak.pth'))
        eval_model(model, optimizer, criterion, testloader, fasteval=fasteval, steps_fasteval=100, debug=True, enable_progress=enable_progress, confusion_matrix=confusion_matrix)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Video emotion recognizer.')
    parser.add_argument('--resume_from', type=int, help='Resume from epoch', default=None)
    parser.add_argument('--imsize', type=int, help='Image (w, h)', default=128)
    parser.add_argument('--no_console', action='store_true', help='Disable console output')
    parser.add_argument('--test_only', action='store_true', help='Disable console output')
    parser.add_argument('--conf_mat', action='store_true', help='Disable console output')
    parser.add_argument('--full_eval', action='store_true', help='Disable console output')
    args = parser.parse_args()
    print('Resuming from', args.resume_from)
    main(resume_from=args.resume_from, enable_progress=not args.no_console, imsize=args.imsize, test_only=args.test_only,
         confusion_matrix=args.conf_mat, fasteval=not args.full_eval)

