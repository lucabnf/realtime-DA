import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import torch.cuda.amp as amp

# -------------------    Validation function    -------------------
def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)
        
        precision = np.mean(precision_record)

        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)

        return precision, miou

# -------------------    Training function    -------------------
def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()

    loss_func = DiceLoss()

    step = 0

    # Start resuming information (if pretrained mode exists)
    epoch_start_i = args.epoch_start_i
    max_miou = args.max_miou
    if epoch_start_i != 0:
        print('Recovered epoch: ', epoch_start_i)
        print('Recovered max_miou: ', max_miou)

    for i_iter in range(epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=i_iter, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (i_iter, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):

            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), i_iter)
        print('loss for train : %f' % (loss_train_mean))

        # --------------------    saving checkpoint    --------------------
        if i_iter % args.checkpoint_step == 0 and i_iter != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)

            state = {
                "epoch": i_iter,
                "max_miou": max_miou,
                "model_state_dict": model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state,
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
            print('Checkpoint saved')

        # --------------------    validation step    --------------------
        if i_iter % args.validation_step == 0 and i_iter != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
                print("Found a better model. Best model updated --> max_miou: ", max_miou)
            writer.add_scalar('epoch/precision_val', precision, i_iter)
            writer.add_scalar('epoch/miou_val', miou, i_iter)


def main(params):

    # --------------------    basic parameters    --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--max_miou', type=float, default=0, help="Maximum value of miou achieved.")

    args = parser.parse_args(params)

    # ---------------------    CamVid Dataset and dataloader    ---------------------
    train_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val')]
    train_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels')]
    test_path = os.path.join(args.data, 'test')
    test_label_path = os.path.join(args.data, 'test_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                           loss='dice', mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        )
    dataset_val = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                         loss='dice', mode='test')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # -------------------    Models building    -------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # -------------------    Optimizer building    -------------------
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # -------------------    Pre-trained model loading    -------------------
    if os.path.exists(args.pretrained_model_path):
        print('load model from %s ...' % args.pretrained_model_path)
        checkpoint = torch.load(args.pretrained_model_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.epoch_start_i = checkpoint['epoch'] + 1
        args.max_miou = checkpoint['max_miou']
        print('Pre-trained model found and recovered!')

    # -------------------    Train and (final) validation    -------------------
    train(args, model, optimizer, dataloader_train, dataloader_val)

    val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--data', './data/CamVid',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './seg_checkpoints',
        '--context_path', 'resnet18',
        '--optimizer', 'sgd',
        '--checkpoint_step', '5',
        '--pretrained_model_path', './seg_checkpoints/latest_dice_loss.pth'
    ]
    main(params)
