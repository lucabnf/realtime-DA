    import argparse
from warnings import showwarning
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
from dataset.IDDA import IDDA
import torch.nn.functional as F
import os
from model.build_BiSeNet import BiSeNet
from model.FC_discriminator import FC_Discriminator
from model.TF_discriminator import TF_Discriminator
from model.DW_discriminator import DW_Discriminator
from model.FC_dropout_discriminator import FC_Dropout_Discriminator
import torch
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler, reverse_one_hot, \
    compute_global_accuracy, fast_hist, per_class_iu
from utils import prune_global
from loss import DiceLoss
import torch.cuda.amp as amp


# -------------------    Validation function    -------------------
def val(args, model, CamVid_dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(CamVid_dataloader):
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
def train(args, model, model_D, optimizer, optimizer_D, CamVid_dataloader_train, CamVid_dataloader_val,
          IDDA_dataloader):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.optimizer_D, args.context_path))

    scaler = amp.GradScaler()

    # Segmentation loss
    loss_func = DiceLoss()

    # Loss used for both discriminator training and segmentation training with target
    bce_loss = torch.nn.BCEWithLogitsLoss()

    step = 0

    # Start resuming information (if pretrained mode exists)
    epoch_start_i = args.epoch_start_i
    max_miou = args.max_miou
    if epoch_start_i != 0:
        print('Recovered epoch: ', epoch_start_i)
        print('Recovered max_miou: ', max_miou)

    for i_iter in range(epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=i_iter, max_iter=args.num_epochs)
        lr_D = poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=i_iter, max_iter=args.num_epochs)

        model.train()
        model_D.train()

        trainloader_iter = enumerate(IDDA_dataloader)
        targetloader_iter = enumerate(CamVid_dataloader_train)

        tq = tqdm(total=len(CamVid_dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_D %f' % (i_iter, lr, lr_D))
        loss_seg_record = []
        loss_ADV_record = []
        loss_D_record = []

        # labels for adversarial training
        source_label = 0
        target_label = 1

        for sub_i in range(len(CamVid_dataloader_train)):

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # ----------------    Train segmentation model    ------------------
            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            _, batch = trainloader_iter.__next__()
            data, label = batch
            data = data.cuda()
            label = label.long().cuda()

            with amp.autocast():
                output_source, output_sup1_source, output_sup2_source = model(data)

                loss1 = loss_func(output_source, label)
                loss2 = loss_func(output_sup1_source, label)
                loss3 = loss_func(output_sup2_source, label)
                loss_seg = loss1 + loss2 + loss3

            scaler.scale(loss_seg).backward()

            # train with target
            _, batch = targetloader_iter.__next__()
            data, label = batch
            data = data.cuda()

            with amp.autocast():
                output_target, _, _ = model(data)

                D_out = model_D(F.softmax(output_target))
                loss_adv_target = bce_loss(D_out,
                                           torch.FloatTensor(D_out.data.size())\
                                           .fill_(source_label).cuda())
                loss_adv_target = float(args.lambda_adv) * loss_adv_target

            scaler.scale(loss_adv_target).backward()

            # ----------------    Train discriminator model    ------------------
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            with amp.autocast():
                output_source = output_source.detach()

                D_out = model_D(F.softmax(output_source))
                loss_D = bce_loss(D_out,
                                  torch.FloatTensor(D_out.data.size())\
                                  .fill_(source_label).cuda())
                loss_D = loss_D / args.iter_size / 2

            scaler.scale(loss_D).backward()

            # train with target
            with amp.autocast():
                output_target = output_target.detach()

                D_out = model_D(F.softmax(output_target))
                loss_D = bce_loss(D_out,
                                  torch.FloatTensor(D_out.data.size())\
                                  .fill_(target_label).cuda())
                loss_D = loss_D / args.iter_size / 2

            scaler.scale(loss_D).backward()

            scaler.step(optimizer)
            scaler.step(optimizer_D)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss_seg='%.6f' % loss_seg)
            tq.set_postfix(loss_adv_target='%.6f' % loss_adv_target)
            tq.set_postfix(loss_D='%.6f' % loss_D)
            step += 1
            writer.add_scalar('loss_seg_step', loss_seg, step)
            writer.add_scalar('loss_adv_target', loss_adv_target)
            writer.add_scalar('loss_Disc', loss_D)

            loss_seg_record.append(loss_seg.item())
            loss_ADV_record.append(loss_adv_target.item())
            loss_D_record.append(loss_D.item())

        tq.close()
        loss_train_mean = np.mean(loss_seg_record)
        loss_ADV_train_mean = np.mean(loss_ADV_record)
        loss_D_train_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), i_iter)
        print('loss_seg for train : %f' % (loss_train_mean))
        print('loss_adv for train : %f' % (loss_ADV_train_mean))
        print('loss_D for train : %f' % (loss_D_train_mean))

        # -----------------------   pruning   ----------------------------------- 
        if args.dt_pruning == 'inc':
            p = i_iter * 0.005 # incremental sparsity
            model_D = prune_global(model_D, p)
        elif args.dt_pruning == 'fix':
            p = 0.2 # fixed sparsity
            model_D = prune_global(model_D, p)

        # -----------------------   checkpoint   -----------------------------------
        if i_iter % args.checkpoint_step == 0 and i_iter != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)

            state = {
                "epoch": i_iter,
                "max_miou": max_miou,
                "model_state_dict": model.module.state_dict(),
                "model_D_state_dict": model_D.module.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state,
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
            print('Checkpoint saved')

        # ----------------         validation step          ------------------------
        if i_iter % args.validation_step == 0 and i_iter != 0:

            precision, miou = val(args, model, CamVid_dataloader_val)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
                print("Best model updated. max_miou: ", max_miou)
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
    parser.add_argument('--learning_rate_D', type=float, default=1e-4, help='learning rate used for train')
    parser.add_argument('--data_CamVid', type=str, default='', help='path of training data')
    parser.add_argument('--data_IDDA', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--optimizer_D', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='lambda coefficient for adversarial loss')
    parser.add_argument("--iter-size", type=int, default=1, help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--epoch_start_i", type=int, default=0, help="Start counting epochs from this number.")
    parser.add_argument("--max_miou", type=float, default=0, help="Maximum value of miou achieved.")
    parser.add_argument('--discriminator', type=str, default='fcd', help='discriminator, support fcd, tfd, dwd, fcd_dropout')
    parser.add_argument('--dt_pruning', type=str, default=None, required=False, help='wheter to apply pruning during training on discriminator (fix or inc)')
    parser.add_argument('--info', default=False, action='store_true', required=False, help='print information about model parameters')
    args = parser.parse_args(params)

    # ---------------------    Datasets and dataloaders    ---------------------
    # CamVid
    CamVid_train_path = [os.path.join(args.data_CamVid, 'train'), os.path.join(args.data_CamVid, 'val')]
    CamVid_train_label_path = [os.path.join(args.data_CamVid, 'train_labels'),
                               os.path.join(args.data_CamVid, 'val_labels')]
    CamVid_test_path = os.path.join(args.data_CamVid, 'test')
    CamVid_test_label_path = os.path.join(args.data_CamVid, 'test_labels')
    CamVid_csv_path = os.path.join(args.data_CamVid, 'class_dict.csv')
    CamVid_dataset_train = CamVid(CamVid_train_path, CamVid_train_label_path, CamVid_csv_path,
                                  scale=(args.crop_height, args.crop_width),
                                  loss='dice', mode='train')

    CamVid_dataloader_train = DataLoader(
        CamVid_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    CamVid_dataset_val = CamVid(CamVid_test_path, CamVid_test_label_path, CamVid_csv_path,
                                scale=(args.crop_height, args.crop_width),
                                loss='dice', mode='test')

    CamVid_dataloader_val = DataLoader(
        CamVid_dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # IDDA
    IDDA_path = os.path.join(args.data_IDDA, 'rgb')
    IDDA_label_path = os.path.join(args.data_IDDA, 'labels')
    IDDA_json_path = os.path.join(args.data_IDDA, 'classes_info.json')
    IDDA_dataset = IDDA(IDDA_path, IDDA_label_path, IDDA_json_path, CamVid_csv_path,
                        scale=(args.crop_height, args.crop_width), loss='dice')

    IDDA_dataloader = DataLoader(
        IDDA_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Set if GPU ids are used for training
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # -------------------    Models building    -------------------
    # a) Segmentation model
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # b) Discriminator model
    if args.discriminator == "fcd": # Fully-convolutional discriminator
        model_D = FC_Discriminator(args.num_classes)
        if torch.cuda.is_available() and args.use_gpu:
            model_D = torch.nn.DataParallel(model_D).cuda()
    elif args.discriminator == "tfd": # Tucker-factorization discriminator
        model_D = TF_Discriminator(args.num_classes)
        if torch.cuda.is_available() and args.use_gpu:
            model_D = torch.nn.DataParallel(model_D).cuda()
    elif args.discriminator == "dwd": # Depth-wise discriminator
        model_D = DW_Discriminator(args.num_classes)
        if torch.cuda.is_available() and args.use_gpu:
            model_D = torch.nn.DataParallel(model_D).cuda()
    elif args.discriminator == "fcd_dropout": # Fully-convolutinal driscriminator with dropout 
        model_D = FC_Dropout_Discriminator(args.num_classes)
        if torch.cuda.is_available() and args.use_gpu:
            model_D = torch.nn.DataParallel(model_D).cuda()

    # ------------------- Parameters of initial model -------------------
    if args.info == True:
        print("Number of parameters of initial segmentation model\n")
        seg_total_params = sum(p.numel() for p in model.parameters())
        seg_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\t# total parameters=", seg_total_params)
        print("\t# trainable parameters=", seg_trainable_params)

        print("Number of parameters of initital discrimininator model\n")
        dis_total_params = sum(p.numel() for p in model_D.parameters())
        dis_trainable_params = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
        print("\t# total parameters=", dis_total_params)
        print("\t# trainable parameters=", dis_trainable_params)

    # -------------------    Optimizer building    -------------------
    # a) Optimizer for the segmentation network
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer for segmentation \n')
        return None

    # b) Optimizer for the discriminator network
    if args.optimizer_D == 'rmsprop':
        optimizer_D = torch.optim.RMSprop(model_D.parameters(), args.learning_rate_D)
    elif args.optimizer_D == 'sgd':
        optimizer_D = torch.optim.SGD(model_D.parameters(), args.learning_rate_D, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer_D == 'adam':
        optimizer_D = torch.optim.Adam(model_D.parameters(), args.learning_rate_D)
    else:  # rmsprop
        print('not supported optimizer for adversarial \n')
        return None

    # -------------------    Pre-trained model loading    -------------------
    if os.path.exists(args.pretrained_model_path):
        print('load model from %s ...' % args.pretrained_model_path)
        checkpoint = torch.load(args.pretrained_model_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        model_D.module.load_state_dict(checkpoint['model_D_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        args.epoch_start_i = checkpoint['epoch'] + 1
        args.max_miou = checkpoint['max_miou']
        print('Pre-trained model found and recovered!')

    # -------------------    train and final validation    -------------------
    train(args, model, model_D, optimizer, optimizer_D, CamVid_dataloader_train, CamVid_dataloader_val, IDDA_dataloader)
        
    val(args, model, CamVid_dataloader_val)

    # ------------------- Understanding during-training pruning and where it takes place -------------------
    if args.dt_pruning is not None:
            
        print("---PRUNED DISCRIMINATOR MODEL---\n")
        sums = [0]
        elements = [0]
        for name, module in model_D.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                sums.append(100.*float(torch.sum(module.weight == 0)))
                elements.append(float(module.weight.nelement()))
            if float(elements[-1]   ) != 0.0:
                print("Sparsity in {}.weight: {:.3f}%".format(name,sums[-1]/elements[-1]))

        sums = np.array(sums)
        elements = np.array(elements)
        print("Global sparsity: {:.2f}%".format(sums.sum()/elements.sum()))

    # --------------- Parameters of final models ---------------
    if args.info == True:
        print("Number of parameters of final segmentation model")
        seg_total_params = sum(p.numel() for p in model.parameters())
        seg_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\t# total parameters=", seg_total_params)
        print("\t# trainable parameters=", seg_trainable_params)

        print("Number of parameters of final discrimininator model")
        dis_total_params = sum(p.numel() for p in model_D.parameters())
        dis_trainable_params = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
        print("\t# total parameters=", dis_total_params)
        print("\t# trainable parameters=", dis_trainable_params)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--learning_rate_D', '1e-4',
        '--lambda_adv', '0.001',
        '--data_CamVid', './data/CamVid',
        '--data_IDDA', './data/IDDA',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './adv_checkpoints',
        '--context_path', 'resnet101',
        '--optimizer', 'sgd',
        '--optimizer_D', 'adam',
        '--checkpoint_step', '1',
        '--validation_step', '5',
        '--pretrained_model_path', './adv_checkpoints/latest_dice_loss.pth',
        '--discriminator', 'fcd',
        '--dt_pruning', 'inc'
    ]
    main(params)