import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import progress_bar, IOStream
from data import ModelNet40
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))

classes = {1: 'Eucalyptus_miniata', 2: 'Picea_abies', 3: 'Pinus_sylvestris', 4: 'Pseudotsuga_menziesii', 5: 'Quercus_robur', 6: 'Quercus_rubra', 7: 'Betula_pendula', 8: 'Fagus_sylvatica', 9: 'Fraxinus_excelsior', 10: 'Abies_alba', 11: 'Larix_decidua', 12: 'Acer_pseudoplatanus', 13: 'Carpinus_betulus', 14: 'Quercus_petraea', 15: 'Acer_campestre', 16: 'Prunus_avium', 17: 'Pinus_nigra', 18: 'Pinus_pinaster', 19: 'Quercus_faginea', 20: 'Quercus_ilex', 21: 'Pinus_contorta', 22: 'Populus_deltoides', 23: 'Populus_tremuloides', 24: 'Acer_saccharum', 25: 'Pinus_resinosa', 26: 'Corylus_avellana', 27: 'Pinus_radiata', 28: 'Crataegus_monogyna', 29: 'Picea_glauca', 30: 'Euonymus_europaeus', 31: 'Fraxinus_angustifolia', 32: 'Tilia_cordata', 33: 'Ulmus_laevis'}

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='model31A', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')

    # Voting evaluation, referring: https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/eval_voting.py
    parser.add_argument('--NUM_REPEAT', type=int, default=10)
    parser.add_argument('--NUM_VOTE', type=int, default=10)

    parser.add_argument('--validate', action='store_true', help='Validate the original testing result.')
    parser.add_argument('--use_avg_instead', action='store_true')
    parser.add_argument("--add_noise", action="store_true")
    return parser.parse_args()


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.01):
    pointcloud = pointcloud.numpy()
    B, N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip).astype(np.float32)
    return torch.from_numpy(pointcloud)


def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    print(f"random seed is set to {args.seed}, the speed will slow down.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    if args.msg is None:
        message = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    else:
        message = "-" + args.msg
    #args.checkpoint = 'checkpoints/' + args.model + message

    print('==> Preparing data..')
    dset = ModelNet40(partition='test', num_points=args.num_points)
    print(f"# of test objects: {len(dset)}")
    test_loader = DataLoader(dset, num_workers=4,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    if args.use_avg_instead:
        checkpoint_path = os.path.join(args.checkpoint, 'best_avg_checkpoint.pth')
    else:
        checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    args.validate = False # added
    if args.validate:
        test_out = validate(net, test_loader, criterion, device, args)
        print(f"Vanilla out: {test_out}")
        print(f"Note 1: Please also load the random seed parameter (if forgot, see out.txt).\n"
              f"Note 2: This result may vary little on different GPUs (and number of GPUs), we tested 2080Ti, P100, and V100.\n"
              f"[note : Original result is achieved with V100 GPUs.]\n\n\n")
        # Interestingly, we get original best_test_acc on 4 V100 gpus, but this model is trained on one V100 gpu.
        # On different GPUs, and different number of GPUs, both OA and mean_acc vary a little.
        # Also, the batch size also affect the testing results, could not understand.

    print(f"===> start voting evaluation...")
    voting(net, test_loader, device, args)


def validate(net, testloader, criterion, device, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            if args.add_noise:
                data = jitter_pointcloud(data)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            try:
                total += label.size(0)
            except:
                total += 1
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


def voting(net, testloader, device, args):
    if not args.add_noise:
        name = '/evaluate_voting' + str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S')) + 'seed_' + str(
            args.seed) + '.log'
    else:
        name = '/evaluate_voting' + str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S')) + 'seed_' + str(
        args.seed) + "_with_noise" + '.log'
    io = IOStream(args.checkpoint + name)
    io.cprint(str(args))

    net.eval()
    best_acc = 0
    best_mean_acc = 0
    # pointscale = PointcloudScale(scale_low=0.8, scale_high=1.18)  # set the range of scaling
    # pointscale = PointcloudScale()
    pointscale = PointcloudScale(scale_low=0.85, scale_high=1.15)

    for i in range(args.NUM_REPEAT):
        test_true = []
        test_pred = []

        for batch_idx, (data, label) in enumerate(tqdm(testloader)):
            if args.add_noise:
                data = jitter_pointcloud(data)
            data, label = data.to(device), label.to(device).squeeze()
            pred = 0
            for v in range(args.NUM_VOTE):
                new_data = data
                # batch_size = data.size()[0]
                if v > 0:
                    new_data.data = pointscale(new_data.data)
                with torch.no_grad():
                    pred += F.softmax(net(new_data.permute(0, 2, 1)), dim=1)  # sum 10 preds
            pred /= args.NUM_VOTE  # avg the preds!
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = 100. * metrics.accuracy_score(test_true, test_pred)
        test_mean_acc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_mean_acc > best_mean_acc:
            best_mean_acc = test_mean_acc
        outstr = 'Voting %d, test acc: %.3f, test mean acc: %.3f,  [current best(all_acc: %.3f mean_acc: %.3f)]' % \
                 (i, test_acc, test_mean_acc, best_acc, best_mean_acc)
        io.cprint(outstr)

    final_outstr = 'Final voting test acc: %.6f,' % (best_acc * 100)
    io.cprint(final_outstr)

if __name__ == '__main__':
    main()
