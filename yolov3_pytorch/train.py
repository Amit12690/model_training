from __future__ import division
import matplotlib as mpl
mpl.use('TkAgg')

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def delete_old_ckpts(folder_path,extension='pth'):
    import glob    
    files = glob.glob(folder_path+'/*.'+extension)
    for f in files:
        os.remove(f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--checkpoint_folder", type=str, default='checkpoints', help="Folder to save the trained model")
    parser.add_argument("--clear_old_ckpt", type=str, default='true', help="Clear old checkpoints when a better accuracy one is found")    
    opt = parser.parse_args()
    print(opt)

    checkpoint_folder = opt.checkpoint_folder + '/trained_checkpoints'
    logger = Logger(opt.checkpoint_folder+"/logs")
    map_log_txt = checkpoint_folder +'/mAP_log.txt'
    f1_log_txt = checkpoint_folder +'/Fscore_log.txt'
    err_log_txt = checkpoint_folder + '/error_log.txt'
    best_map_log_txt = checkpoint_folder +'/mAP_log_best.txt'
    best_f1_log_txt = checkpoint_folder +'/Fscore_log_best.txt'
    best_err_log_txt = checkpoint_folder + '/error_log_best.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_folder, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_mAP= -1
    best_f1 = -1
    least_loss = 99999999999

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:

            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            print(f"---- best mAP {best_mAP}")
            print(f"---- f1 {f1.mean()}")
            print(f"---- best f1 {best_f1}")

            map_log = open(map_log_txt, 'a')
            if AP.mean() >= best_mAP:
                best_mAP = max(best_mAP,AP.mean())
                torch.save(model.state_dict(), f"%s/ckpt_bestMAP.pth" % checkpoint_folder)
                best_map_log = open(best_map_log_txt, 'w')
                best_map_log.write("{},{}\n".format(epoch, best_mAP))
                best_map_log.close()
            map_log.write("{},{}\n".format(epoch, AP.mean()))
            map_log.close()

            f1_log = open(f1_log_txt, 'a')
            if f1.mean() >= best_f1:
                best_f1 = max(best_f1, f1.mean())
                torch.save(model.state_dict(), f"%s/ckpt_bestFscore.pth"%checkpoint_folder)
                best_f1_log = open(best_f1_log_txt, 'w')
                best_f1_log.write("{},{}\n".format(epoch, best_f1))
                best_f1_log.close()
            f1_log.write("{},{}\n".format(epoch, f1.mean()))
            f1_log.close()

            err_log = open(err_log_txt, 'a')
            if loss.item() <= least_loss:
                least_loss = min(least_loss, loss.item())
                torch.save(model.state_dict(), f"%s/ckpt_leastLoss.pth"%checkpoint_folder)
                best_err_log = open(best_err_log_txt, 'w')
                best_err_log.write("{},{}\n".format(epoch, least_loss))
                best_err_log.close()
            err_log.write("{},{}\n".format(epoch, loss.item()))
            err_log.close()

        #if epoch % opt.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), f"%s/ckpt_epoch_%d.pth" %(checkpoint_folder, epoch))

    err_log.close()
    f1_log.close()
    map_log.close()
    print("Trained checkpoints stored in : {}".format(checkpoint_folder))
