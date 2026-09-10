import torch
import numpy as np
import os
from torch.optim.lr_scheduler import MultiStepLR
import config
import engine
from Dataset_loaders.dataset import DataLoader
from Utils.utils import compute_criteria
from Loss.loss_r import Neg_Pearson
from Utils.model_utils import save_model_checkpoint, setup_seed
from Models import diff_model
import sys
import argparse
from dataset_train_test_split import train_test_split
os.chdir(sys.path[0])


def run_training(lr, dim, depth, heads, mlpdim):

    setup_seed(999)

    if torch.cuda.is_available():
        print("GPU available... Using GPU")
    else:
        print("GPU not available, using CPU")

    input_path = f'/home/anonymous/dataset/{args.dataset}_MST/'

    TRAIN_CHK_PATH = f"./Checkpoint/{args.dataset}/train"
    BEST_CHK_PATH = f"./Checkpoint/{args.dataset}/best"
    os.makedirs(TRAIN_CHK_PATH,exist_ok=True)
    os.makedirs(BEST_CHK_PATH,exist_ok=True)

    model_configs = f'FreqPhys_{args.dataset}_Epoch{args.EPOCHS}_LRate{lr}_Dim{dim}_Depth{depth}_TotalSteps{config.Total_steps}_Ktimes{config.K}'
    if args.dataset == "VIPL":
        model_configs = f"Fold_{config.VIPL_Fold}_" + model_configs

    os.makedirs(f'./logs/{args.dataset}', exist_ok=True)
    log_file = f"./logs/{args.dataset}/{model_configs}.txt"
    print('log file path: ' + log_file)

    with open(log_file, 'w') as f:
        f.writelines([f"model=FreqPhys, Dataset={args.dataset}, Train_bs={args.Train_batchsize}, learning_rate={lr}, k_times={config.K},  total_steps={config.Total_steps}\n",
                      "==============================\n"])
        f.flush()

    train, val, test, transform_train, transform_val, transform_test = train_test_split(args.dataset, args.fold, input_path)

    dataset_train = DataLoader(train, args.dataset, transforms=transform_train, data_aug=False)
    dataset_val = DataLoader(val, args.dataset, transforms=transform_val)
    dataset_test = DataLoader(test, args.dataset, transforms=transform_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.Train_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=args.Val_batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.Test_batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    model = diff_model.main(
        image_height=63,
        image_width=300,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlpdim,
        dropout=0.1,
        emb_dropout=0.1,
        is_test = False
    )

    model.cuda()
    model=torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    lossfunc_mse = torch.nn.MSELoss()
    lossfunc_pearson = Neg_Pearson(downsample_mode=0)

    val_min_mae = 1e3
    val_min_rmse = 1e3
    val_max_r = -1e3
    test_min_mae = 1e3
    test_min_rmse = 1e3
    test_max_r = -1e3

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_metrics_per_epoch = []

    test_loss_per_epoch = []
    test_metrics_per_epoch = []

    val_list = [x.split('/')[-1] for x in val]
    test_list = [x.split('/')[-1] for x in test]

    for epoch in range(args.EPOCHS):

        # training
        _, _, _, train_loss_per_batch = engine.train_fn(model, train_loader, optimizer, lossfunc_pearson, lossfunc_mse)
        epoch_model_file_name = f"{epoch}_of_{model_configs}.pt"
        last_epoch_model_file_name = f"{epoch-1}_of_{model_configs}.pt"
        print(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name),"this is last epoch model")
        if os.path.exists(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name)):
            os.remove(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name))
        save_model_checkpoint(model, epoch_model_file_name,checkpoint_path=TRAIN_CHK_PATH)
        
        train_loss_per_epoch.append(np.mean(train_loss_per_batch))

        with open(log_file, "a") as f:
            f.writelines([f"\nTraining! [Epoch: {epoch + 1}/{args.EPOCHS}]",
                            "\nTraining Loss: {:.3f} | \n".format(train_loss_per_epoch[-1]),
                            ])
            f.flush()

        # validation
        model_val_temp = diff_model.main(
            image_height=63,
            image_width=300,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlpdim,
            dropout=0.1,
            emb_dropout=0.1,
            is_test=True
        )
        model_val_temp.load_state_dict(torch.load(os.path.join(TRAIN_CHK_PATH , epoch_model_file_name))['model_state_dict'])

        model_val_temp.cuda()
        model_val_temp = torch.nn.DataParallel(model_val_temp)
        val_target_hr_per_batch, val_predicted_hr_per_batch, val_target_bvp_per_batch, val_predicted_bvp_per_batch, val_loss_per_batch, path_per_batch = engine.val_fn(
            model_val_temp, val_loader, lossfunc_pearson, lossfunc_mse, args.dataset)
        val_loss_per_epoch.append(np.mean(val_loss_per_batch))
        val_target_hr_per_batch = np.array(val_target_hr_per_batch)
        val_predicted_hr_per_batch = np.array(val_predicted_hr_per_batch)

        target_hr_per_video = []
        predicted_hr_per_video = []

        # print(val_list)

        for x in val_list:
            sum_target_hr_per_map = 0
            sum_predicted_hr_per_map = 0
            count = 0
            for i, y in enumerate(path_per_batch):
                if x in y:
                    count += 1
                    sum_target_hr_per_map += val_target_hr_per_batch[i]
                    sum_predicted_hr_per_map += val_predicted_hr_per_batch[i]
            if count == 0:
                print(x)
            target_hr_per_video.append(round(sum_target_hr_per_map / count))
            predicted_hr_per_video.append(round(sum_predicted_hr_per_map / count))

        val_metrics = compute_criteria(np.array(target_hr_per_video), np.array(predicted_hr_per_video))
        val_metrics_per_epoch.append(val_metrics)
        
        # for x in range(len(val_list)):
        #     print(f'{val_list[x]}, gt_hr:{target_hr_per_video[x]}, pred_hr:{predicted_hr_per_video[x]}, mae:{target_hr_per_video[x]-predicted_hr_per_video[x]}')
        # print(np.array(target_hr_per_video)-np.array(predicted_hr_per_video))

        val_metrics = compute_criteria(np.array(target_hr_per_video), np.array(predicted_hr_per_video))
        val_metrics_per_epoch.append(val_metrics)
        scheduler.step(val_metrics["RMSE"])

        with open(log_file, "a") as f:
            f.writelines([f"Validating! [Epoch: {epoch + 1}/{args.EPOCHS}]",
                            "\nValidating Loss: {:.3f} |".format(val_loss_per_epoch[-1]),
                            "HR_MAE : {:.3f} |".format(val_metrics["MAE"]),
                            "HR_RMSE : {:.3f} |".format(val_metrics["RMSE"]),
                            "HR_MER: {:.3f} |".format(val_metrics["MER"]),
                            "HR_r: {:.3f} |".format(val_metrics["r"]),
                            "HR_std: {:.3f} |\n".format(val_metrics["std"])
                            ])
            f.flush()


        # testing
        model_test_temp = diff_model.main(
            image_height=63,
            image_width=300,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlpdim,
            dropout=0.1,
            emb_dropout=0.1,
            is_test=True
        )
        model_test_temp.load_state_dict(torch.load(os.path.join(TRAIN_CHK_PATH , epoch_model_file_name))['model_state_dict'])

        model_test_temp.cuda()
        model_test_temp = torch.nn.DataParallel(model_test_temp)
        test_target_hr_per_batch, test_predicted_hr_per_batch, test_target_bvp_per_batch, test_predicted_bvp_per_batch, test_loss_per_batch, path_per_batch = engine.test_fn(
            model_test_temp, test_loader, lossfunc_pearson, lossfunc_mse, args.dataset)
        test_loss_per_epoch.append(np.mean(test_loss_per_batch))
        test_target_hr_per_batch = np.array(test_target_hr_per_batch)
        test_predicted_hr_per_batch = np.array(test_predicted_hr_per_batch)

        target_hr_per_video = []
        predicted_hr_per_video = []

        # print(test_list)

        for x in test_list:
            sum_target_hr_per_map = 0
            sum_predicted_hr_per_map = 0
            count = 0
            for i, y in enumerate(path_per_batch):
                if x in y:
                    count += 1
                    sum_target_hr_per_map += test_target_hr_per_batch[i]
                    sum_predicted_hr_per_map += test_predicted_hr_per_batch[i]
            if count == 0:
                print(x)
            target_hr_per_video.append(round(sum_target_hr_per_map / count))
            predicted_hr_per_video.append(round(sum_predicted_hr_per_map / count))

        test_metrics = compute_criteria(np.array(target_hr_per_video), np.array(predicted_hr_per_video))
        test_metrics_per_epoch.append(test_metrics)
        
        # for x in range(len(test_list)):
        #     print(f'{test_list[x]}, gt_hr:{target_hr_per_video[x]}, pred_hr:{predicted_hr_per_video[x]}, mae:{target_hr_per_video[x]-predicted_hr_per_video[x]}')
        # print(np.array(target_hr_per_video)-np.array(predicted_hr_per_video))

        test_metrics = compute_criteria(np.array(target_hr_per_video), np.array(predicted_hr_per_video))

        with open(log_file, "a") as f:
            f.writelines([f"Testing! [Epoch: {epoch + 1}/{args.EPOCHS}]",
                            "\nTesting Loss: {:.3f} |".format(test_loss_per_epoch[-1]),
                            "HR_MAE : {:.3f} |".format(test_metrics["MAE"]),
                            "HR_RMSE : {:.3f} |".format(test_metrics["RMSE"]),
                            "HR_MER: {:.3f} |".format(test_metrics["MER"]),
                            "HR_r: {:.3f} |".format(test_metrics["r"]),
                            "HR_std: {:.3f} |\n".format(test_metrics["std"])
                            ])
            f.flush()


        if len(val_loss_per_epoch) > 0 and val_min_rmse >= val_metrics["RMSE"]:
            RMSE_metric = val_metrics["RMSE"]
            best_model_file_name = f"RMSE({RMSE_metric:.3f})_{epoch}_of_{model_configs}.pt"
            best_model_files = os.listdir(BEST_CHK_PATH)
            for file in best_model_files:
                if model_configs in file:
                    file_path = os.path.join(BEST_CHK_PATH, file)
                    os.remove(file_path)
            save_model_checkpoint(model, best_model_file_name, checkpoint_path=BEST_CHK_PATH)
            val_min_rmse = val_metrics['RMSE']
            val_min_mae = val_metrics['MAE']
            val_max_r = val_metrics['r']
            test_min_rmse = test_metrics['RMSE']
            test_min_mae = test_metrics['MAE']
            test_max_r = test_metrics['r']


    with open(log_file, "a") as f:
        f.writelines([f"VAL MIN MAE OF {args.EPOCHS} epochs: {val_min_mae}  ",
                      f"VAL MIN RMSE OF {args.EPOCHS} epochs: {val_min_rmse}  "
                        f"VAL MAX r OF {args.EPOCHS} epochs: {val_max_r}\n"])
        f.writelines([f"TEST MIN MAE OF {args.EPOCHS} epochs: {test_min_mae}  ",
                      f"TEST MIN RMSE OF {args.EPOCHS} epochs: {test_min_rmse}  "
                        f"TEST MAX r OF {args.EPOCHS} epochs: {test_max_r}\n"])
        f.flush()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser(description="FreqPhys")
    parser.add_argument('--dataset', type=str, default="MMPD")
    parser.add_argument('--fold', type=int, default=1, help='select fold from vipl dataset')
    parser.add_argument('--Train_batchsize', type=int, default=32, help='Train_batchsize')
    parser.add_argument('--Val_batchsize', type=int, default=32, help='Val_batchsize')
    parser.add_argument('--Test_batchsize', type=int, default=32, help='Test_batchsize')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--EPOCHS', type=int, default=10, help='total training epochs')

    args = parser.parse_args()

    LearnRate = [1e-3]
    Dim = [128]
    Depth = [4]
    Heads = [4]
    Mlpdim = [128]

    for lr in LearnRate:
        for dim in Dim:
            for depth in Depth:
                for heads in Heads:
                    for mlpdim in Mlpdim:
                        run_training(lr, dim, depth, heads, dim)