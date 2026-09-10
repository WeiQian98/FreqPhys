import glob
import os
import scipy
import torchvision.transforms as T

import config

# 5-fold cross-validation setting for VIPL dataset
def SplitforSubject(file_list, dataset, fold_num):

    file_list = [x for x in file_list if os.listdir(x)]
    train_ls = []
    test_ls = []
    if "VIPL" == dataset:
        fold_data_dict = {}
        fold_files = glob.glob(f"./Utils/fold_split/fold{fold_num}.mat")
        for fold in fold_files:
            dataset = fold.split('/')[-1].split('.')[0]
            fold_data = scipy.io.loadmat(fold)
            fold_data_dict[dataset] = fold_data[dataset]
        subjects = [x + 1 for x in range(107)]
        for idx, fold in enumerate(fold_data_dict.keys()):
            test_subject = [f'p{x}_' for x in fold_data_dict[fold].squeeze(0)]
            # test_subject = list(map(str, test_subject))
            train_subject = [f'p{x}_' for x in subjects if f'p{x}_' not in test_subject]
            # train_subject = list(map(str, train_subject))
            train_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in train_subject]
            test_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in test_subject]
            train_ls.append(train_video)
            test_ls.append(test_video)
    else:
        print("not find such dataset!")

    return train_ls, test_ls

def train_test_split(dataset_type, VIPL_Fold, Map_path):
    Maps_path_per_video = glob.glob(os.path.join(Map_path, '*'))
    train_list = []
    test_list = []
    val_list = []

    # UBFC
    if dataset_type == "UBFC":
        test_list = [f'{Map_path}/subject{x}' for x in range(38, 50)]
        train_list = [x for x in Maps_path_per_video if x not in test_list]

    # MMPD
    elif dataset_type == "MMPD":
        test_subject = ['p27','p28','p29','p30','p31','p32','p33']
        val_subject = ['p24', 'p25', 'p26']
        for sj in test_subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                test_list.append(data)
        for sj in val_subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                val_list.append(data)
        train_list = [x for x in Maps_path_per_video if x not in test_list and x not in val_list]
        # train_list = [x for x in Maps_path_per_video if x not in test_list]


    # PURE
    elif dataset_type == "PURE":
        subject = ['01', '02', '04', '03']
        for sj in subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                test_list.append(data)
        train_list = [x for x in Maps_path_per_video if x not in test_list]


    # VIPL
    elif dataset_type == "VIPL":
        traindata, testdata = SplitforSubject(Maps_path_per_video, dataset="VIPL", fold_num=VIPL_Fold)
        train_list = traindata[0]
        test_list = testdata[0]


    # MR-NIRP-Car
    elif dataset_type == "MR-NIRP-Car":
        test_subject = ['subject15', 'subject17', 'subject18', 'subject19']
        val_subject = ['subject13', 'subject14']
        for sj in test_subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                test_list.append(data)
        for sj in val_subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                val_list.append(data)
        train_list = [x for x in Maps_path_per_video if x not in test_list and x not in val_list]

    # buaa    
    elif dataset_type == "BUAA":
        test_subject = ['p10', 'p11', 'p12', 'p13']
        for sj in test_subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                test_list.append(data)
        train_list = [x for x in Maps_path_per_video if x not in test_list]
    

    # MSTmap for UBFC_MST
    ubfc_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.49826374650001526, 0.4897593557834625, 0.48688215017318726, 0.4903276562690735, 0.4862031638622284, 0.5129157900810242],
                    [0.2629663944244385, 0.25689175724983215, 0.25882282853126526, 0.2566501200199127, 0.2559047341346741, 0.26456427574157715])])

    # MSTmap for PURE_MST
    pure_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5100011229515076, 0.5100011229515076, 0.5043121576309204, 0.501944899559021, 0.49301525950431824, 0.5126286745071411],
                    [0.2933797240257263, 0.27947697043418884, 0.27273887395858765, 0.2857646644115448, 0.2774929702281952, 0.27686169743537903])])

    # MSTmap for VIPL_MST
    vipl_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703, 0.5182220339775085],
                    [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709, 0.2556385099887848])])

    # MSTmap for MMPD_MST
    mmpd_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
                    [0.4905047118663788, 0.48812583088874817, 0.4907720386981964, 0.4882889688014984, 0.4916536808013916, 0.5013716816902161],
                    [0.2586327791213989, 0.25982195138931274, 0.26021915674209595, 0.26085031032562256, 0.2508475184440613, 0.25173524022102356])])
    
    # MSTmap for MR-NIRP-Car_MST
    MR_NIRP_Car_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
                    [0.47663211822509766, 0.4767824411392212, 0.45914119482040405, 0.4746647775173187, 0.526628315448761, 0.5024640560150146],
                    [0.28687697649002075, 0.28599631786346436, 0.2797909379005432, 0.28588107228279114, 0.2782760262489319, 0.2628795802593231])])
    
    # MSTmap for BUAA_MST
    buaa_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
                    [0.47663211822509766, 0.4803699851036072, 0.48702332377433777, 0.48239558935165405, 0.490437388420105, 0.5040978193283081],
                    [0.23703548312187195, 0.24310331046581268, 0.2071295827627182, 0.2427489459514618, 0.19415003061294556, 0.22418928146362305])])    

    if (dataset_type == "UBFC"):
        transform_train = ubfc_transform
        transform_test = ubfc_transform
    elif (dataset_type == "BUAA"):
        transform_train = buaa_transform
        transform_test = buaa_transform
    elif (dataset_type == "PURE"):
        transform_train = pure_transform
        transform_test = pure_transform
    elif (dataset_type == "VIPL"):
        transform_train = vipl_transform
        transform_test = vipl_transform
    elif (dataset_type == "MMPD"):
        transform_train = mmpd_transform
        transform_val = mmpd_transform
        transform_test = mmpd_transform
    elif (dataset_type == "MR-NIRP-Car"):
        transform_train = MR_NIRP_Car_transform
        transform_val = MR_NIRP_Car_transform
        transform_test = MR_NIRP_Car_transform

    if dataset_type == "MR-NIRP-Car" or dataset_type == "MMPD":
        return train_list, val_list, test_list, transform_train, transform_val, transform_test
    
    return train_list, test_list, transform_train, transform_test



def cross_train_test_split(dataset_type):
    train_list = []
    test_list = []

    # cross dataset testing from UBFC to PURE
    if dataset_type == "UBFC2PURE":
        train_list = glob.glob(os.path.join('/home/anonymous/dataset/UBFC_MST/','*'))
        test_list = glob.glob(os.path.join('/home/anonymous/dataset/PURE_MST/','*'))

    # cross dataset testing from UBFC to MMPD
    elif dataset_type == "UBFC2MMPD":
        train_list = glob.glob(os.path.join('/home/anonymous/dataset/UBFC_MST/','*'))
        test_list = glob.glob(os.path.join('/home/anonymous/dataset/MMPD_MST/','*'))

    # cross dataset testing from PURE to UBFC
    elif dataset_type == "PURE2UBFC":
        train_list = glob.glob(os.path.join('/home/anonymous/dataset/PURE_MST/','*'))
        test_list = glob.glob(os.path.join('/home/anonymous/dataset/UBFC_MST/','*'))

    # cross dataset testing from PURE to MMPD
    elif dataset_type == "PURE2MMPD":
        train_list = glob.glob(os.path.join('/home/anonymous/dataset/PURE_MST/','*'))
        test_list = glob.glob(os.path.join('/home/anonymous/dataset/MMPD_MST/','*'))
    

    # MSTmap for UBFC_MST
    ubfc_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.49826374650001526, 0.4897593557834625, 0.48688215017318726, 0.4903276562690735, 0.4862031638622284, 0.5129157900810242],
                    [0.2629663944244385, 0.25689175724983215, 0.25882282853126526, 0.2566501200199127, 0.2559047341346741, 0.26456427574157715])])

    # MSTmap for PURE_MST
    pure_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5100011229515076, 0.5100011229515076, 0.5043121576309204, 0.501944899559021, 0.49301525950431824, 0.5126286745071411],
                    [0.2933797240257263, 0.27947697043418884, 0.27273887395858765, 0.2857646644115448, 0.2774929702281952, 0.27686169743537903])])

    # MSTmap for VIPL_MST
    vipl_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703, 0.5182220339775085],
                    [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709, 0.2556385099887848])])

    # MSTmap for MMPD_MST
    mmpd_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
                    [0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703, 0.5182220339775085],
                    [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709, 0.2556385099887848])])

    if (dataset_type == "UBFC2PURE"):
        transform_train = ubfc_transform
        transform_test = ubfc_transform
    elif (dataset_type == "UBFC2MMPD"):
        transform_train = ubfc_transform
        transform_test = ubfc_transform
    elif (dataset_type == "PURE2UBFC"):
        transform_train = pure_transform
        transform_test = pure_transform
    elif (dataset_type == "PURE2MMPD"):
        transform_train = pure_transform
        transform_test = pure_transform


    return train_list, test_list, transform_train, transform_test