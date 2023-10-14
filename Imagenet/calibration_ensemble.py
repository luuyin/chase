import os 
import torch
import numpy as np
import torch.nn.functional as F



# direction = 'DST_ImageNet_Models/Independent_BS64'

# arr = os.listdir(direction)
# arr = sorted(arr)

# file_list = os.listdir('../data/imagenet-c')
# file_list.sort()

# for severity in range(1,6):

#     all_folds_preds_c = []

#     for checkpoint in range(len(arr)):

#         print(arr[checkpoint])
#         all_preds = []
#         all_targets = []

#         for file_name in file_list:
#             y_pred = np.load(os.path.join(direction+'_predict', '{}-{}-{}.npy'.format(arr[checkpoint], file_name, severity)))    
#             y_true = np.load(os.path.join(direction+'_predict', '{}-{}-{}-label.npy'.format(arr[checkpoint], file_name, severity)))              
#             all_preds.append(y_pred)
#             all_targets.append(y_true)

#         all_preds = np.concatenate(all_preds, axis=0)
#         all_targets = np.concatenate(all_targets, axis=0)
#         print('{}-Acc = {}'.format(severity, np.mean(np.argmax(all_preds,1)==all_targets)))

#         all_folds_preds_c.append(all_preds)


#     output_mean = np.mean(np.stack(all_folds_preds_c[:2], 0), 0)
#     print(output_mean.shape)
#     print('{}-ensemble2-Acc = {}'.format(severity, np.mean(np.argmax(output_mean,1)==all_targets)))

#     output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
#     print(output_mean.shape)
#     print('{}-ensembleAll-Acc = {}'.format(severity, np.mean(np.argmax(output_mean,1)==all_targets)))

















# direction = 'DST_ImageNet_Models/BS_64'

# arr = os.listdir(direction)
# arr = sorted(arr)

# file_list = os.listdir('../data/imagenet-c')
# file_list.sort()

# for severity in range(1,6):

#     all_folds_preds_c = []

#     for checkpoint in range(len(arr)):

#         print(arr[checkpoint])
#         all_preds = []
#         all_targets = []

#         for file_name in file_list:
#             y_pred = np.load(os.path.join(direction+'_predict', '{}-{}-{}.npy'.format(arr[checkpoint], file_name, severity)))    
#             y_true = np.load(os.path.join(direction+'_predict', '{}-{}-{}-label.npy'.format(arr[checkpoint], file_name, severity)))              
#             all_preds.append(y_pred)
#             all_targets.append(y_true)

#         all_preds = np.concatenate(all_preds, axis=0)
#         all_targets = np.concatenate(all_targets, axis=0)
#         print('{}-Acc = {}'.format(severity, np.mean(np.argmax(all_preds,1)==all_targets)))

#         all_folds_preds_c.append(all_preds)


#     output_mean = np.mean(np.stack(all_folds_preds_c[:2], 0), 0)
#     print(output_mean.shape)
#     print('{}-ensemble2-Acc = {}'.format(severity, np.mean(np.argmax(output_mean,1)==all_targets)))

#     output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
#     print(output_mean.shape)
#     print('{}-ensembleAll-Acc = {}'.format(severity, np.mean(np.argmax(output_mean,1)==all_targets)))







def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]









direction = 'DST_ImageNet_Models/Independent_BS64'

arr = os.listdir(direction)
arr = sorted(arr)

file_list = os.listdir('../data/imagenet-c')
file_list.sort()

ece={}
nll={}
for checkpoint in range(len(arr)):
    ece[arr[checkpoint]] = []
    nll[arr[checkpoint]] = []

ece['ensemble-2'] = []
nll['ensemble-2'] = []

ece['ensemble-all'] = []
nll['ensemble-all'] = []


for severity in range(1,6):
    print(severity)
    for file_name in file_list:

        all_folds_preds_c = []
        for checkpoint in range(len(arr)):
            y_pred = np.load(os.path.join(direction+'_predict', '{}-{}-{}.npy'.format(arr[checkpoint], file_name, severity)))    
            y_true = np.load(os.path.join(direction+'_predict', '{}-{}-{}-label.npy'.format(arr[checkpoint], file_name, severity)))              
            all_folds_preds_c.append(y_pred)

            ece_s = expected_calibration_error(y_true, y_pred)
            nll_s = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
            ece[arr[checkpoint]].append(ece_s)
            nll[arr[checkpoint]].append(nll_s.item())

        output_mean = np.mean(np.stack(all_folds_preds_c[:2], 0), 0)
        ece_s = expected_calibration_error(y_true, output_mean)
        nll_s = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
        ece['ensemble-2'].append(ece_s)
        nll['ensemble-2'].append(nll_s.item())

        output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
        ece_s = expected_calibration_error(y_true, output_mean)
        nll_s = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
        ece['ensemble-all'].append(ece_s)
        nll['ensemble-all'].append(nll_s.item())


for key in ece.keys():
    ece_array = np.array(ece[key])
    nll_array = np.array(nll[key])
    print('key = {}, length = {}, ece = {}, nll = {}'.format(key, ece_array.shape, np.mean(ece_array), np.mean(nll_array)))


# direction = 'DST_ImageNet_Models/BS_64'

# arr = os.listdir(direction)
# arr = sorted(arr)
# all_folds_preds_c = []

# for checkpoint in range(len(arr)):

#     all_preds = []
#     all_targets = []

#     print(arr[checkpoint])

#     file_list = os.listdir('../data/imagenet-c')
#     file_list.sort()

#     for severity in range(1,6):
#         for file_name in file_list:
#             y_pred = np.load(os.path.join(direction+'_predict', '{}-{}-{}.npy'.format(arr[checkpoint], file_name, severity)))    
#             y_true = np.load(os.path.join(direction+'_predict', '{}-{}-{}-label.npy'.format(arr[checkpoint], file_name, severity)))              
#             all_preds.append(y_pred)
#             all_targets.append(y_true)

#     all_preds = np.concatenate(all_preds, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     all_folds_preds_c.append(all_preds)

#     ece = expected_calibration_error(all_targets, all_preds)
#     nll = F.nll_loss(torch.from_numpy(all_preds).log(), torch.from_numpy(all_targets), reduction="mean")
#     print('* c-ECE = {}'.format(ece))
#     print('* c-NLL = {}'.format(nll))

# output_mean = np.mean(np.stack(all_folds_preds_c[:2], 0), 0)
# print(output_mean.shape)
# ece = expected_calibration_error(all_targets, output_mean)
# nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
# print('* c-ECE-2 = {}'.format(ece))
# print('* c-NLL-2 = {}'.format(nll))
# print('* Acc-2 = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))


# output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
# print(output_mean.shape)
# ece = expected_calibration_error(all_targets, output_mean)
# nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
# print('* c-ECE-a = {}'.format(ece))
# print('* c-NLL-a = {}'.format(nll))
# print('* Acc-a = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))

