from src.datamodules.datasets.greecefire_dataset import FireDatasetWholeDay
from src.models.greece_fire_models import combine_dynamic_static_inputs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from captum.attr import ShapleyValues, Lime
from matplotlib import pyplot as plt


def predict_map(ds, day, pl_module, access_mode, problem_class, patch_size, lag, dynamic_features, static_features,
                nan_fill, batch_size, mask_sea=True):
    """
    This function returns the predictions of model (pl_module) for a given day.
    """
    dataset = FireDatasetWholeDay(ds, day, access_mode, problem_class, patch_size, lag,
                                  dynamic_features,
                                  static_features,
                                  nan_fill)
    len_x = dataset.len_x
    len_y = dataset.len_y
    pl_module.eval()
    num_iterations = max(1, len(dataset) // batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    for i, (dynamic, static) in tqdm(enumerate(dataloader), total=num_iterations):
        inputs = combine_dynamic_static_inputs(dynamic, static, access_mode)
        if pl_module.on_gpu:
            inputs = inputs.cuda()
        logits = pl_module(inputs)
        preds_proba = torch.exp(logits)[:, 1]
        outputs.append(preds_proba.detach().cpu())
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.reshape(len_y, len_x)
    outputs = outputs.detach().cpu().numpy().squeeze()

    if mask_sea:
        # these are used to mask out land cover classes
        # that are not dangerous for fire (sea, water bodies, urban areas)
        clc = np.isin(ds['clc_2012'].values, list(range(12, 33)), invert=True)
        pop_den = np.isnan(ds['population_density_2012'].values)
        outputs[clc] = 0
        outputs[pop_den] = 0

    return outputs


import seaborn as sns
import pandas as pd


def lime_feature_ranking(model, dynamic_x, static_x, y, feature_names, access_mode):
    lime = Lime(model)
    x = combine_dynamic_static_inputs(dynamic_x, static_x, access_mode)
    feature_mask = torch.zeros_like(x, dtype=torch.long) + 1
    if access_mode == 'temporal':
        d = {}
        for i in range(len(feature_names)):
            feature_mask[:, :, i] = feature_mask[:, :, i] * i
            d[feature_names[i]] = x[:, :, i].squeeze().cpu().detach().numpy().tolist()
        d['index'] = list(range(10))
        df = pd.DataFrame.from_dict(d)
    elif access_mode == 'spatial':
        for i in range(len(feature_names)):
            feature_mask[:, i, :, :] = feature_mask[:, i, :, :] * i
    elif access_mode == 'spatiotemporal':
        for i in range(len(feature_names)):
            feature_mask[:, :, i, :, :] = feature_mask[:, :, i, :, :] * i
    attr_coarse = lime.attribute(x, target=y, feature_mask=feature_mask, n_samples=500, return_input_shape=False)
    attr_coarse = attr_coarse[0, :].cpu().detach().numpy()
    result = sorted(zip(list(attr_coarse), feature_names), reverse=True)
    fi_vector, ordered_features = list(zip(*result))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    raw_pred = model(x)[:, 1]
    pred_proba = torch.exp(raw_pred).cpu().detach().numpy()[0]
    raw_pred = raw_pred.cpu().detach().numpy()[0]
    # ax.set_title(f"Lime Features Ranking - Target {y} / Pred {pred:.2f}")
    axs[0].barh(ordered_features, fi_vector)
    if access_mode == 'temporal':
        for i in range(len(feature_names)):
            if feature_names[i] in ordered_features[:3] + ordered_features[-3:]:
                axs[1].plot('index', feature_names[i], data=df)
                axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[1].legend()
    fig.tight_layout()
    fig.suptitle(f'Lime: Raw Pred {raw_pred:.2f} / Pred {pred_proba:.2f} / Label {y}')
    return fig, pred_proba, y
