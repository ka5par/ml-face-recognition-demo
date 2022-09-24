from tracemalloc import reset_peak
from model import AnimalModel
from dataloader import create_dataloaders, get_transform
import torch
import torch.nn.functional as F
import distance_metrics
import pickle
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from functools import partial
import json

torch.manual_seed(0)


def run_model(model, loader, projection="wdout_projection", softmax=True):
    # run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    for images, target in loader:
        total += images.size(0)
        images = images.cuda()
        if softmax:
            output, classifier = model(images, feat_cls=True)
            cls_list.append(F.softmax(classifier, dim=1).data.cpu())
        else:
            output = model(
                images, return_feat=True if projection == "wdout_projection" else False
            )

        out_list.append(output.data.cpu())
        tgt_list.append(target)

    return torch.cat(out_list), torch.cat(tgt_list), torch.cat(cls_list)


def get_cache(model, train_set, val_set):
    # Get normal embeddings
    if not os.path.exists("cache.pkl"):
        cache = {}
        print("Compute sample mean for training data....")
        train_emb, train_targets, train_sfmx = run_model(model, train_set)
        train_acc = float(
            torch.sum(torch.argmax(train_sfmx, dim=1) == train_targets)
        ) / len(train_sfmx)
        print("Accuracy of train instances : ", train_acc)
        train_cache = []
        train_cache.append(train_emb)
        train_cache.append(train_targets)
        train_cache.append(train_sfmx)
        cache["train"] = train_cache
        test_emb, test_targets, test_sfmx = run_model(model, val_set)
        test_acc = float(
            torch.sum(torch.argmax(test_sfmx, dim=1) == test_targets)
        ) / len(test_sfmx)
        print("Accuracy of val instances : ", test_acc)
        test_cache = []
        test_cache.append(test_emb)
        test_cache.append(test_targets)
        test_cache.append(test_sfmx)
        cache["test"] = test_cache
        pickle.dump(cache, open("cache.pkl", "wb"))

    else:
        print("Fetching sample mean from cache data.....")
        cache = pickle.load(open("cache.pkl", "rb"))
        train_emb = cache["train"][0].cuda()
        train_targets = cache["train"][1].cuda()
        train_sfmx = cache["train"][2].cuda()
        test_emb = cache["test"][0].cuda()
        test_targets = cache["test"][1].cuda()
        test_sfmx = cache["test"][2].cuda()

    return train_emb, train_targets, train_sfmx, test_emb, test_targets, test_sfmx


def get_distances(emb, sfmx, classes_mean, classes_feats):
    distances = ["Softmax", "Mahalanobis", "Euclidean", "Cosine"]

    output_dict = {}
    for distance in distances:
        if distance == "Softmax":
            score_calc = partial(torch.max, axis=1)
        else:
            score_calc = partial(torch.min, axis=1)

        if distance == "Euclidean":
            distances_list = distance_metrics.euclidean(emb, classes_mean)
        elif distance == "Mahalanobis":
            from sklearn.covariance import LedoitWolf

            inv_cov_fpath = "inv_cov.pkl"
            if not os.path.exists(inv_cov_fpath):
                print("Calculating inv covariance for support or training set")
                sup_inv_cov = [
                    torch.from_numpy(
                        LedoitWolf().fit(cls_feat.cpu().numpy()).precision_
                    )
                    .float()
                    .cuda()
                    for cls_feat in classes_feats
                ]
                pickle.dump(sup_inv_cov, open(inv_cov_fpath, "wb"))
            else:
                sup_inv_cov = pickle.load(open(inv_cov_fpath, "rb"))
            distances_list = distance_metrics.mahalanobis(
                emb, classes_mean, sup_inv_cov
            )
        elif distance == "Softmax":
            distances_list = sfmx
        elif distance == "Cosine":
            distances_list = distance_metrics.cosine(emb, classes_mean)

        score, cls = score_calc(distances_list)
        cls = int(cls.detach().cpu().numpy()[0])
        score = round(float(score.detach().cpu().numpy()[0]), 2)

        translation = json.load(open("data/translation.json"))
        list_translation = list(translation)
        latin_animal = list_translation[cls]
        human_readable_animal = translation[latin_animal]

        output_dict[distance] = {human_readable_animal: score}

    return output_dict


def load_model():
    model = AnimalModel.load_from_checkpoint(
        "lightning_logs/version_0/checkpoints/last.ckpt"
    )
    model.eval()
    model = model.cuda()

    train_set, val_set = create_dataloaders()

    train_set, val_set = DataLoader(
        train_set, shuffle=True, num_workers=4, batch_size=128
    ), DataLoader(val_set, shuffle=True, num_workers=4, batch_size=128)

    return model, train_set, val_set


def get_classes_means(train_targets=None, train_emb=None, calc_new=True):
    if calc_new:
        in_classes = torch.unique(train_targets)
        class_idx = [
            torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1)
            for cls in in_classes
        ]
        classes_feats = [train_emb[idx] for idx in class_idx]

        classes_mean = torch.stack(
            [torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0
        )
        pickle.dump(classes_feats, open("classes_feats.pkl", "wb"))
        pickle.dump(classes_mean, open("classes_mean.pkl", "wb"))
    else:
        classes_feats = pickle.load(open("classes_feats.pkl", "rb"))
        classes_mean = pickle.load(open("classes_mean.pkl", "rb"))

    return classes_feats, classes_mean


def inference_id_dataset():

    model, train_set, val_set = load_model()

    train_emb, train_targets, train_sfmx, test_emb, _, test_sfmx = get_cache(
        model, train_set, val_set
    )

    classes_feats, classes_mean = get_classes_means(
        train_targets=train_targets, train_emb=train_emb, calc_new=True
    )

    train_distances_cls = get_distances(
        train_emb, train_sfmx, classes_mean, classes_feats
    )
    test_distances_cls = get_distances(test_emb, test_sfmx, classes_mean, classes_feats)

    pickle.dump(train_distances_cls, open("train_distances.pkl", "wb"))
    pickle.dump(test_distances_cls, open("test_distances.pkl", "wb"))
    return train_distances_cls, test_distances_cls


# def to_animal(res):

#     for key, item in res.items():
#         if "cls" in key:
#             idx = item
#             latin_animal = list_translation[idx]
#             human_readable_animal = translation[latin_animal]
#             res[key] = human_readable_animal
#         else:
#             # To make number JSON seralizable
#             # try:
#             res[key] = float(round(res[key], 2))

#     return res


def inference_input(input_image, name=None):
    img = Image.open(input_image)
    img = np.array(img)
    transform = get_transform(split="not_train")
    img = transform(image=img)["image"].cuda()

    model, _, _ = load_model()

    output, classifier = model(img.unsqueeze(0), feat_cls=True)
    sfmx = F.softmax(classifier, dim=1).data.cpu()

    classes_feats, classes_mean = get_classes_means(calc_new=False)

    distances_cls = get_distances(
        output.repeat(512, 1), sfmx.repeat(512, 1), classes_mean, classes_feats
    )
    try:
        distances_cls = {key: item for key, item in distances_cls.items()}
    except IndexError as e:
        print(distances_cls)
        raise e

    # preds = to_animal(distances_cls)
    preds = distances_cls
    print(preds)
    if name is not None:
        with open(f"data/preds/{name}.json", "w") as f:
            json.dump(preds, f)


if __name__ == "__main__":
    if not os.path.exists("train_distances.pkl"):
        train_distances_cls, test_distances_cls = inference_id_dataset()
    else:
        train_distances_cls, test_distances_cls = pickle.load(
            open("train_distances.pkl", "rb")
        ), pickle.load(open("test_distances.pkl", "rb"))

    inference_input(
        "data/faces/Bill Gates.jpg",
    )
