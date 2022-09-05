from torchmetrics.functional import accuracy, f1_score
import pytorch_lightning
import torch.nn
import torchvision.models
from torch.nn import functional as F
import torch.nn as nn

torch.manual_seed(0)


def resnet_model(num_classes=151):
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    # model.maxpool = torch.nn.Identity()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model


class AnimalModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        # init a pretrained resnet
        backbone = torchvision.models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, 151)

    def forward(self, x, feat_cls=False):

        x = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        logits = self.classifier(x)

        if feat_cls:
            return x, logits
        else:
            return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.half())

        log_dict = {}
        log_dict["loss"] = F.cross_entropy(pred, y)
        # log_dict["accuracy"] = accuracy(pred, y, average="macro", num_classes=151)
        # self.log_dict(log_dict, prog_bar=True)

        return log_dict["loss"]

    def validation_step(self, batch, batcn_idx):
        x, y = batch
        pred = self(x.half())

        log_dict = {}
        log_dict["val_loss"] = F.cross_entropy(pred, y)
        log_dict["val_acc"] = accuracy(pred, y, average="macro", num_classes=151)
        log_dict["val_acc_topk5"] = accuracy(
            pred, y, average="macro", num_classes=151, top_k=5
        )
        log_dict["val_f1"] = f1_score(pred, y, average="macro", num_classes=151)
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.97
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_acc"},
        }
