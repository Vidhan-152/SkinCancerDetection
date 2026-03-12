"""
Training utilities: losses, metrics, and trainer classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Losses ──────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for segmentation. Expects logits."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedSegLoss(nn.Module):
    """BCE + Dice combined loss for robust segmentation training."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def dice_score(pred_mask, true_mask, threshold=0.5, smooth=1.0):
    """Compute Dice coefficient from probability maps."""
    pred = (pred_mask > threshold).float()
    intersection = (pred * true_mask).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + true_mask.sum() + smooth)).item()


def iou_score(pred_mask, true_mask, threshold=0.5, smooth=1.0):
    """Compute IoU from probability maps."""
    pred = (pred_mask > threshold).float()
    intersection = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ─── Segmentation Trainer ────────────────────────────────────────────────────

class SegTrainer:
    def __init__(self, model, optimizer, scheduler=None, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = CombinedSegLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss, total_dice = 0.0, 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                total_dice += dice_score(probs.cpu(), masks.cpu())
        n = len(loader)
        return total_loss / n, total_dice / n

    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            probs = torch.sigmoid(logits)
            total_loss += loss.item()
            total_dice += dice_score(probs.cpu(), masks.cpu())
            total_iou += iou_score(probs.cpu(), masks.cpu())
        n = len(loader)
        return total_loss / n, total_dice / n, total_iou / n

    def step_scheduler(self, metric=None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()


# ─── Classification Trainer ──────────────────────────────────────────────────

class ClsTrainer:
    def __init__(self, model, optimizer, scheduler=None, device="cpu", pos_weight=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        w = torch.tensor([pos_weight]).to(device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=w)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.unsqueeze(1).to(self.device)
            out = self.model(imgs)
            loss = self.criterion(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.unsqueeze(1).to(self.device)
            out = self.model(imgs)
            loss = self.criterion(out, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_probs.extend(probs.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().squeeze().tolist())
        return total_loss / len(loader), correct / total

    def step_scheduler(self, metric=None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
