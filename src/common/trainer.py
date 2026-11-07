"""Validation-only early stopping with an actual best checkpoint and final test pass."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger

from utils.topk_evaluator import TopKEvaluator
from utils.checkpoint import load_checkpoint


class Trainer:
    def __init__(self, config, model):
        self.config, self.model = config, model
        self.logger = getLogger()
        self.device = config['device']
        self.valid_metric = config['valid_metric'].lower()
        self.bigger = config['valid_metric_bigger']
        self.best_valid_score = -float('inf') if self.bigger else float('inf')
        self.best_valid_result, self.best_test_upon_valid = {}, {}
        self.best_epoch = None
        self.epoch_seconds = []
        self.history = []
        self.evaluator = TopKEvaluator(config)
        optimizer = {'adam': optim.Adam, 'sgd': optim.SGD, 'adagrad': optim.Adagrad, 'rmsprop': optim.RMSprop}
        self.optimizer = optimizer[config['learner'].lower()](
            (p for p in model.parameters() if p.requires_grad), lr=config['learning_rate'],
            weight_decay=float(config['weight_decay'] or 0))
        decay, period = config['learning_rate_scheduler']
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: decay ** (epoch / period))
        self.output = Path(config['tensorboard_log_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output / 'best.pt'
        self.writer = SummaryWriter(str(self.output))

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        if valid_data is None or len(valid_data) == 0:
            raise ValueError('A nonempty validation set is required')
        if self.config['epochs'] < 1 or self.config['eval_step'] < 1:
            raise ValueError('epochs and eval_step must be positive')
        stale = 0
        try:
            for epoch in range(self.config['epochs']):
                self.model.train()
                self.model.pre_epoch_processing()
                self._sync()
                start = perf_counter()
                totals, count = None, 0
                for batch in train_data:
                    self.optimizer.zero_grad(set_to_none=True)
                    parts = self.model.calculate_loss(batch)
                    parts = parts if isinstance(parts, tuple) else (parts,)
                    loss = sum(parts)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss at epoch {}'.format(epoch + 1))
                    loss.backward()
                    if self.config['clip_grad_norm']:
                        clip_grad_norm_(self.model.parameters(), **self.config['clip_grad_norm'])
                    self.optimizer.step()
                    values = np.array([p.detach().item() for p in parts])
                    totals = values if totals is None else totals + values
                    count += 1
                if not count:
                    raise ValueError('Training split is empty')
                self._sync()
                elapsed = perf_counter() - start
                self.epoch_seconds.append(elapsed)
                self.scheduler.step()
                self.model.post_epoch_processing()
                row = {'epoch': epoch + 1, 'seconds': elapsed, 'loss_components': (totals / count).tolist()}
                self.writer.add_scalar('train/loss', float(totals.sum() / count), epoch + 1)
                for i, value in enumerate(totals / count):
                    self.writer.add_scalar('train/component_{}'.format(i), value, epoch + 1)
                if (epoch + 1) % self.config['eval_step'] == 0 or epoch + 1 == self.config['epochs']:
                    result = self.evaluate(valid_data)
                    score = result[self.valid_metric]
                    if not np.isfinite(score):
                        raise FloatingPointError('Nonfinite validation score')
                    row['validation'] = result
                    for metric, value in result.items():
                        self.writer.add_scalar('validation/' + metric, value, epoch + 1)
                    improved = score > self.best_valid_score if self.bigger else score < self.best_valid_score
                    if improved:
                        self.best_valid_score, self.best_valid_result, self.best_epoch = score, result, epoch + 1
                        stale = 0
                        # Fixed graphs are rebuilt from the recorded data/config, not serialized.
                        torch.save({'state_dict': self.model.state_dict(), 'epoch': self.best_epoch,
                                    'validation': result}, self.checkpoint)
                    else:
                        stale += 1
                self.history.append(row)
                self.logger.info('epoch %d loss %.6f time %.2fs validation %s',
                                 epoch + 1, totals.sum() / count, elapsed, row.get('validation', {}))
                if stale >= self.config['stopping_step']:
                    break
            if self.best_epoch is None:
                raise RuntimeError('No validation checkpoint was produced')
            checkpoint = load_checkpoint(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            if test_data is not None:
                self.best_test_upon_valid = self.evaluate(test_data, is_test=True)
            (self.output / 'history.json').write_text(json.dumps(self.history, indent=2))
            if not saved:
                self.checkpoint.unlink(missing_ok=True)
            return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid
        finally:
            self.writer.close()

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        predictions = []
        if max(self.config['topk']) > self.model.n_items:
            raise ValueError('topk exceeds number of items')
        try:
            if hasattr(self.model, 'prepare_evaluation'):
                self.model.prepare_evaluation()
            for batch in eval_data:
                scores = self.model.full_sort_predict(batch)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite recommendation scores')
                mask = batch[1]
                scores[mask[0], mask[1]] = -torch.inf
                ranking = scores.topk(max(self.config['topk']), dim=1).indices
                # If fewer than k unseen items exist, do not count masked items as hits.
                selected = scores.gather(1, ranking)
                ranking[~torch.isfinite(selected)] = -1
                predictions.append(ranking.cpu())
            result = self.evaluator.evaluate(predictions, eval_data, is_test=is_test, idx=idx)
            if is_test and self.config['export_rankings']:
                np.savez_compressed(self.output / 'test_rankings.npz',
                                    users=eval_data.get_eval_users().numpy(),
                                    topk=torch.cat(predictions).numpy())
            return result
        finally:
            if hasattr(self.model, 'clear_evaluation'):
                self.model.clear_evaluation()
