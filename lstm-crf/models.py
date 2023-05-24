import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch import logsumexp as _log_sum_exp
from all_utils import *
from transformers import AutoModel
from torchmetrics.classification import MulticlassF1Score
from typing import Dict, List, Tuple

class NERModel(pl.LightningModule):
    def __init__(self, config: Config):
        super(NERModel, self).__init__()
        self.lstm_module = LSTM(config.tokenizer, config.num_layers, config.hidden_size, config.num_labels, config.dropout)
        self.crf_module = LinearCRF(config.num_labels)
        self.f1 = MulticlassF1Score(num_classes=config.num_labels, ignore_index=0, 
                                    validate_args=False, average='micro')
        self.config = config
        self.pred = []
        self.golden_truth = []

    def forward(self, input_ids, word_ids, mask):
        emissions = self.lstm_module(input_ids)
        feats = NERModel.aggragate(emissions, word_ids, mask)
        pred = self.crf_module._viterbi_decode(feats, mask)
        return pred

    def training_step(self, batch, batch_idx):
        input_ids, labels, word_ids, mask = batch
        emissions = self.lstm_module(input_ids)
        feats = NERModel.aggragate(emissions, word_ids, mask)
        loss = self.crf_module(feats, labels, mask)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, labels, word_ids, mask = batch
        emissions = self.lstm_module(input_ids)
        feats = NERModel.aggragate(emissions, word_ids, mask)
        pred = self.crf_module._viterbi_decode(feats, mask)
        self.pred.append(pred.flatten())
        self.golden_truth.append(labels.flatten())

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.pred, dim=0)
        all_labels = torch.cat(self.golden_truth, dim=0)
        f1 = self.f1(all_preds, all_labels)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.pred.clear()
        self.golden_truth.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-4)
        return optimizer
    
    @staticmethod
    def aggragate(lstm_feats: torch.Tensor, word_ids: Tuple[torch.Tensor], mask: torch.Tensor):
        hidden_size = lstm_feats.shape[-1]
        batch_size, seq_len = mask.size()
        feats = torch.zeros(batch_size, seq_len, hidden_size, device=lstm_feats.device)
        for i, ids in enumerate(word_ids):
            for j, id in enumerate(ids):
                if id >= 0: feats[i, id] += lstm_feats[i, j]
        feats = F.log_softmax(feats, dim=-1)
        return feats

    
class LSTM(nn.Module):
    def __init__(self, embedding_name, num_layers, hidden_size, num_labels, dropout):
        super(LSTM, self).__init__()
        self.embeddings = AutoModel.from_pretrained(embedding_name).get_input_embeddings()
        self.lstm = nn.LSTM(self.embeddings.embedding_dim, hidden_size=hidden_size, batch_first=True,
                             num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.linear(lstm_out)
        return lstm_feats

class LinearCRF(pl.LightningModule):
    def __init__(self, num_tags):
        super(LinearCRF, self).__init__()

        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def _compute_log_numerator(self, logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor):
        batch_size, seq_len, num_tags = logits.size()
        # idx_targets = targets  * mask
        score = logits.gather(2, targets.unsqueeze(-1)).squeeze(2)
        score[:, 0] += self.start_transitions.gather(0, targets[:, 0])
        last_non_padded = mask.sum(dim=-1).long() - 1
        stop_scores = self.stop_transitions.gather(0, targets.gather(1, last_non_padded.view(-1, 1)).squeeze(1))

        score = (score * mask).sum(dim=-1) + stop_scores

        # Calculate transition scores using vectorized operations
        # （9，9）的张量A，两个（32，46）的张量 C,D. 以C,D对应位置的一组数作为索引，去取A中的值，得到一个（32，46）的输出。
        targets_shifted = torch.roll(targets, shifts=-1, dims=-1).flatten() # latter time step
        mask_shifted = torch.roll(mask, shifts=-1, dims=-1)
        transition_scores = (self.transitions[targets_shifted, targets.flatten()].view(batch_size, seq_len) * mask_shifted)[:, :-1]
        score += transition_scores.sum(dim=-1)

        return score

    def _compute_log_partition_function(self, logits: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, num_tags = logits.size()
        mask = mask.unsqueeze(-1).expand(-1, -1, num_tags)

        alpha = logits[:, 0] + self.start_transitions.unsqueeze(0)
        for t in range(1, seq_len):
            # torch.Size([32, 1, 9]) torch.Size([1, 9, 9]) torch.Size([32, 9, 1])
            # print("$$#", alpha.unsqueeze(1).size(), self.transitions.unsqueeze(0).size(), logits[:, t].unsqueeze(-1).size())
            alpha_t = alpha.unsqueeze(1) + self.transitions.unsqueeze(0) + logits[:, t].unsqueeze(-1)
            alpha_t_sum = torch.logsumexp(alpha_t, dim=-1)
            alpha = torch.where(mask[:, t], alpha_t_sum, alpha)

        alpha += self.stop_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=-1)

    def forward(self, logits, targets, mask):
        log_numerator = self._compute_log_numerator(logits, mask, targets)
        log_partition_function = self._compute_log_partition_function(logits, mask)
        log_likelihood = log_partition_function - log_numerator  

        return log_likelihood.mean()

    def _viterbi_decode(self, logits: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, num_tags = logits.size()
        scores = logits[:, 0] + self.start_transitions.unsqueeze(0)

        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            #######
            # print("$$#", self.transitions.size(), logits[:, t].unsqueeze(1).size(), scores.size())
            # $$# torch.Size([9, 9]) torch.Size([32, 1, 9]) torch.Size([32, 9])
            scores_t = scores.unsqueeze(1) + self.transitions.unsqueeze(0)

            scores_t, backpointers_t = torch.max(scores_t, dim=-1)
            backpointers.append(backpointers_t)
            scores = torch.where(mask[:, t].unsqueeze(-1), scores_t + logits[:, t], scores)

        scores += self.stop_transitions.unsqueeze(0)

        _, best_tags = torch.max(scores, dim=-1)
        best_paths = [best_tags]
        # for backpointers_t in reversed(backpointers):
        #     # assert not any(best_tags >= 9), f"{best_tags}" # .unsqueeze(1).long()
        #     best_tags = backpointers_t.gather(1, best_tags)
        #     best_paths.append(best_tags)
        for t in range(seq_len-2, -1, -1):
            best_tags = torch.where(mask[:, t+1], backpointers[t][torch.arange(batch_size), best_tags], best_tags)
            assert best_tags.size() == (batch_size,)
            best_paths.append(best_tags)
        best_paths.reverse()
        # $$ torch.Size([32, 46]) torch.Size([32, 1]) 46 torch.Size([32, 1])                                                                              
        # print("$$$", ret.shape, best_tags.shape, len(best_paths), best_paths[-1].shape)
        return torch.stack(best_paths, dim=1).squeeze() * mask


