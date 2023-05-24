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
        self.crf_module = CRFModule(config.num_labels)
        self.f1 = MulticlassF1Score(num_classes=config.num_labels, ignore_index=0, average='micro')
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer
    
    @staticmethod
    def aggragate(lstm_feats: torch.Tensor, word_ids: Tuple[torch.Tensor], mask: torch.Tensor):
        hidden_size = lstm_feats.shape[-1]
        batch_size, seq_len = mask.size()
        feats = torch.zeros(batch_size, seq_len, hidden_size, device=lstm_feats.device)
        for i, ids in enumerate(word_ids):
            for j, id in enumerate(ids):
                if id < 0: continue
                feats[i, id] += lstm_feats[i, j]
        feats = F.log_softmax(feats, dim=-1)
        return feats

    
class LSTM(nn.Module):
    def __init__(self, embedding_name, num_layers, hidden_size, num_labels, dropout):
        super(LSTM, self).__init__()
        self.embeddings = AutoModel.from_pretrained(embedding_name).get_input_embeddings()
        self.lstm = nn.LSTM(self.embeddings.embedding_dim, hidden_size=hidden_size,
                             num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.linear(lstm_out)
        return lstm_feats

class CRFModule(nn.Module):
    def __init__(self, tagset_size):
        super(CRFModule, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.zeros(tagset_size, tagset_size))
        self.start_transitions = nn.Parameter(torch.zeros(tagset_size))
        self.stop_transitions = nn.Parameter(torch.zeros(tagset_size))

    def _forward_alg(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, seq_len = mask.shape
        mask_in_tag = mask.unsqueeze(-1).expand(-1, -1, self.tagset_size)
        forward_var = self.start_transitions + feats[:, 0]
        for t in range(1, seq_len):
            next_score = forward_var.unsqueeze(1) + self.transitions.unsqueeze(0) + feats[:, t].unsqueeze(-1)
            next_score = _log_sum_exp(next_score, dim=-1)
            forward_var = torch.where(mask_in_tag[:, t], next_score, forward_var)
        terminal_var = forward_var + self.stop_transitions
        alpha = _log_sum_exp(terminal_var, dim=-1) # (batch_size,)
        return alpha

    def _score_sentence(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        batch_idx = torch.arange(0, batch_size, dtype=torch.long, device=feats.device)
        seq_idx = torch.arange(0, seq_len, dtype=torch.long, device=feats.device)
        transition_score = self.transitions[tags[:, 1:], tags[:, :-1]]
        emit_score = feats[batch_idx.view(-1, 1), seq_idx.view(1, -1), tags]
        score = transition_score + emit_score[:, 1:]
        score = torch.where(mask[:, 1:], score, torch.zeros_like(score))
        score = score.sum(dim=-1) + emit_score[:, 0] + self.start_transitions[tags[:, 0]]
        last_idx = mask.long().sum(-1) - 1
        score += self.stop_transitions[tags[batch_idx, last_idx]]
        return score

    def forward(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor):
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return (forward_score - gold_score).mean()
    
    # def _viterbi_decode(self, logits: torch.Tensor, mask: torch.Tensor):
    #     batch_size, max_len, n_tags = logits.size()
    #     seq_len = mask.long().sum(1)
    #     logits = logits.transpose(0, 1).data  # L, B, H
    #     mask = mask.transpose(0, 1).data.eq(True)  # L, B
    #     flip_mask = mask.eq(False)

    #     # dp
    #     vpath = logits.new_zeros((max_len, batch_size, n_tags),
    #                              dtype=torch.long)
    #     vscore = logits[0]  # bsz x n_tags
    #     transitions = logits.new_zeros(n_tags + 2, n_tags + 2)
    #     transitions[:n_tags, :n_tags] += self.transitions.transpose(0, 1).data
    #     transitions[n_tags, :n_tags] += self.start_transitions.data
    #     transitions[:n_tags, n_tags + 1] += self.stop_transitions.data

    #     vscore += transitions[n_tags, :n_tags]

    #     trans_score = transitions[:n_tags, :n_tags].view(1, n_tags,
    #                                                      n_tags).data
    #     end_trans_score = transitions[:n_tags,
    #                                   n_tags + 1].view(1, 1, n_tags).repeat(
    #                                       batch_size, 1, 1)  # bsz, 1, n_tags

    #     # 针对长度为1的句子
    #     vscore += transitions[:n_tags, n_tags + 1].view(1, n_tags).repeat(
    #         batch_size, 1).masked_fill(seq_len.ne(1).view(-1, 1), 0)
    #     for i in range(1, max_len):
    #         prev_score = vscore.view(batch_size, n_tags, 1)
    #         cur_score = logits[i].view(batch_size, 1, n_tags) + trans_score
    #         score = prev_score + cur_score.masked_fill(flip_mask[i].view(
    #             batch_size, 1, 1), 0)  # bsz x n_tag x n_tag
    #         # 需要考虑当前位置是该序列的最后一个
    #         score += end_trans_score.masked_fill(
    #             seq_len.ne(i + 1).view(-1, 1, 1), 0)

    #         best_score, best_dst = score.max(1)
    #         vpath[i] = best_dst
    #         # 由于最终是通过last_tags回溯，需要保持每个位置的vscore情况
    #         vscore = best_score.masked_fill(
    #             flip_mask[i].view(batch_size, 1), 0) + \
    #             vscore.masked_fill(mask[i].view(batch_size, 1), 0)

    #     # backtrace
    #     batch_idx = torch.arange(
    #         batch_size, dtype=torch.long, device=logits.device)
    #     seq_idx = torch.arange(max_len, dtype=torch.long, device=logits.device)
    #     lens = (seq_len - 1)
    #     # idxes [L, B], batched idx from seq_len-1 to 0
    #     idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % max_len

    #     ans = logits.new_empty((max_len, batch_size), dtype=torch.long)
    #     ans_score, last_tags = vscore.max(1)
    #     ans[idxes[0], batch_idx] = last_tags
    #     for i in range(max_len - 1):
    #         last_tags = vpath[idxes[i], batch_idx, last_tags]
    #         ans[idxes[i + 1], batch_idx] = last_tags

    #     paths = (ans * mask).transpose(0, 1)
    #     return paths
    def _viterbi_decode(self, feats: torch.Tensor, mask: torch.Tensor):
        batch_size, sequence_length, num_tags = feats.shape

        # Initialize the viterbi variables in log space
        path_scores = feats[:, 0] + self.start_transitions

        # Create a tensor to hold accumulated sequence scores at each step
        # path_scores_history = feats.new_zeros(size=(batch_size, sequence_length, num_tags))

        # Create a tensor to hold the backpointers
        backpointers = feats.new_zeros(size=(batch_size, sequence_length, num_tags), dtype=torch.long)

        for t in range(1, sequence_length):
            # Broadcast the transition scores to one more dimension
            scores_with_trans = path_scores.unsqueeze(1) + self.transitions

            # Take the maximum over the tag dimension
            max_scores, max_score_tags = torch.max(scores_with_trans, dim=-1)

            # Use the mask to find the valid path scores and update accordingly
            mask_t = mask[:, t].unsqueeze(-1)
            path_scores = torch.where(mask_t, feats[:, t] + max_scores, path_scores)
            # path_scores_history[:, t] = path_scores
            backpointers[:, t] = max_score_tags

        # Transition to STOP_TAG
        path_scores += self.stop_transitions

        # Traceback
        best_paths = torch.zeros_like(mask, dtype=torch.long)
        _, best_tags = torch.max(path_scores, dim=-1)
        best_paths[:, -1] = best_tags
        for t in range(sequence_length-2, -1, -1):
            mask_t = mask[:, t+1]
            # print("$$==", best_tags.shape, best_paths.shape, backpointers[torch.arange(batch_size), t+1, best_tags].shape)
            best_tags = torch.where(mask_t, backpointers[torch.arange(batch_size), t+1, best_tags], best_tags)
            # print("$$", best_tags.shape, best_paths.shape)
            best_paths[:, t] = best_tags

        # The best_paths tensor has a dimension size of [batch_size, sequence_length]
        # where best_paths[i, :] is the best path for the i-th sample in the batch.
        best_paths[mask == False] = 0
        return best_paths


