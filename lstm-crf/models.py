import pytorch_lightning as pl
import torch
from torch import nn
from all_utils import *
from transformers import AutoModel
from torchmetrics.classification import MulticlassF1Score
from typing import Dict, List, Tuple

class NERModel(pl.LightningModule):
    def __init__(self, config: Config):
        super(NERModel, self).__init__()
        self.lstm_module = LSTM(config.tokenizer, config.num_layers, config.hidden_size, config.dropout)
        self.crf_module = CRFModule(config.num_labels)
        self.f1 = MulticlassF1Score(num_classes=config.num_labels, ignore_index=0, average='micro')
        self.config = config
        self.pred = []
        self.golden_truth = []

    def forward(self, input_ids, word_ids, mask):
        emissions = self.lstm_module(input_ids)
        feats = NERModel.aggragate(emissions, word_ids)
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
        feats = NERModel.aggragate(emissions, word_ids)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer
    
    @staticmethod
    def aggragate(lstm_feats: torch.Tensor, word_ids: Tuple[List[int]], mask: torch.Tensor):
        hidden_size = lstm_feats.shape[-1]
        batch_size, seq_len = mask.size()
        feats = torch.zeros(batch_size, seq_len, hidden_size, device=lstm_feats.device)
        for i, ids in enumerate(word_ids):
            for j, id in enumerate(ids):
                if id < 0: continue
                feats[i, id] += lstm_feats[i, j]
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
        lstm_out, _ = self.lstm(embeds, self.hidden)
        lstm_feats = self.linear(lstm_out)
        return lstm_feats

class CRFModule(nn.Module):
    def __init__(self, tagset_size):
        super(CRFModule, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.start_transitions = nn.Parameter(torch.randn(tagset_size))
        self.stop_transitions = nn.Parameter(torch.randn(tagset_size))

    def _log_sum_exp(self, vec):
        max_score, _ = vec.max(dim=-1)
        return max_score + torch.log(torch.exp(vec - max_score.unsqueeze(-1)).sum(dim=-1))

    def _forward_alg(self, feats, mask):
        forward_var = self.start_transitions + feats[:, 0]
        for t in range(1, feats.size(1)):
            emit_score = feats[:, t]
            emit_score = emit_score.unsqueeze(-1)
            transition_scores = self.transitions.unsqueeze(0)
            broadcasted_forward_vars = forward_var.unsqueeze(2)
            next_score = broadcasted_forward_vars + emit_score + transition_scores
            next_score = self._log_sum_exp(next_score)
            forward_var = torch.where(mask[:, t].unsqueeze(-1), next_score, forward_var)
        terminal_var = forward_var + self.stop_transitions
        alpha = self._log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, mask):
        score = feats.new_zeros(tags.size(0))
        first_tag_indices = tags[:, 0]
        score += self.start_transitions[first_tag_indices]
        for i in range(feats.size(1) - 1):
            cur_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            transition_score = self.transitions[next_tag, cur_tag]
            emit_score = feats[:, i].gather(1, cur_tag.view(-1, 1)).squeeze()
            score = torch.where(mask[:, i+1], score + transition_score + emit_score, score)
        last_tag_indices = tags.gather(1, mask.sum(dim=1).long().view(-1, 1) - 1)
        score += self.stop_transitions[last_tag_indices.view(-1)]
        return score

    def forward(self, feats, tags, mask):
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return (forward_score - gold_score).mean()
    
    def _viterbi_decode(self, feats, mask):
        batch_size, sequence_length, num_tags = feats.shape

        # Initialize the viterbi variables in log space
        path_scores = feats.new_zeros(size=(batch_size, num_tags))
        path_scores += feats[:, 0] + self.start_transitions

        # Create a tensor to hold accumulated sequence scores at each step
        path_scores_history = feats.new_zeros(size=(batch_size, sequence_length, num_tags))

        # Create a tensor to hold the backpointers
        backpointers = feats.new_zeros(size=(batch_size, sequence_length, num_tags), dtype=torch.long)

        for t in range(1, sequence_length):
            # Broadcast the transition scores to one more dimension
            scores_with_trans = path_scores.unsqueeze(2) + self.transitions

            # Take the maximum over the tag dimension
            max_scores, max_score_tags = torch.max(scores_with_trans, dim=1)

            # Add emission scores
            path_scores = feats[:, t] + max_scores

            # Apply the mask and save scores and backpointers
            path_scores = torch.where(mask[:, t].unsqueeze(-1), path_scores, path_scores.clone().detach())
            path_scores_history[:, t] = path_scores
            backpointers[:, t] = max_score_tags

        # Transition to STOP_TAG
        path_scores += self.stop_transitions

        # Traceback
        best_paths = torch.zeros_like(backpointers, dtype=torch.long)
        _, best_tags = torch.max(path_scores, dim=1)
        best_paths[:, -1] = best_tags
        for t in range(sequence_length-2, -1, -1):
            best_tags = backpointers[torch.arange(batch_size), t+1, best_tags]
            best_paths[:, t] = best_tags

        # The best_paths tensor has a dimension size of [batch_size, sequence_length]
        # where best_paths[i, :] is the best path for the i-th sample in the batch.
        return best_paths

