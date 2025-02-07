# region Import

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

# endregion

class Fine_GPT2_2part(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        # self.requires_head = config.requires_head
        # self.requires_tail = config.requires_tail

        # if config.requires_head:
        # 对于一维整数值，就通过nn.Embedding来转换；对于高维浮点数值，就通过nn.Linear来转换
        # dance_
        self.n_modality = config.n_modality
        self.tok_emb_body = nn.Embedding(config.vocab_size_body, config.n_embd)
        # self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        self.tok_emb_hands = nn.Embedding(config.vocab_size_hands, config.n_embd)


        self.layer_no = config.layer_no
        if config.layer_no == 2:
            self.tok_emb_body_2 = nn.Embedding(config.vocab_size_body, config.n_embd)
            # self.tok_emb_down_2 = nn.Embedding(config.vocab_size_down, config.n_embd)
            self.tok_emb_hands_2 = nn.Embedding(config.vocab_size_hands, config.n_embd)
        elif config.layer_no == 3:
            self.tok_emb_body_2 = nn.Embedding(config.vocab_size_body, config.n_embd)
            # self.tok_emb_down_2 = nn.Embedding(config.vocab_size_down, config.n_embd)
            self.tok_emb_hands_2 = nn.Embedding(config.vocab_size_hands, config.n_embd)
            self.tok_emb_body_3 = nn.Embedding(config.vocab_size_body, config.n_embd)
            # self.tok_emb_down_3 = nn.Embedding(config.vocab_size_down, config.n_embd)
            self.tok_emb_hands_3 = nn.Embedding(config.vocab_size_hands, config.n_embd)

        # self.tok_emb_face = nn.Embedding(config.vocab_size_face, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*self.n_modality, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.n_embd,
                                nhead=config.n_head,
                                dim_feedforward=int(4*config.n_embd),
                                dropout=config.attn_pdrop,
                                activation='gelu',
                                batch_first=True),
                    num_layers=config.n_layer)
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head_body = nn.Linear(config.n_embd, config.vocab_size_body, bias=False)
        # self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_hands = nn.Linear(config.n_embd, config.vocab_size_hands, bias=False)
        
        self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idxs, cond, targets=None):
        if self.layer_no == 1:
            idx_body, idx_hands = idxs
            b, t = idx_body.size()
            # print(b, t)
            token_embeddings_body = self.tok_emb_body(idx_body)  # each index maps to a (learnable) vector
            # token_embeddings_down = self.tok_emb_down(idx_down)  # each index maps to a (learnable) vector
            token_embeddings_hands = self.tok_emb_hands(idx_hands)  # each index maps to a (learnable) vector

        elif self.layer_no == 2:
            idx_1, idx_2 = idxs
            idx_body_1, idx_hands_1 = idx_1
            idx_body_2, idx_hands_2 = idx_2
            b, t = idx_body_1.size()

            token_embeddings_body_1 = self.tok_emb_body(idx_body_1)  # each index maps to a (learnable) vector
            token_embeddings_body_2 = self.tok_emb_body_2(idx_body_2)  # each index maps to a (learnable) vector
            token_embeddings_body = token_embeddings_body_1 + token_embeddings_body_2

            token_embeddings_hands_1 = self.tok_emb_hands(idx_hands_1)  # each index maps to a (learnable) vector
            token_embeddings_hands_2 = self.tok_emb_hands_2(idx_hands_2)  # each index maps to a (learnable) vector
            token_embeddings_hands = token_embeddings_hands_1 + token_embeddings_hands_2
        elif self.layer_no == 3:
            idx_1, idx_2, idx_3 = idxs
            idx_body_1, idx_hands_1 = idx_1
            idx_body_2, idx_hands_2 = idx_2
            idx_body_3, idx_hands_3 = idx_3
            b, t = idx_body_1.size()

            token_embeddings_body_1 = self.tok_emb_body(idx_body_1)
            token_embeddings_body_2 = self.tok_emb_body_2(idx_body_2)
            token_embeddings_body_3 = self.tok_emb_body_3(idx_body_3)
            token_embeddings_body = token_embeddings_body_1 + token_embeddings_body_2 + token_embeddings_body_3

            token_embeddings_hands_1 = self.tok_emb_hands(idx_hands_1)
            token_embeddings_hands_2 = self.tok_emb_hands_2(idx_hands_2)
            token_embeddings_hands_3 = self.tok_emb_hands_3(idx_hands_3)
            token_embeddings_hands = token_embeddings_hands_1 + token_embeddings_hands_2 + token_embeddings_hands_3

        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings_body, token_embeddings_hands], dim=1)

        # print("token_embeddings_whole.shape = ", token_embeddings_body.shape, token_embeddings_down.shape)

        # 相当于是学了4个pe，只不过把他们concat起来了
        position_embeddings = torch.cat(
            [self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size + t, :],
                self.pos_emb[:, self.block_size * 2:self.block_size * 2 + t, :], 
            ],
            dim=1)  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        # x = self.ln_f(x)

        x = self.ln_f(x)
        N, T, C = x.size()
        t = T // self.n_modality
        # 在这里舍弃掉了[:, 0:t]的与audio信息相关的部分
        logits_body = self.head_body(x[:, t:t * 2, :])
        logits_hands = self.head_hands(x[:, t * 2:t * 3, :])
        # logits_face = self.head_face(x[:, t * 4:t * 5, :])

        # if we are given some desired targets also calculate the loss
        loss_body, loss_hands = None, None
        metrics = None
            
        probs_body = F.softmax(logits_body.view(-1, logits_body.size(-1)).clone().detach(), dim=-1)
        probs_hands = F.softmax(logits_hands.view(-1, logits_hands.size(-1)).clone().detach(), dim=-1)
        # probs_face = F.softmax(logits_face.view(-1, logits_face.size(-1)).clone().detach(), dim=-1)

        _, ix_body = torch.topk(probs_body, k=1, dim=-1)
        _, ix_hands = torch.topk(probs_hands, k=1, dim=-1)
        # _, ix_face = torch.topk(probs_face, k=1, dim=-1)

        if targets is not None:
            targets_body, targets_hands = targets

            loss_body = F.cross_entropy(logits_body.view(-1, logits_body.size(-1)), targets_body.view(-1))
            loss_hands = F.cross_entropy(logits_hands.view(-1, logits_hands.size(-1)), targets_hands.view(-1))
            # loss_face = F.cross_entropy(logits_face.view(-1, logits_face.size(-1)), targets_face.view(-1))
        
            ix_body = ix_body.view(-1)
            ix_hands = ix_hands.view(-1)
            # ix_face = ix_face.view(-1)

            # assert ix_body.size()[0] == targets_body.view(-1).size()[0]
            acc_body = torch.mean((ix_body == targets_body.view(-1).clone().detach()).type(torch.float))
            acc_hands = torch.mean((ix_hands == targets_hands.view(-1).clone().detach()).type(torch.float))
            # acc_face = torch.mean((ix_face == targets_face.view(-1).clone().detach()).type(torch.float))

            metrics = {
                "accuracy_body": acc_body,
                "accuracy_hands": acc_hands
            }

        if loss_body is not None and loss_hands is not None:
            loss = loss_body + loss_hands
        else:
            loss = None

        # return (logits_body, logits_down, logits_hands), loss, metrics
        return (ix_body.reshape(-1, 15), ix_hands.reshape(-1, 15)), loss, metrics