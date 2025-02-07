"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

# region Import

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

# endregion


class CrossCondGPT2_2part(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, shift=None):
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_body, x_hands = xs
        for k in range(cond.size(1)):
            
            x_cond_body = x_body if x_body.size(1) <= block_size else x_body[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_hands = x_hands if x_hands.size(1) <= block_size else x_hands[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed

            cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]

            logits, _, _ = self.forward((x_cond_body, x_cond_hands), cond_input)

            # pluck the logits at the final step and scale by temperature
            logit_body, logit_hands = logits
            logit_body = logit_body[:, -1, :]
            logit_hands = logit_hands[:, -1, :]

            probs_body = F.softmax(logit_body, dim=-1)
            probs_hands = F.softmax(logit_hands, dim=-1)

            # 在这里设置inference时的top-x值，x非1时可以增加生成的多样性，x=1时为确定性模型
            _, ix_body = torch.topk(probs_body, k=5, dim=-1)
            _, ix_hands = torch.topk(probs_hands, k=5, dim=-1)
            
            # 手动写了一个概率选择器，可以根据概率分布来选择下一个token
            prob_for_top5 = torch.tensor([0.9, 0.05, 0.02, 0.02, 0.01])
            prob = torch.distributions.Categorical(prob_for_top5)
            ix_body = (ix_body[0][prob.sample()]).unsqueeze(0)
            ix_hands = (ix_hands[0][prob.sample()]).unsqueeze(0)
            
            # append to the sequence and continue
            x_body = torch.cat((x_body, ix_body.unsqueeze(0)), dim=1)
            x_hands = torch.cat((x_hands, ix_hands.unsqueeze(0)), dim=1)

        return ([x_body], [x_hands])

    def forward(self, idxs, cond, targets=None):
        idx_body, idx_hands = idxs

        feat = self.gpt_base(idx_body, idx_hands, cond)
        logits_body, logits_hands, loss_body, loss_hands, metrics = self.gpt_head(feat, targets)

        if loss_body is not None and loss_hands is not None:
            loss = loss_body + loss_hands
        else:
            loss = None

        return (logits_body, logits_hands), loss, metrics


# 带mask的self-attention层，只能看到前面的信息
class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # self.mask = se
        self.n_head = config.n_head
        self.n_modality = config.n_modality

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T // self.n_modality
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print("t, att:", t, att.shape)
       
        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, self.n_modality, self.n_modality) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondGPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # if config.requires_head:
        # 对于一维整数值，就通过nn.Embedding来转换；对于高维浮点数值，就通过nn.Linear来转换
        self.n_modality = config.n_modality
        self.tok_emb_body = nn.Embedding(config.vocab_size_body, config.n_embd)
        self.tok_emb_hands = nn.Embedding(config.vocab_size_hands, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*self.n_modality, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.block_size = config.block_size


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

    def forward(self, idx_body, idx_hands, cond):
        b, t = idx_body.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_hands.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # if self.requires_head:
        token_embeddings_body = self.tok_emb_body(idx_body)  # each index maps to a (learnable) vector
        token_embeddings_hands = self.tok_emb_hands(idx_hands)  # each index maps to a (learnable) vector

        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings_body, token_embeddings_hands], dim=1)


        # 相当于是学了4个pe，只不过把他们concat起来了
        position_embeddings = torch.cat(
            [self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size + t, :],
                self.pos_emb[:, self.block_size * 2:self.block_size * 2 + t, :], 
            ],
            dim=1)  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)

        return x


class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.n_modality = config.n_modality
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.head_body = nn.Linear(config.n_embd, config.vocab_size_body, bias=False)
        self.head_hands = nn.Linear(config.n_embd, config.vocab_size_hands, bias=False)
        
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
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

    def forward(self, x, targets=None):
        x = self.blocks(x)
        x = self.ln_f(x)
        N, T, C = x.size()
        t = T // self.n_modality
        # 在这里舍弃掉了[:, 0:t]的与audio信息相关的部分
        logits_body = self.head_body(x[:, t:t * 2, :])
        logits_hands = self.head_hands(x[:, t * 2:t * 3, :])

        # if we are given some desired targets also calculate the loss
        loss_body, loss_hands = None, None
        metrics = None
        if targets is not None:
            targets_body, targets_hands = targets

            loss_body = F.cross_entropy(logits_body.view(-1, logits_body.size(-1)), targets_body.view(-1))
            loss_hands = F.cross_entropy(logits_hands.view(-1, logits_hands.size(-1)), targets_hands.view(-1))

    
            probs_body = F.softmax(logits_body.view(-1, logits_body.size(-1)).clone().detach(), dim=-1)
            probs_hands = F.softmax(logits_hands.view(-1, logits_hands.size(-1)).clone().detach(), dim=-1)

            _, ix_body = torch.topk(probs_body, k=1, dim=-1)
            _, ix_hands = torch.topk(probs_hands, k=1, dim=-1)

            ix_body = ix_body.view(-1)
            ix_hands = ix_hands.view(-1)
            acc_body = torch.mean((ix_body == targets_body.view(-1).clone().detach()).type(torch.float))
            acc_hands = torch.mean((ix_hands == targets_hands.view(-1).clone().detach()).type(torch.float))

            metrics = {
                "accuracy_body": acc_body,
                "accuracy_hands": acc_hands
            }

        return logits_body, logits_hands, loss_body, loss_hands, metrics