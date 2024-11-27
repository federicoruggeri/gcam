import torch as th
from transformers import AutoConfig, AutoModel

from modeling.layers import GuidelinesConcatLookup, GuidelinesDotLookup


class ConcatGuidelines(th.nn.Module):
    """
    GCAM Concat model in Table 4
    """

    def __init__(
            self,
            hf_model_name,
            freeze_hf=False,
            dropout_rate=0.0,
    ):
        super().__init__()

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        # Guidelines
        self.guideline_lookup = GuidelinesConcatLookup(embedding_dim=self.embedding_config.hidden_size,
                                                       hidden_size=self.embedding_config.hidden_size,
                                                       dropout_rate=dropout_rate)

    def forward(
            self,
            input_ids,
            attention_mask,
            kb_input_ids,
            kb_attention_mask
    ):
        # input_ids:            [bs, N]
        # attention_mask:       [bs, N]
        # kb_input_ids:         [G, N]
        # kb_attention_mask:    [G, N]

        # Input
        # [bs, N, d]
        input_embeddings = self.embedding(input_ids=input_ids,
                                          attention_mask=attention_mask).last_hidden_state
        # [bs, d]
        input_embedding = input_embeddings[:, 0, :]

        # Guidelines
        # [G, N, d]
        guideline_embeddings = self.embedding(input_ids=kb_input_ids,
                                              attention_mask=kb_attention_mask).last_hidden_state
        # [G, d]
        guideline_embedding = guideline_embeddings[:, 0, :]

        # [bs, G, 2]
        guideline_logits = self.guideline_lookup(input_embedding=input_embedding,
                                                 guideline_embedding=guideline_embedding)

        return guideline_logits


class DotGuidelines(th.nn.Module):
    """
    GCAM Dot model in Table 4
    """

    def __init__(
            self,
            hf_model_name,
            freeze_hf=False,
            dropout_rate=0.0,
    ):
        super().__init__()

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        # Guidelines
        self.guideline_lookup = GuidelinesDotLookup(embedding_dim=self.embedding_config.hidden_size,
                                                    hidden_size=self.embedding_config.hidden_size,
                                                    dropout_rate=dropout_rate)

    def forward(
            self,
            input_ids,
            attention_mask,
            kb_input_ids,
            kb_attention_mask
    ):
        # input_ids:            [bs, N]
        # attention_mask:       [bs, N]
        # kb_input_ids:         [G, N]
        # kb_attention_mask:    [G, N]

        # Input
        # [bs, N, d]
        input_embeddings = self.embedding(input_ids=input_ids,
                                          attention_mask=attention_mask).last_hidden_state
        # [bs, d]
        input_embedding = input_embeddings[:, 0, :]

        # Guidelines
        # [G, N, d]
        guideline_embeddings = self.embedding(input_ids=kb_input_ids,
                                              attention_mask=kb_attention_mask).last_hidden_state
        # [G, d]
        guideline_embedding = guideline_embeddings[:, 0, :]

        # [bs, G, 2]
        guideline_logits = self.guideline_lookup(input_embedding=input_embedding,
                                                 guideline_embedding=guideline_embedding)

        return guideline_logits


class EntailGuidelines(th.nn.Module):
    """
    GCAM Entail model in Table 2 and Table 4
    """

    def __init__(
            self,
            hf_model_name,
            freeze_hf=False,
            dropout_rate=0.0,
            sep_token_id=2
    ):
        super().__init__()

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.classifier = th.nn.Sequential(
            th.nn.Linear(in_features=self.embedding_config.hidden_size,
                         out_features=self.embedding_config.hidden_size),
            th.nn.ReLU(),
            th.nn.Dropout(p=dropout_rate),
            th.nn.Linear(in_features=self.embedding_config.hidden_size, out_features=2)
        )

        self.sep_token_id = sep_token_id

    def forward(
            self,
            input_ids,
            attention_mask,
            kb_input_ids,
            kb_attention_mask
    ):
        # input_ids:            [bs, N]
        # attention_mask:       [bs, N]
        # kb_input_ids:         [G, N]
        # kb_attention_mask:    [G, N]

        G = kb_input_ids.shape[0]
        batch_size = input_ids.shape[0]

        # [bs, G, 1]
        sep_tensor = th.tensor([self.sep_token_id], dtype=input_ids.dtype, device=input_ids.device)
        sep_tensor = sep_tensor[None, None, :].expand(batch_size, G, -1)

        # [bs, G, 1]
        sep_mask_tensor = th.ones_like(sep_tensor)

        # [bs, G, N]
        input_ids = input_ids[:, None, :].expand(-1, G, -1)
        attention_mask = attention_mask[:, None, :].expand(-1, G, -1)
        kb_input_ids = kb_input_ids[None, :, :].expand(batch_size, -1, -1)
        kb_attention_mask = kb_attention_mask[None, :, :].expand(batch_size, -1, -1)

        # [bs, G, 2N + 1]
        pair_input_ids = th.concat((input_ids, sep_tensor, kb_input_ids), dim=-1)
        pair_attention_mask = th.concat((attention_mask, sep_mask_tensor, kb_attention_mask), dim=-1)

        # [bs * G, d]
        pair_embeddings = self.embedding(input_ids=pair_input_ids.view(-1, pair_input_ids.shape[-1]),
                                         attention_mask=pair_attention_mask.view(-1, pair_attention_mask.shape[
                                             -1])).last_hidden_state[:, 0, :]

        # [bs, G, 2]
        guideline_logits = self.classifier(pair_embeddings).view(batch_size, G, -1)

        return guideline_logits
