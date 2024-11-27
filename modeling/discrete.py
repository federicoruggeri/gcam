import torch as th
from transformers import AutoConfig, AutoModel


class Discrete(th.nn.Module):
    """
    SAM Binary model in Table 2
    """

    def __init__(
            self,
            num_classes,
            hf_model_name,
            freeze_hf=False,
            dropout_rate=0.0
    ):
        super().__init__()

        self.num_classes = num_classes

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.pre_classifier = th.nn.Linear(in_features=self.embedding_config.hidden_size,
                                           out_features=self.embedding_config.hidden_size)
        self.pre_activation = th.nn.ReLU()
        self.classifier = th.nn.Linear(out_features=self.num_classes,
                                       in_features=self.embedding_config.hidden_size)

    def forward(
            self,
            input_ids,
            attention_mask
    ):
        # input_ids:        [bs, N]
        # attention_mask:   [bs, N]

        # [bs, N, d]
        embeddings = self.embedding(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        # [bs, C]
        pre_logits = self.pre_classifier(embeddings[:, 0, :])
        pre_logits = self.pre_activation(pre_logits)
        if self.training:
            pre_logits = self.dropout(pre_logits)
        logits = self.classifier(pre_logits)

        return logits


class MultiLabel(th.nn.Module):
    """
    GCAM Fine-grained model in Table 2
    """

    def __init__(
            self,
            num_labels,
            hf_model_name,
            freeze_hf=False,
            dropout_rate=0.0
    ):
        super().__init__()

        self.num_labels = num_labels

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.pre_classifier = th.nn.Linear(in_features=self.embedding_config.hidden_size,
                                           out_features=self.embedding_config.hidden_size)
        self.pre_activation = th.nn.ReLU()
        self.label_classifiers = th.nn.ModuleList(
            [self.build_label_classifier(hidden_size=self.embedding_config.hidden_size)
             for _ in range(num_labels)])

    def build_label_classifier(
            self,
            hidden_size
    ):
        return th.nn.Linear(out_features=2,
                            in_features=hidden_size)

    def forward(
            self,
            input_ids,
            attention_mask
    ):
        # input_ids:        [bs, N]
        # attention_mask:   [bs, N]

        # [bs, N, d]
        embeddings = self.embedding(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        # [bs, d]
        pre_logits = self.pre_classifier(embeddings[:, 0, :])
        pre_logits = self.pre_activation(pre_logits)
        if self.training:
            pre_logits = self.dropout(pre_logits)

        # [bs, G, 2]
        logits = [self.label_classifiers[label_idx](pre_logits) for label_idx in th.arange(self.num_labels)]
        logits = th.stack(logits, dim=1)

        return logits
