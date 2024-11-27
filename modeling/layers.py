import torch as th


class GuidelinesConcatLookup(th.nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 dropout_rate=0.2
                 ):
        super(GuidelinesConcatLookup, self).__init__()

        self.lookup_mlp = th.nn.Sequential(
            th.nn.Linear(in_features=embedding_dim * 2, out_features=hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Dropout(p=dropout_rate),
            th.nn.Linear(in_features=hidden_size, out_features=2)
        )

    def forward(
            self,
            input_embedding,
            guideline_embedding,
    ):
        # input_embedding:     [bs, d]
        # guideline_embedding: [G, d]

        G = guideline_embedding.shape[0]
        batch_size = input_embedding.shape[0]

        # [bs * G, 2 * d]
        mlp_input = th.concat((input_embedding[:, None, :].expand(-1, G, -1),
                               guideline_embedding[None, :, :].expand(batch_size, -1, -1)), dim=-1)
        mlp_input = mlp_input.view(batch_size * G, -1)

        # [bs, G, 2]
        memory_scores = self.lookup_mlp(mlp_input)
        memory_scores = memory_scores.view(batch_size, G, -1)

        return memory_scores


class GuidelinesDotLookup(th.nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 dropout_rate=0.2
                 ):
        super(GuidelinesDotLookup, self).__init__()

        self.lookup_mlp = th.nn.Sequential(
            th.nn.Linear(in_features=embedding_dim, out_features=hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Dropout(p=dropout_rate),
            th.nn.Linear(in_features=hidden_size, out_features=2)
        )

    def forward(
            self,
            input_embedding,
            guideline_embedding,
    ):
        # input_embedding:     [bs, d]
        # guideline_embedding: [G, d]

        G = guideline_embedding.shape[0]
        batch_size = input_embedding.shape[0]

        # [bs * G, d]
        mlp_input = input_embedding[:, None, :] * guideline_embedding[None, :, :]
        mlp_input = mlp_input.view(batch_size * G, -1)

        # [bs, G, 2]
        memory_scores = self.lookup_mlp(mlp_input)
        memory_scores = memory_scores.view(batch_size, G, -1)

        return memory_scores
