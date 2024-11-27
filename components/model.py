import lightning as L
import torch as th
from torchmetrics import MetricCollection


class DiscreteModel(L.LightningModule):
    """
    Lightning wrapper for SAM Binary model (check modeling/discrete.py)
    """

    def __init__(
            self,
            model: th.nn.Module,
            optimizer_class,
            num_classes,
            class_weights=None,
            val_metrics: MetricCollection = None,
            test_metrics: MetricCollection = None,
            log_metrics: bool = True,
            optimizer_kwargs={}
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model',
                                          'val_metrics',
                                          'test_metrics'])

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics

        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        self.clf_loss = th.nn.CrossEntropyLoss(reduction='none')

        class_weights = class_weights if class_weights is not None else [1.] * num_classes
        self.register_buffer('class_weights', th.tensor(class_weights, dtype=th.float32))

    def forward(
            self,
            batch
    ):
        input_ids, attention_mask, y_true = batch
        return self.model(input_ids, attention_mask)

    def compute_loss(
            self,
            y_hat,
            y_true
    ):
        # y_hat:    [bs, C]
        # y_true:   [bs,]

        total_loss = 0
        losses = {}

        # [bs,]
        sample_weights = th.where(y_true == 1, self.class_weights[1], self.class_weights[0])

        clf_loss = self.clf_loss(y_hat, y_true)
        sample_weights = sample_weights.to(clf_loss.dtype)
        clf_loss = (clf_loss * sample_weights).sum() / sample_weights.sum()

        total_loss += clf_loss
        losses['CE'] = clf_loss

        return total_loss, losses

    def training_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true = batch
        y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=y_hat, y_true=y_true)

        self.log(name='train_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'train_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
            self,
            batch,
            batch_idx,
    ):
        input_ids, attention_mask, y_true = batch
        y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=y_hat, y_true=y_true)

        self.log(name='val_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'val_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        if self.val_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_metrics.update(y_hat, y_true)

        return total_loss

    def on_validation_epoch_end(
            self,
    ) -> None:
        if self.val_metrics is not None:
            metric_values = self.val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'val_{key}', value, prog_bar=self.log_metrics)
            self.val_metrics.reset()

    def test_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true = batch
        y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=y_hat, y_true=y_true)

        self.log(name='test_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'test_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        if self.test_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_metrics.update(y_hat, y_true)

        return total_loss

    def on_test_epoch_end(
            self,
    ) -> None:
        if self.test_metrics is not None:
            metric_values = self.test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'test_{key}', value, prog_bar=self.log_metrics)
            self.test_metrics.reset()

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)


class MultilabelModel(L.LightningModule):
    """
    Lightning wrapper for GCAM Fine-grained model (check modeling/discrete.py)
    """

    def __init__(
            self,
            model: th.nn.Module,
            optimizer_class,
            num_classes,
            class_weights=None,
            val_metrics: MetricCollection = None,
            test_metrics: MetricCollection = None,
            log_metrics: bool = True,
            optimizer_kwargs={},
            kb_val_metrics: MetricCollection = None,
            kb_test_metrics: MetricCollection = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['model',
                                          'val_metrics',
                                          'kb_val_metrics',
                                          'test_metrics',
                                          'kb_test_metrics'])

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics

        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.kb_val_metrics = kb_val_metrics
        self.kb_test_metrics = kb_test_metrics

        self.clf_loss = th.nn.CrossEntropyLoss(reduction='none')

        class_weights = class_weights if class_weights is not None else [1.] * num_classes
        self.register_buffer('class_weights', th.tensor(class_weights, dtype=th.float32))

    def forward(
            self,
            batch
    ):
        input_ids, attention_mask, y_true, kb_y_true = batch
        return self.model(input_ids, attention_mask)

    def compute_loss(
            self,
            y_hat,
            y_true
    ):
        # y_hat:    [bs, G, 2]
        # y_true:   [bs, G]

        total_loss = 0
        losses = {}

        # [bs * G, 2]
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        # [bs * G]
        y_true = y_true.view(-1)

        # [bs,]
        sample_weights = th.where(y_true == 1, self.class_weights[1], self.class_weights[0])

        clf_loss = self.clf_loss(y_hat, y_true)
        sample_weights = sample_weights.to(clf_loss.dtype)
        clf_loss = (clf_loss * sample_weights).sum() / sample_weights.sum()

        total_loss += clf_loss
        losses['CE'] = clf_loss

        return total_loss, losses

    def training_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true, kb_y_true = batch
        kb_y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        self.log(name='train_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'train_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
            self,
            batch,
            batch_idx,
    ):
        input_ids, attention_mask, y_true, kb_y_true = batch
        kb_y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        self.log(name='val_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'val_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        y_hat = th.max(th.argmax(kb_y_hat, dim=-1), dim=-1).values
        if self.val_metrics is not None:
            self.val_metrics.update(y_hat, y_true)

        kb_y_hat = th.argmax(kb_y_hat, dim=-1)
        if self.kb_val_metrics is not None:
            self.kb_val_metrics.update(kb_y_hat, kb_y_true)

        return total_loss

    def on_validation_epoch_end(
            self,
    ) -> None:
        if self.val_metrics is not None:
            metric_values = self.val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'val_{key}', value, prog_bar=self.log_metrics)
            self.val_metrics.reset()

        if self.kb_val_metrics is not None:
            metric_values = self.kb_val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'kb_val_{key}', value, prog_bar=self.log_metrics)
            self.kb_val_metrics.reset()

    def test_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true, kb_y_true = batch
        kb_y_hat = self.model(input_ids, attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        self.log(name='test_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(name=f'test_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True)

        y_hat = th.max(th.argmax(kb_y_hat, dim=-1), dim=-1).values
        if self.test_metrics is not None:
            self.test_metrics.update(y_hat, y_true)

        kb_y_hat = th.argmax(kb_y_hat, dim=-1)
        if self.kb_test_metrics is not None:
            self.kb_test_metrics.update(kb_y_hat, kb_y_true)

        return total_loss

    def on_test_epoch_end(
            self,
    ) -> None:
        if self.test_metrics is not None:
            metric_values = self.test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'test_{key}', value, prog_bar=self.log_metrics)
            self.test_metrics.reset()

        if self.kb_test_metrics is not None:
            metric_values = self.kb_test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'kb_test_{key}', value, prog_bar=self.log_metrics)
            self.kb_test_metrics.reset()

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)


class GuidelineModel(L.LightningModule):
    """
    Lightning wrapper for GCAM Text-based models (check modeling/guidelines.py)
    """

    def __init__(
            self,
            model: th.nn.Module,
            optimizer_class,
            num_classes,
            class_weights=None,
            val_metrics: MetricCollection = None,
            test_metrics: MetricCollection = None,
            log_metrics: bool = True,
            optimizer_kwargs={},
            kb_val_metrics: MetricCollection = None,
            kb_test_metrics: MetricCollection = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['model',
                                          'val_metrics',
                                          'kb_val_metrics',
                                          'test_metrics',
                                          'kb_test_metrics'])

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics

        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.kb_val_metrics = kb_val_metrics
        self.kb_test_metrics = kb_test_metrics

        self.clf_loss = th.nn.CrossEntropyLoss(reduction='none')

        class_weights = class_weights if class_weights is not None else [1.] * num_classes
        self.register_buffer('class_weights', th.tensor(class_weights, dtype=th.float32))

        # Support variable to store predictions during evaluate() or test()
        self.store_predictions = False
        self.val_predictions = []
        self.test_predictions = []

    def forward(
            self,
            batch
    ):
        input_ids, attention_mask, y_true, kb_y_true, kb_input_ids, kb_attention_mask = batch
        return self.model(input_ids, attention_mask, kb_input_ids, kb_attention_mask)

    def compute_loss(
            self,
            y_hat,
            y_true
    ):
        # y_hat:    [bs, G, 2]
        # y_true:   [bs, G]

        total_loss = 0
        losses = {}

        # [bs * G, 2]
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        # [bs * G]
        y_true = y_true.view(-1)

        # [bs,]
        sample_weights = th.where(y_true == 1, self.class_weights[1], self.class_weights[0])

        clf_loss = self.clf_loss(y_hat, y_true)
        sample_weights = sample_weights.to(clf_loss.dtype)
        clf_loss = (clf_loss * sample_weights).sum() / sample_weights.sum()

        total_loss += clf_loss
        losses['CE'] = clf_loss

        return total_loss, losses

    def training_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true, kb_y_true, kb_input_ids, kb_attention_mask = batch
        kb_y_hat = self.model(input_ids, attention_mask, kb_input_ids, kb_attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        batch_size = y_true.shape[0]
        self.log(name='train_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for loss_name, loss_value in losses.items():
            self.log(name=f'train_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return total_loss

    def validation_step(
            self,
            batch,
            batch_idx,
    ):
        input_ids, attention_mask, y_true, kb_y_true, kb_input_ids, kb_attention_mask = batch
        kb_y_hat = self.model(input_ids, attention_mask, kb_input_ids, kb_attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        batch_size = y_true.shape[0]
        self.log(name='val_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for loss_name, loss_value in losses.items():
            self.log(name=f'val_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        y_hat = th.max(th.argmax(kb_y_hat, dim=-1), dim=-1).values
        if self.val_metrics is not None:
            self.val_metrics.update(y_hat, y_true)

        kb_y_hat = th.argmax(kb_y_hat, dim=-1)
        if self.kb_val_metrics is not None:
            self.kb_val_metrics.update(kb_y_hat, kb_y_true)

        return total_loss

    def on_validation_epoch_end(
            self,
    ) -> None:
        if self.val_metrics is not None:
            metric_values = self.val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'val_{key}', value, prog_bar=self.log_metrics)
            self.val_metrics.reset()

        if self.kb_val_metrics is not None:
            metric_values = self.kb_val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'kb_val_{key}', value, prog_bar=self.log_metrics)
            self.kb_val_metrics.reset()

    def test_step(
            self,
            batch,
            batch_idx
    ):
        input_ids, attention_mask, y_true, kb_y_true, kb_input_ids, kb_attention_mask = batch
        kb_y_hat = self.model(input_ids, attention_mask, kb_input_ids, kb_attention_mask)

        total_loss, losses = self.compute_loss(y_hat=kb_y_hat, y_true=kb_y_true)

        batch_size = y_true.shape[0]
        self.log(name='test_loss', value=total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for loss_name, loss_value in losses.items():
            self.log(name=f'test_{loss_name}', value=loss_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        y_hat = th.max(th.argmax(kb_y_hat, dim=-1), dim=-1).values
        if self.test_metrics is not None:
            self.test_metrics.update(y_hat, y_true)

        kb_y_hat = th.argmax(kb_y_hat, dim=-1)
        if self.kb_test_metrics is not None:
            self.kb_test_metrics.update(kb_y_hat, kb_y_true)

        return total_loss

    def on_test_epoch_end(
            self,
    ) -> None:
        if self.test_metrics is not None:
            metric_values = self.test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'test_{key}', value, prog_bar=self.log_metrics)
            self.test_metrics.reset()

        if self.kb_test_metrics is not None:
            metric_values = self.kb_test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'kb_test_{key}', value, prog_bar=self.log_metrics)
            self.kb_test_metrics.reset()

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
