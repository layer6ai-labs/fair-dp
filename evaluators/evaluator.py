from inspect import getmembers, isfunction

from . import metrics

metric_fn_dict = dict(getmembers(metrics, predicate=isfunction))


class Evaluator:
    def __init__(self, model, *,
                 valid_loader, test_loader,
                 valid_metrics=None,
                 test_metrics=None,
                 **kwargs):
        self.model = model
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_metrics = valid_metrics or {}
        self.test_metrics = test_metrics or valid_metrics
        self.metric_kwargs = kwargs or {}

    def evaluate(self, dataloader, metric):
        assert metric in metric_fn_dict, f"Metric name {metric} not present in `metrics.py`"

        metric_fn = metric_fn_dict[metric]

        self.model.eval()
        return metric_fn(self.model, dataloader, **self.metric_kwargs)

    def validate(self):
        print(f"Validating {self.valid_metrics}")
        return {metric: self.evaluate(self.valid_loader, metric)
                for metric in self.valid_metrics}

    def test(self):
        print(f"Testing {self.test_metrics}")
        return {metric: self.evaluate(self.test_loader, metric)
                for metric in self.test_metrics}


def create_evaluator(model, valid_loader, test_loader, valid_metrics, test_metrics, **kwargs):
    valid_metrics = set(valid_metrics)
    test_metrics = set(test_metrics)

    return Evaluator(
        model,
        valid_loader=valid_loader,
        test_loader=test_loader,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        **kwargs
    )
