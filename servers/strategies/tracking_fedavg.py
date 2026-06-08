import flwr as fl

from monitoring.timing import Timer
from monitoring.memory import MemoryTracker
from monitoring.logger import append_jsonl


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, monitoring_log_path=None, **kwargs):
        self.monitoring_log_path = monitoring_log_path
        self.latest_parameters = kwargs.get("initial_parameters", None)

        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        memory_tracker = MemoryTracker("server_aggregate_fit_memory")
        memory_tracker.start()

        with Timer("server_aggregate_fit") as timer:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round,
                results,
                failures,
            )

        memory_metrics = memory_tracker.stop()

        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        if self.monitoring_log_path:
            append_jsonl(
                {
                    "event": "server_aggregate_fit",
                    "round": server_round,
                    "aggregation_time_sec": timer.elapsed,
                    "num_results": len(results),
                    "num_failures": len(failures),
                    **memory_metrics,
                },
                self.monitoring_log_path,
            )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        memory_tracker = MemoryTracker("server_aggregate_evaluate_memory")
        memory_tracker.start()

        with Timer("server_aggregate_evaluate") as timer:
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )

        memory_metrics = memory_tracker.stop()

        if self.monitoring_log_path:
            append_jsonl(
                {
                    "event": "server_aggregate_evaluate",
                    "round": server_round,
                    "aggregation_time_sec": timer.elapsed,
                    "num_results": len(results),
                    "num_failures": len(failures),
                    "loss": aggregated_loss,
                    **memory_metrics,
                },
                self.monitoring_log_path,
            )

        return aggregated_loss, aggregated_metrics
