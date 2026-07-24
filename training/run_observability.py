import time
from training.observability.cli import parse_args
from training.observability.metadata import MonitoringSession, MonitoringContext
from training.observability.engine import ObservabilityEngine
from training.observability.health import HealthMonitor
from training.observability.profiler import ProfilerEngine
from training.observability.registry import ObservabilityRegistry

import training.observability.collectors.cpu  # noqa: F401
import training.observability.collectors.memory  # noqa: F401
import training.observability.collectors.gpu  # noqa: F401
import training.observability.collectors.latency  # noqa: F401
import training.observability.collectors.queue  # noqa: F401
import training.observability.exporters.json  # noqa: F401
import training.observability.exporters.csv  # noqa: F401
import training.observability.exporters.markdown  # noqa: F401
import training.observability.plotting.runtime  # noqa: F401
import training.observability.plotting.latency  # noqa: F401
import training.observability.plotting.memory  # noqa: F401
import training.observability.plotting.throughput  # noqa: F401

def main():
    args = parse_args()
    
    if args.action == "health":
        print(f"Running health checks for session {args.session_id}...")
        from training.observability.buffer import ObservationBuffer
        buf = ObservationBuffer()
        monitor = HealthMonitor(buf)
        monitor.check_all()
        for obs in buf.flush():
            print(f"[{obs.severity.name}] {obs.component}: {obs.message}")
            
    elif args.action == "profile":
        print("Running short profile...")
        profiler = ProfilerEngine()
        profiler.start()
        # dummy work
        sum(i * i for i in range(1000000))
        stats = profiler.stop()
        print(stats)
        
    elif args.action in ["monitor", "report"]:
        print(f"Starting {args.action} for session {args.session_id}...")
        
        context = MonitoringContext(
            session_id=args.session_id,
            enabled_collectors=("cpu_collector", "memory_collector", "gpu_collector", "latency_collector", "queue_collector"),
            enabled_exporters=("json_exporter", "csv_exporter", "markdown_exporter", "latency_plotter", "memory_plotter", "runtime_plotter", "throughput_plotter"),
            enabled_tracing=True,
            sampling_interval=1.0,
            retention_policy="keep_all",
            output_directory=args.output
        )
        session = MonitoringSession(session_id=args.session_id, start_time=time.time(), context=context)
        engine = ObservabilityEngine(session)
        
        # Initialize exporters/plotters
        exporters = []
        for name in context.enabled_exporters:
            cls = ObservabilityRegistry.get_module(name)
            mod = cls()
            mod.initialize(output_dir=args.output)
            exporters.append(mod)
            
        engine.start()
        print("Engine started. Running for 3 seconds...")
        for _ in range(3):
            engine.process_tick()
            time.sleep(1)
            
        print("Shutting down engine...")
        engine.shutdown()
        
        observations = engine.aggregator.history
        print(f"Exporting {len(observations)} observations...")
        for exp in exporters:
            exp.export(args.session_id, observations)
            
        print(f"Done. Check {args.output} for artifacts.")

if __name__ == "__main__":
    main()
