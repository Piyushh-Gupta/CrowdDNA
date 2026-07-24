import pytest
from training.observability.metadata import (
    MetricObservation, EventObservation, EventPriority, 
    MonitoringContext, MonitoringSession
)
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock
from training.observability.events import EventBus
from training.observability.engine import ObservabilityEngine
from training.observability.sampling import AlwaysSample, EveryNSample
import training.observability.collectors.cpu  # noqa: F401
import training.observability.collectors.memory  # noqa: F401

def test_observation_immutability():
    obs = MetricObservation(timestamp=1.0, name="cpu", value=50.0, unit="%")
    with pytest.raises(Exception): # dataclass frozen raises FrozenInstanceError
        obs.value = 60.0

def test_observation_buffer():
    buf = ObservationBuffer(maxlen=2)
    obs1 = MetricObservation(1.0, "m", 1.0, "u")
    obs2 = MetricObservation(2.0, "m", 2.0, "u")
    obs3 = MetricObservation(3.0, "m", 3.0, "u")
    
    buf.add(obs1)
    buf.add(obs2)
    buf.add(obs3) # Should evict obs1
    
    assert len(buf) == 2
    flushed = buf.flush()
    assert len(flushed) == 2
    assert flushed[0].value == 2.0
    assert flushed[1].value == 3.0
    assert len(buf) == 0

def test_event_bus():
    buf = ObservationBuffer()
    bus = EventBus(buf)
    
    bus.emit("test_event", EventPriority.HIGH, "message")
    assert len(buf) == 1
    
    flushed = buf.flush()
    assert isinstance(flushed[0], EventObservation)
    assert flushed[0].priority == EventPriority.HIGH

def test_sampling_policies():
    always = AlwaysSample()
    assert always.should_sample() is True
    
    every_n = EveryNSample(2)
    assert every_n.should_sample() is False
    assert every_n.should_sample() is True
    assert every_n.should_sample() is False

def test_engine_initialization():
    context = MonitoringContext(
        session_id="test_sess",
        enabled_collectors=("cpu_collector", "memory_collector"),
        enabled_exporters=(),
        enabled_tracing=False,
        sampling_interval=1.0,
        retention_policy="test",
        output_directory="test_dir"
    )
    session = MonitoringSession(session_id="test_sess", start_time=Clock.now(), context=context)
    
    engine = ObservabilityEngine(session)
    assert len(engine.active_modules) == 2
    
    engine.start()
    engine.process_tick()
    engine.shutdown()
    
    # We should have collected some metrics from cpu and memory
    assert len(engine.aggregator.history) > 0
