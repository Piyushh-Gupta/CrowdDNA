from training.framework.cli import parse_args
from training.framework.metadata import FrameworkMetadata, FrameworkRuntime
from training.framework.context import FrameworkContext
from training.framework.plugins.registry import PluginManifestRegistry
from training.framework.plugins.discovery import PluginDiscovery
from training.framework.resolver import ConfigurationMerger, ConfigurationResolver
from training.framework.validation import ConfigurationValidator
from training.framework.dependency import DependencyGraph
from training.framework.plugins.loader import PluginLoader

def main():
    args = parse_args()
    
    print("Initializing Framework...")
    # 1. Framework Context
    meta = FrameworkMetadata(framework_version="1.0.0", crowddna_version="1.0.0")
    runtime = FrameworkRuntime(session_id="val_session", start_time=0.0, metadata=meta)
    _ = FrameworkContext(runtime=runtime, environment="dev")
    
    # 2. Plugin Discovery
    print("Discovering Plugins...")
    discovery = PluginDiscovery(directories=["training/framework/plugins"])
    discovery.discover(framework_api_version="1.0.0")
    
    # 3. Configuration Loading & Resolution
    print("Loading Configuration...")
    merger = ConfigurationMerger()
    merger.add_layer('default', {"example": "value"})
    merger.add_layer('cli', {"cli_param": "override"})
    bundle = merger.merge(configuration_version="1.0")
    bundle = ConfigurationResolver.resolve(bundle)
    
    # 4. Validation
    print("Validating Configuration...")
    report = ConfigurationValidator.validate_all(bundle)
    if not report.is_valid:
        print(f"Validation failed: {report.errors}")
        return
    print("Validation passed.")
    
    # Freeze config (bundle is frozen by dataclass)
    
    # 5. Dependency Graph
    print("Constructing Dependency Graph...")
    graph = DependencyGraph()
    for name, p_meta in PluginManifestRegistry.get_all_metadata().items():
        graph.add_plugin(p_meta)
        
    try:
        order = graph.resolve_topological_order()
    except Exception as e:
        print(f"Dependency graph error: {e}")
        return
        
    print(f"Topological Order: {[m.name for m in order]}")
    
    # 6. Dry Run Check
    if args.dry_run:
        print("Dry run successful. Framework is correctly configured.")
        return
        
    # 7. Loading
    print("Loading and Initializing Plugins...")
    loader = PluginLoader(context_provider=lambda name: None) # Context dummy for now
    loader.load_and_initialize(order)
    
if __name__ == "__main__":
    main()
