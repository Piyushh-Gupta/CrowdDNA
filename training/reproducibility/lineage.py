from typing import List, Tuple

class LineageManager:
    """Manages the experiment lineage graph."""
    
    @staticmethod
    def construct_lineage(parents: List[str]) -> Tuple[str, ...]:
        """Constructs a lineage tuple from parent hashes/UUIDs."""
        if not parents:
            return ()
        return tuple(sorted(parents))
