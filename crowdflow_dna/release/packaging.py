class PackagingManager:
    def __init__(self, builder):
        self.builder = builder
        
    def assemble(self, version: str):
        return [
            self.builder.build_wheel(version),
            self.builder.build_sdist(version)
        ]
