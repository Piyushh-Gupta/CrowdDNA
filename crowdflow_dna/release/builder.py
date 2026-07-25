import os

class Builder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def build_wheel(self, version: str) -> str:
        path = os.path.join(self.output_dir, f"crowddna-{version}-py3-none-any.whl")
        with open(path, 'w') as f:
            f.write("mock wheel content")
        return path
        
    def build_sdist(self, version: str) -> str:
        path = os.path.join(self.output_dir, f"crowddna-{version}.tar.gz")
        with open(path, 'w') as f:
            f.write("mock sdist content")
        return path
        
    def generate_docker_metadata(self, version: str) -> dict:
        return {"image": "crowddna", "tag": version}
