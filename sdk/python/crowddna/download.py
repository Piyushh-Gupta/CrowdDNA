class DownloadManager:
    def __init__(self, transport, config):
        self.transport = transport
        self.config = config
        
    def download_file(self, path, destination, callback=None):
        resp = self.transport.send("GET", path, stream=True)
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=self.config.download_buffer_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if callback:
                        callback(downloaded, total)
        return True
