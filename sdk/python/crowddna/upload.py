import os

class UploadManager:
    def __init__(self, transport, config):
        self.transport = transport
        self.config = config
        
    def upload_file(self, path, filepath, callback=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        
        size = os.path.getsize(filepath)
        uploaded = 0
        
        with open(filepath, 'rb') as f:
            while uploaded < size:
                chunk = f.read(self.config.upload_chunk_size)
                if not chunk:
                    break
                
                # Resumable mock logic
                headers = {
                    "Content-Range": f"bytes {uploaded}-{uploaded+len(chunk)-1}/{size}"
                }
                
                self.transport.send("PUT", path, data=chunk, headers=headers)
                uploaded += len(chunk)
                
                if callback:
                    callback(uploaded, size)
                    
        return True
