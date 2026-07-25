
export class ResumableUpload {
  file: File;
  chunkSize: number;
  uploadedBytes: number = 0;
  private uploadId: string;
  
  constructor(file: File, chunkSize = 1024 * 1024) {
    this.file = file;
    this.chunkSize = chunkSize;
    this.uploadId = `upload_${file.name}_${file.size}_${file.lastModified}`;
    const state = localStorage.getItem(this.uploadId);
    if (state) {
      const parsed = parseInt(state, 10);
      if (!isNaN(parsed) && parsed <= file.size) {
        this.uploadedBytes = parsed;
      } else {
        localStorage.removeItem(this.uploadId);
      }
    }
  }

  async start(onProgress: (pct: number) => void, signal?: AbortSignal) {
    while (this.uploadedBytes < this.file.size) {
      if (signal?.aborted) throw new Error('Upload aborted');
      const end = Math.min(this.uploadedBytes + this.chunkSize, this.file.size);
      
      // Mock upload fetch chunk logic here
      await new Promise(res => setTimeout(res, 100));
      
      this.uploadedBytes = end;
      localStorage.setItem(this.uploadId, this.uploadedBytes.toString());
      onProgress(Math.round((this.uploadedBytes / this.file.size) * 100));
    }
    localStorage.removeItem(this.uploadId);
  }
}
