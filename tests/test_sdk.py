import sys
import os
sys.path.insert(0, os.path.abspath('sdk/python'))
import pytest
import os
from unittest.mock import MagicMock
from crowddna.client import CrowdDNAClient
from crowddna.configuration import TestingConfiguration
from crowddna.authentication import APIKeyProvider
from crowddna.retry import RetryPolicy
from crowddna.builders import WorkflowRequestBuilder, ValidationError
from crowddna.events import EventSystem
from crowddna.upload import UploadManager
from crowddna.download import DownloadManager
from crowddna.streaming import StreamingClient
from crowddna.pagination import Paginator

def test_client_init():
    config = TestingConfiguration()
    auth = APIKeyProvider("test_key")
    client = CrowdDNAClient(config, auth)
    assert client.config.base_url == "http://testserver"

def test_retry_idempotency():
    config = TestingConfiguration()
    config.max_retries = 3
    policy = RetryPolicy(config)
    assert policy.is_retryable("GET", {}, 503) is True
    assert policy.is_retryable("POST", {}, 503) is False
    assert policy.is_retryable("POST", {"Idempotency-Key": "123"}, 503) is True
    assert policy.is_retryable("GET", {}, 429) is True

def test_middleware_hooks():
    config = TestingConfiguration()
    auth = APIKeyProvider("test_key")
    client = CrowdDNAClient(config, auth)
    
    flag = {"called": False}
    def mock_interceptor(req):
        flag["called"] = True
        return req
        
    client.middleware.add_request_interceptor(mock_interceptor)
    
    client.transport.connection.request = MagicMock()
    client.transport.connection.request.return_value = MagicMock(status_code=200)
    
    client.transport.send("GET", "/test")
    assert flag["called"] is True
    
def test_event_system_safety():
    sys = EventSystem()
    flag = {"called": False}
    
    def bad_callback():
        raise Exception("boom")
        
    def good_callback():
        flag["called"] = True
        
    sys.register("on_request", bad_callback)
    sys.register("on_request", good_callback)
    
    sys.trigger("on_request")
    assert flag["called"] is True

def test_builder_validation():
    b = WorkflowRequestBuilder()
    with pytest.raises(ValidationError):
        b.build()
        
    assert b.with_id("123").build() == {"id": "123"}
    
def test_paginator():
    transport = MagicMock()
    transport.send.return_value.json.side_effect = [
        {"items": [1, 2], "next_cursor": "c1"},
        {"items": [3], "next_cursor": None}
    ]
    paginator = Paginator(transport, "/test")
    items = list(paginator)
    assert items == [1, 2, 3]

def test_streaming():
    transport = MagicMock()
    resp = MagicMock()
    resp.iter_lines.return_value = [b"id: 1", b"data: test", b""]
    transport.send.return_value = resp
    
    stream = StreamingClient(transport, TestingConfiguration())
    events = list(stream.listen("/stream"))
    assert "id: 1" in events
    
def test_upload():
    transport = MagicMock()
    mgr = UploadManager(transport, TestingConfiguration())
    
    with open("test.txt", "w") as f:
        f.write("test")
        
    mgr.upload_file("/upload", "test.txt")
    os.remove("test.txt")
    transport.send.assert_called()

def test_download():
    transport = MagicMock()
    resp = MagicMock()
    resp.headers = {"content-length": "4"}
    resp.iter_content.return_value = [b"test"]
    transport.send.return_value = resp
    
    mgr = DownloadManager(transport, TestingConfiguration())
    mgr.download_file("/download", "test.txt")
    
    with open("test.txt", "r") as f:
        assert f.read() == "test"
    os.remove("test.txt")
