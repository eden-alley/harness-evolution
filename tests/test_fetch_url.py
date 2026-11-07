"""HTTP 工具边界：真实 urllib 调度与本地假传输，不建立网络连接。"""
import contextlib
from email.message import Message
from http.client import HTTPConnection, HTTPResponse
import io
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request
import urllib.response

from scenario_support import OfflineScenario, permissions, tools


class Body(io.BytesIO):
    def __init__(self, content=b""):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


@contextlib.contextmanager
def http_fixture(responses, *, proxies=None):
    pending = iter(responses)
    requests = []

    def do_open(handler, connection_type, request, **kwargs):
        requests.append((request.full_url, request.host, request.timeout))
        connection = HTTPConnection("offline.invalid")
        connection._validate_path(request.selector)
        connection._encode_request("GET " + request.selector + " HTTP/1.1")
        item = next(pending)
        if isinstance(item, BaseException):
            raise item
        code, headers, body = item
        message = Message()
        for key, value in headers.items():
            message[key] = value
        response = urllib.response.addinfourl(body, message, request.full_url, code)
        response.msg = "offline response"
        return response

    with patch.object(urllib.request, "getproxies", return_value=proxies or {}), \
         patch.object(urllib.request, "proxy_bypass", return_value=False), \
         patch.object(urllib.request.AbstractHTTPHandler, "do_open", new=do_open), \
         patch.object(urllib.request, "_opener", None):
        yield requests


class FetchUrlScenarios(OfflineScenario):
    def test_redirects_keep_standard_library_path_encoding(self):
        self.agent.permissions.set_auto_approve(True)
        for location, target in [
            ("/some file", "http://public.invalid/some%20file"),
            ("/caf\u00e9", "http://public.invalid/caf%E9"),
        ]:
            with self.subTest(location=location):
                redirect_body, final = Body(b"not-drained"), Body(b"done")
                with http_fixture([(302, {"Location": location}, redirect_body),
                                   (200, {}, final)]) as requests:
                    result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/start"})
                self.assertTrue(result.ok, result)
                self.assertEqual([item[0] for item in requests],
                                 ["http://public.invalid/start", target])
                self.assertEqual(redirect_body.read_sizes, [])
                self.assertTrue(redirect_body.closed)
                self.assertTrue(final.closed)

    def test_real_http_response_does_not_report_early_eof_as_success(self):
        class FakeSocket:
            def __init__(self, data):
                self.data = data

            def makefile(self, *args, **kwargs):
                return io.BytesIO(self.data)

        self.agent.permissions.set_auto_approve(True)
        for headers, body, complete in [
            (b"Content-Length: 100\r\n", b"partial", False),
            (b"Content-Length: 7\r\n", b"partial", True),
            (b"Connection: close\r\n", b"partial", True),
            (b"Transfer-Encoding: chunked\r\n", b"10\r\npartial", False),
        ]:
            with self.subTest(headers=headers):
                response = HTTPResponse(FakeSocket(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body))
                response.begin()
                opened = []

                def open_response(request, timeout):
                    opened.append(request.full_url)
                    return response

                with patch.object(urllib.request, "build_opener",
                                  return_value=SimpleNamespace(open=open_response)):
                    result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/body"})
                self.assertEqual(result.ok, complete, result)
                if complete:
                    self.assertEqual(result.data, "partial")
                else:
                    self.assertEqual(result.code, "execution_error")
                    self.assertIsNone(result.data)
                self.assertTrue(response.isclosed())
                self.assertEqual(opened, ["http://public.invalid/body"])

    def test_non_http_urls_are_denied_before_asking_or_opening(self):
        asked = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: (asked.append(name) or "y", ""))
        bad_urls = [(self.workspace / "not-present.txt").as_uri(),
                    "data:text/plain,ignored", "ftp://offline.invalid/file", "https:///missing-host"]
        with http_fixture([]) as requests:
            for url in bad_urls:
                with self.subTest(url=url):
                    result = self.agent._invoke_tool("fetch_url", {"url": url})
                    self.assertFalse(result.ok)
                    self.assertEqual(result.code, "policy_denied")
            self.assertEqual(requests, [])
        self.assertEqual(asked, [])
        self.agent.permissions.set_auto_approve(True)
        result = self.agent._invoke_tool("fetch_url", {"url": bad_urls[0]})
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "policy_denied")

    def test_host_network_range_proxy_and_permission_modes_remain_available(self):
        asked = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: (asked.append(name) or "y", ""))
        urls = ["http://public.invalid/page", "https://public.invalid/page",
                "http://localhost:8080/api", "http://127.0.0.1:8080/api",
                "http://192.168.1.20/api", "http://[::1]:8080/api"]
        for index, url in enumerate(urls):
            self.agent.permissions.set_auto_approve(index != 0)
            body = Body("本地样本".encode())
            with http_fixture([(200, {}, body)]) as requests:
                result = self.agent._invoke_tool("fetch_url", {"url": url})
            self.assertTrue(result.ok, result)
            self.assertEqual(result.data, "本地样本")
            self.assertEqual(requests, [(url, urllib.request.Request(url).host, 15)])
            self.assertEqual(body.read_sizes, [2 * 1024 * 1024 + 1])
            self.assertTrue(body.closed)
        self.assertEqual(asked, ["fetch_url"])
        body = Body(b"proxy-response")
        with http_fixture([(200, {}, body)], proxies={"http": "http://proxy.invalid:8080"}) as requests:
            result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/proxy"})
        self.assertTrue(result.ok, result)
        self.assertEqual(requests, [("http://public.invalid/proxy", "proxy.invalid:8080", 15)])
        self.assertTrue(body.closed)

    def test_redirect_limits_recheck_protocol_and_close_without_draining(self):
        self.agent.permissions.set_auto_approve(True)
        bodies = [Body(b"do-not-drain") for _ in range(5)]
        final = Body(b"done")
        rows = [(302, {"Location": f"/page-{index + 1}"}, bodies[index]) for index in range(5)]
        with http_fixture(rows + [(200, {}, final)]) as requests:
            result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/page-0"})
        self.assertTrue(result.ok, result)
        self.assertEqual(result.data, "done")
        self.assertEqual(len(requests), 6)
        self.assertTrue(all(body.closed and not body.read_sizes for body in bodies))
        self.assertTrue(final.closed)

        for target, expected in [("/loop", "output_limit_exceeded"),
                                 ("file:///ignored", "policy_denied"),
                                 ("ftp://offline.invalid/file", "policy_denied")]:
            bodies = [Body(b"not-drained") for _ in range(6)]
            with http_fixture([(302, {"Location": target}, body) for body in bodies]) as requests:
                result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/loop"})
            with self.subTest(target=target):
                self.assertFalse(result.ok)
                self.assertEqual(result.code, expected)
                used = 6 if target == "/loop" else 1
                self.assertEqual(len(requests), used)
                self.assertTrue(all(body.closed and not body.read_sizes for body in bodies[:used]))
        body = Body(b"redirect-without-location")
        with http_fixture([(302, {}, body)]) as requests:
            result = self.agent._invoke_tool("fetch_url", {"url": "http://public.invalid/no-location"})
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "execution_error")
        self.assertTrue(body.closed)
        self.assertEqual(len(requests), 1)

    def test_actual_read_limit_overrides_headers_and_failures_do_not_retry(self):
        self.agent.permissions.set_auto_approve(True)
        for length, expected_ok in [(2 * 1024 * 1024, True), (2 * 1024 * 1024 + 1, False)]:
            body = Body(b"x" * length)
            with http_fixture([(200, {"Content-Length": "0"}, body)]) as requests:
                result = self.agent._invoke_tool("fetch_url", {"url": "https://public.invalid/large"})
            with self.subTest(length=length):
                self.assertEqual(result.ok, expected_ok)
                if expected_ok:
                    self.assertEqual(len(result.data), length)
                else:
                    self.assertEqual(result.code, "output_limit_exceeded")
                    self.assertIsNone(result.data)
                self.assertEqual(body.read_sizes, [2 * 1024 * 1024 + 1])
                self.assertTrue(body.closed)
                self.assertEqual(len(requests), 1)
        with http_fixture([TimeoutError("offline timeout")]) as requests:
            result = self.agent._invoke_tool("fetch_url", {"url": "https://public.invalid/slow"})
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "execution_error")
        self.assertEqual(len(requests), 1)
        body = Body(b"not-consumed-error")
        with http_fixture([(403, {}, body)]) as requests:
            result = self.agent._invoke_tool("fetch_url", {"url": "https://public.invalid/forbidden"})
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "execution_error")
        self.assertTrue(body.closed)
        self.assertEqual(body.read_sizes, [])

