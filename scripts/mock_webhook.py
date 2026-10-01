"""
운영자 알림 수신 테스트용 모의 웹훅 서버 (#16 증빙).

실행:
    python scripts/mock_webhook.py --port 9009            # 정상 수신 (200)
    python scripts/mock_webhook.py --port 9009 --fail     # 전송 실패 검증용 (500)
서빙 서버는 AIOPS_ALERT_WEBHOOK_URL=http://localhost:9009 로 띄운다.
받은 알림은 콘솔과 logs/alerts_received.jsonl에 남는다.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = "logs/alerts_received.jsonl"


def make_handler(fail: bool):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            if fail:
                self.send_response(500)
                self.end_headers()
                print("[MOCK] 500 응답 (전송 실패 시나리오)", flush=True)
                return
            payload = json.loads(body)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print("[MOCK] 알림 수신\n" + payload.get("text", body), flush=True)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--fail", action="store_true")
    args = ap.parse_args()
    print(f"[MOCK] http://localhost:{args.port} 대기 중 (fail={args.fail})", flush=True)
    HTTPServer(("0.0.0.0", args.port), make_handler(args.fail)).serve_forever()


if __name__ == "__main__":
    main()
