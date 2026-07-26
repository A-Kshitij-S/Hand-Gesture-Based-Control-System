#!/usr/bin/env python3
"""
Tiny local server for the fish-tank-vr experiment.

Webcam APIs (`getUserMedia`) are blocked on `file://` URLs, so the page
must be served over `http://localhost`. Run this script from the project
folder and visit http://localhost:8000.

Usage:
    python3 serve.py            # serves on port 8000
    python3 serve.py 8080       # custom port
"""

import http.server
import socketserver
import os
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Disable caching so edits to index.html show up immediately."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # Allow webcam in case the page is loaded inside an iframe
        self.send_header("Permissions-Policy", "camera=(self)")
        super().end_headers()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"Serving fish-tank-vr at {url}")
        print("Press Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
