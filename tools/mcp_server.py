# -*- coding: utf-8 -*-
"""Minimal MCP-style stdio server exposing the 11 tools as JSON-RPC 2.0.

Production: replace with `mcp.server.FastMCP`; this file is dependency-free
and is enough for an end-to-end local demo / integration test.
"""
import json
import sys

from tools.registry import BINDINGS

FUNCS = {spec["name"]: func for spec, func in BINDINGS}
SPECS = {spec["name"]: spec for spec, _ in BINDINGS}


def list_tools():
    return [{"name": s["name"], "description": s["description"],
             "inputSchema": s["input_params"]} for s in SPECS.values()]


def handle(req):
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "h2-tools", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name not in FUNCS:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown tool"}}
        try:
            result = {"content": [{"type": "text", "text": json.dumps(FUNCS[name](**args), ensure_ascii=False)}]}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(exc)}}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle(req)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
