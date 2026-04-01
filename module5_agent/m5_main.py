"""
main.py
=======
Module 5 — Entry Point

Usage:
    # Start FastAPI server
    python main.py

    # Generate all personas offline (no server)
    python main.py --generate-personas

    # Generate all nudges offline
    python main.py --generate-nudges

    # Both personas + nudges offline
    python main.py --setup
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Module 5 — Agent Interface")
    parser.add_argument("--generate-personas", action="store_true",
                        help="Generate all cluster personas (offline, no server)")
    parser.add_argument("--generate-nudges", action="store_true",
                        help="Pre-generate all user nudges (offline)")
    parser.add_argument("--setup", action="store_true",
                        help="Run --generate-personas + --generate-nudges then start server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()

    # ── Offline tasks ──────────────────────────────────────────────────
    if args.generate_personas or args.setup:
        print("=" * 55)
        print("Generating cluster personas...")
        print("=" * 55)
        from module5_agent.m5_persona_engine import generate_all_personas
        personas = generate_all_personas()
        print(f"✅ {len(personas)} personas generated\n")

    if args.generate_nudges or args.setup:
        print("=" * 55)
        print("Generating user nudges...")
        print("=" * 55)
        from module5_agent.m5_nudge_engine import generate_all_nudges
        nudges = generate_all_nudges()
        print(f"✅ {len(nudges)} nudges generated\n")

    if args.generate_personas or args.generate_nudges:
        if not args.setup:
            return   # offline mode done, don't start server

    # ── Start server ───────────────────────────────────────────────────
    import socket

    def _is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) != 0

    port = args.port
    if not _is_port_free(port):
        print(f"⚠️  Port {port} is already in use.")
        for alt in range(port + 1, port + 10):
            if _is_port_free(alt):
                port = alt
                print(f"   → Using port {alt} instead.")
                break
        else:
            print("❌ No free port found in range. Kill existing process or specify --port manually.")
            sys.exit(1)

    print("=" * 55)
    print(f"Starting Module 5 API server")
    print(f"  URL:  http://{args.host}:{port}")
    print(f"  Docs: http://{args.host}:{port}/docs")
    print("=" * 55)

    import uvicorn
    uvicorn.run(
        "module5_agent.m5_api:app",
        host=args.host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
# """
# main.py
# =======
# Module 5 — Entry Point

# Usage:
#     # Start FastAPI server
#     python main.py

#     # Generate all personas offline (no server)
#     python main.py --generate-personas

#     # Generate all nudges offline
#     python main.py --generate-nudges

#     # Both personas + nudges offline
#     python main.py --setup
# """

# import os
# import sys
# import argparse

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# def main():
#     parser = argparse.ArgumentParser(description="Module 5 — Agent Interface")
#     parser.add_argument("--generate-personas", action="store_true",
#                         help="Generate all cluster personas (offline, no server)")
#     parser.add_argument("--generate-nudges", action="store_true",
#                         help="Pre-generate all user nudges (offline)")
#     parser.add_argument("--setup", action="store_true",
#                         help="Run --generate-personas + --generate-nudges then start server")
#     parser.add_argument("--host", type=str, default="0.0.0.0")
#     parser.add_argument("--port", type=int, default=8000)
#     parser.add_argument("--reload", action="store_true", default=False)
#     args = parser.parse_args()

#     # ── Offline tasks ──────────────────────────────────────────────────
#     if args.generate_personas or args.setup:
#         print("=" * 55)
#         print("Generating cluster personas...")
#         print("=" * 55)
#         from module5_agent.m5_persona_engine import generate_all_personas
#         personas = generate_all_personas()
#         print(f"✅ {len(personas)} personas generated\n")

#     if args.generate_nudges or args.setup:
#         print("=" * 55)
#         print("Generating user nudges...")
#         print("=" * 55)
#         from module5_agent.m5_nudge_engine import generate_all_nudges
#         nudges = generate_all_nudges()
#         print(f"✅ {len(nudges)} nudges generated\n")

#     if args.generate_personas or args.generate_nudges:
#         if not args.setup:
#             return   # offline mode done, don't start server

#     # ── Start server ───────────────────────────────────────────────────
#     print("=" * 55)
#     print(f"Starting Module 5 API server")
#     print(f"  URL:  http://{args.host}:{args.port}")
#     print(f"  Docs: http://{args.host}:{args.port}/docs")
#     print("=" * 55)

#     import uvicorn
#     uvicorn.run(
#         "module5_agent.m5_api:app",
#         host=args.host,
#         port=args.port,
#         reload=args.reload,
#     )


# if __name__ == "__main__":
#     main()
