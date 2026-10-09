#!/usr/bin/env python3
"""
Auslan Sign Language Recognition - Web Application Entry Point
Starts the Flask server to host the interactive frontend and inference API.
"""

import sys
from pathlib import Path

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from server import app, load_cached_model, index_dataset

if __name__ == "__main__":
    print("=" * 60)
    print("  AUSLAN SIGN LANGUAGE RECOGNITION - WEB APPLICATION")
    print("  Problem 118 | UE24CS352A Machine Learning Mini-Project")
    print("=" * 60)
    print("Initializing trained model and indexing sign dataset...")
    load_cached_model()
    index_dataset()
    port = 5001
    print(f"\n✨ Web App Live at: http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
