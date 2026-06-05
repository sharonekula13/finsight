"""List the models your API key can access (diagnostic).

    python -m finsight.list_models

Useful if embedding/chat calls 404 -- it shows the exact model names and
which methods (embedContent / generateContent) each supports.
"""
from __future__ import annotations

from .config import settings


def main() -> None:
    if not settings.gemini_api_key:
        print("No GEMINI_API_KEY set. In PowerShell:")
        print('  $env:GEMINI_API_KEY="your-key"')
        return
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=settings.gemini_api_key)
    print("Models available to your key:\n")
    for m in genai.list_models():
        methods = ", ".join(getattr(m, "supported_generation_methods", []) or [])
        print(f"  {m.name}")
        if methods:
            print(f"      methods: {methods}")


if __name__ == "__main__":
    main()
