import sys
from pathlib import Path

# Add src/ so that `import autopilot` resolves when running without pip install -e .
sys.path.insert(0, str(Path(__file__).parent / "src"))
