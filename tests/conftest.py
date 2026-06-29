import sys
from pathlib import Path

# Make `import airfoil_rbf` work without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
