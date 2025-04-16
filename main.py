import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src")) #wanted to simplify running the project, idk if this is the best way

from src.experiments.train import main

if __name__ == "__main__":
    main()
