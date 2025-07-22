# phenoct
Utilities for analysing data from the PhenoCT system at The Plant Accelerator.



# Installation
`pip install git+https://github.com/aus-plant-phenomics-facility/phenoct.git@v1.3.1`


examples are in the tests.ipynb Notebook.

# Development
```python
import os
import sys
module_path = os.path.abspath("/Users/a1132077/development/")
if module_path not in sys.path:
    sys.path.append(module_path)
from phenoct.src import phenoct as pct

```
