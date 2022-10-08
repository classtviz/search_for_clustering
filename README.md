# Hypercluster
A package for clustering optimization with sklearn + tslearn. 

### Quickstart with python
```python
import pandas as pd
from sklearn.datasets import make_blobs
import hypercluster

data, labels = make_blobs()
data = pd.DataFrame(data)
labels = pd.Series(labels, index=data.index, name='labels')

# With a single clustering algorithm
clusterer = hypercluster.AutoClusterer()
clusterer.fit(data).evaluate(
  methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metrics, 
  gold_standard = labels
  )

# With a range of algorithms

clusterer = hypercluster.MultiAutoClusterer()
clusterer.fit(data).evaluate(
  methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metrics, 
  gold_standard = labels
  )

```
