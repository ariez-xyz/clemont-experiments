# Ideal would be something like:

from aimon import monitor
from aimon.backends.faissbf import FaissBruteForce

#backend = backends.bdd(data)

backend = FaissBruteForce(df, pred, eps)

m = monitor.Monitor(backend)

for row in df.iterrows:
    m.observe(row)


