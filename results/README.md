# Results

There are subdirectories for each experiment: 

* For each usecase
	* `adversarial/`
	* `fairness/`
	* `semantic/`
* For performance comparison between backends:
	* `dimcomp/` - In dimensions
	* `performance/` - In samples
	* `par/` - When parallelized

There are also some utility scripts.

* `get_stats.py` - A script to extract core statistics of several runs, used to produce e.g. adversarials/performance.txt
* `get_table_data.py` - Pretty printing and inferring per-sample statistics from a performance.txt file
* `mem_scatter.py` -  A script to create a scatterplot of the memory usage of various workers.

