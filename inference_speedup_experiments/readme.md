# inference implementation

- [understanding tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)
- csr: compressed sparse row matrix
- sklearn implementation: loops over samples and traces each one's decision path
- ways to speed up
  - use low-bit params (e.g. binary)
- first / last lay are just indexing arrays
- extensions
  - categorical vars
  - non-binary splits
  - work on forests / gbms?
  - faster to do np.vectorize(dict.get)(idxs) or arr[idxs]?
    - The [`vectorize`](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.vectorize.html#numpy.vectorize) function is provided primarily for convenience, not for performance. The implementation is essentially a for loop
```python
for i in range(n_samples):
  node = self.nodes
  # While node not a leaf
  while node.left_child != _TREE_LEAF:
    # ... and node.right_child != _TREE_LEAF:
    if X_ndarray[i, node.feature] <= node.threshold:
      node = &self.nodes[node.left_child]
    else:
    	node = &self.nodes[node.right_child]

  out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
```