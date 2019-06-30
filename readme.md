## Bridging random forests and deep neural networks.
**Code here converts sklearn tree -> pytorch DNN**


`rf_to_dnn_ex.ipynb` shows a simple example of converting a random forest to a DNN


Both random forests and DNNs are very strong predictive models. This [cool recent paper title "Neural Random Forests"](https://arxiv.org/pdf/1604.07143.pdf) gives a simple algorithm for exactly rewriting any random forest as a sparse neural network. This could be useful then for combining the inductive biases of both and other interesting things.

## how does it work?

![](biau_19_figw.png)

The idea is to rewrite the neural network into 3 layers:

1. The first layer identifies whether a point is on the left or right side of a split (outputs -1, 1)
   - $out_{split} = \text{sign}(in - thresh)$
   - this can be made faster by doing indexing rather than a matrix multiply (maybe sparse tensor will be fast enough)
2. The second layer determines whether a a point is in a leaf or not (ouputs 0/1)
	- $out_{leaf} = (\sum w \cdot in) ==  depth(leaf)$
3. the final layer simply multiplies the vector of (0s/1s) by the value of the leaf
  - $out_{pred} = \sum_{leaf} out_{leaf} \cdot val_{leaf}$
  - the 0-1 helps make it sparser (and simpler :smile:)
  

