## Reference
Vaswani, A. _et al._ Attention is All you Need. in _Advances in Neural Information Processing Systems_ vol. 30 (Curran Associates, Inc., 2017).

## General:
This paper introduces the concept of #Transformer architecture, a neural network model for sequence transduction tasks such as machine translation.
It removes #recurrent and #convolutional structures in favor of #attention mechanisms.

- Recurrent neural networks #RNN or #convolutional struggled with long-range dependencies and parallelization.
- #self-attention allows model dependencies across entire sequences efficiently and in parallel.
## Model architecture
![The transformer, model architecture](../figures/Pasted%20image%2020260607124038.png)
In the left side of Fig1 the basic block of the transformer is presented. From bottom to top, the inputs are words so they are #token ized and then compute their embeddings. This means, conver the words into a vector and then projected into a space. Layer these embedded words are given to the *multi-head attention*, which is one of the novelties of this paper. The multi-head attention is show in the next figure:
![Multi-head attention](Pasted%20image%2020260607124404.png)
The right part of the figure shows the multi-head attention structure while the left-side part shows the basic blocks of the multi-head attention.
This left side is the computation and the forming blocks of the whole structure. the *scaled dot-product attention* computes the *self-attention* as $$\text{Attention(Q,K,V)} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$ 
Where Q, K and V are matrices that have different roles. *Q* is the *query* matrix, *K* is the *Key* matrix, and *V* is the *value* matrix. Let's understand how these matrices work with an example:
Given the phrase: **The dog plays in the park**, each word is splitted (into the embedding representation), for example
The embedding matrix is multiplied by three learned weight matrices $$Q = XW_Q$$, $$K=XW_K$$, $$V=XW_V$$
The resulting Q; k; and V matrices are used to compute attention, where X is the matrix for the phrase. Additionally, the weight in these matrices are **trainable** parameters. The meaning of these matrices is:
- **Query (Q):** What information I am looking for?
- **Key (K):** What information do I contain?
- **Value (V):** What information should be passes forward if I am considered relevant?
Suppose we want to compute the new representation for the word **plays**. The **Query** vector of "plays" is compared with the **Key** vectors of all words in the sentence
$$[  
Q_{\text{plays}}  
\cdot  
K_{\text{the}}  
]

[  
Q_{\text{plays}}  
\cdot  
K_{\text{dog}}  
]

[  
Q_{\text{plays}}  
\cdot  
K_{\text{plays}}  
]

[  
Q_{\text{plays}}  
\cdot  
K_{\text{in}}  
]

[  
Q_{\text{plays}}  
\cdot  
K_{\text{park}}  
]$$

The dot products measure how relevant each word is to "plays". Larger values indicate a stronger relationship. The result is transformed into probabilities using the **softmax** function $$\text{softmax}  
\left(  
\frac{QK^T}{\sqrt{d_k}}  
\right)V  
$$
The resulting vector becomes the **new representation** of the word "plays" **enriched with contextual information from the entire sentence**.
- Now the **Multi-head attention** applies the above procedure but multiple times, using different weights for each **head** of the involved matrices. In this way, when the model is trained, the weights are updated, but from different paths enriching the performance of the model.

