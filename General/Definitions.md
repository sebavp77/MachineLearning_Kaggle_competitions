Recopilation of Keras functions, common terms, architectures and more.

## Einsumdense
Consider the following case examples:
```python
self.expand = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length,self.channels_expand),activation='linear',bias_axes='d')

self.project = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length,self.channel_size),activation='linear',bias_axes='d')
```
despite `self.expand`and `self.project`are very similar they different in the output, the former expands the number of elements while the later reduces the number of elements.

Mathematically, the `EinSumDense` does: $$y_{abd} = \sum_c x_{abc}W_{cd}$$
then, the dimension `d` which is in the weights and in the output defines the size of the outputs. This is equivalent to apply a feed-forward net

![feed forward basic](../Excalidraw/feed_forward_basic.md)

## Gated Liner Unit (GLU)
It is a neural network building block that regulates information flow using a learned gating mechanism. **It allows networks to dynamically emphasize important features and suppress noise by deciding which information passes through a layer**

### How it works
1. _Splitting the input:_ An input tensor is ==divided into two equal halves== along the *feature dimension*: the information path and the gate.
2. _Applying the gate:_ The second half is passed through a sigmoid activation function, converting the values to a range from 0 to 1.
3. _Element-wise multiplication:_ The two halves are multiplied together. Because values are multiplied by 0 (blocking information) or 1 (passing information), it acts as a soft gate.
Example:
```python
class GLU(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self, x, mask=None):
        x, gate = tf.split(x,2,axis=-1) #splits the tensor into two equal parts along the last dimension. It is assumed: [data_features | gating_features]
        x = x*tf.keras.activations.swish(gate) #Applies the swish activation to the gate: swish(z)=zσ(z), σ: sigmoid function
        # Note: this is used to tell the system which information to keep, controled by the gate, for example: x = actual signal
        # gate = importance score, if gate → small then output ≈ 0 and feature is ignored.
        # Useful when: many input variables exist and only some are important for a given sample -> The gate learns this automatically.
        return x
```


*GLU* is most useful when you perform an expansion first, so the feature dimension in increased, and now you want to define which of these new features are more relevant and are worth to be kept.
```java
models in the Google PaLM family and several recent Transformer variants use **SwiGLU**, which generally performs better than a standard GLU because the gate becomes more expressive and less prone to problematic saturation.
```

## Gated Linear Unit Multi-Layer Perceptron (GLUMlp)
A GLUMlp is a feed-forward neural network that uses a *GLU* instead of a standard activation function such as *ReLU* or *GELU*
In transformer, the feed-forward network usually looks like
```markdown
x -> Linear -> GELU -> Linear
```
A **GLUMlp** replaces the activation with a **learned gate**. An implementation example is
```python
class GLUMlp(tf.keras.layers.Layer):
    def __init__(self, dim_expand, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim_expand = dim_expand
        self.dim = dim
        self.dense_1 = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(None,self.dim_expand),activation='linear',bias_axes='d')
        self.glu_1 = GLU()
        self.dense_2 = tf.keras.layers.EinsumDense("abc,cd->abd", output_shape=(None,self.dim),activation='linear',bias_axes='d')
        
    def call(self, x, training=False):
        x = self.dense_1(x)
        x = self.glu_1(x)
        x = self.dense_2(x)
        return x
```

## Residual connection
It add the input back to the output of a layer. Instead of computing
$$y = f(x)$$
it computes $$y = x + f(x)$$
Without residual connections deep networks suffer from:
1. vanishing gradients
2. exploding gradients
3. optimization becoming increasingly difficult
It makes training very deep networks more stable

## ScaleBias
it add a bias (offset) to the learnable layer, so instead of returning
$$y = x$$
this layer returns
$$y = \alpha x + \beta$$
where $\alpha$ is a learnable scale and $\beta$ is a learnable bias
**Why use it?** 
stabilizes training, reduces sensibility to initialization, allows larger learning rates. ==Unlike Batch Normalization, layer normalization does not depend on the batch size==

## Depthwise 1D convolution

A standard convolution applies multiple kernels to all input channels simultaneously. ==Therefore, every output channel depends on every input channel.==
Conversely, a **depthwise convolution** applies a unique kernel to each channel
```markdown
channel 1
↓
Kernel 1
↓
Output 1
------------
Channel 2
↓
Kernel 2
↓
Output 2
```

Among the **advantages** are:
- fewer parameters than standard convolution
- faster
- preserves channel independence
## Efficient Channel Attention (ECA)
Not every feature channel is equally useful. Suppose a convolution we obtain (each channel is a kernel)
```
Channel 1 → edges

Channel 2 → texture

Channel 3 → noise

Channel 4 → corners
```

the model should learn that, for example, _Edges_ is more important than _Noise_

**How ECA works**
Suppose the output of a convolutional layer is $$X \epsilon R^{L\times C}$$
where L is a sequence length (or time steps) and C is the number of channels.
For example, the feature map is

|Time|Ch1|Ch2|Ch3|Ch4|
|---|--:|--:|--:|--:|
|1|2|5|1|7|
|2|3|4|2|6|
|3|1|6|1|8|
|4|2|5|3|7|
|5|4|4|2|9|


1. Compute one summary value per channel using global average pooling. For our example, channel 1 is: $$\frac{2+3+1+2+4}{5}=2.4$$
	so now we have for each channel
				[2.4, 4.8, 1.8, 7.4]
2. Apply a small 1D convolution to allow neighboring channel to interact. It applies the convolution to this vector of channel summaries. Suppose the kernel is
				[0.25, 0.5, 0.25]
	for channel 2, the convolution computes something like $$0.25*2.4 + 0.5*4.8 + 0.25*1.8 = 3.45$$ Notice that the kernel is applied to the channel 2 and its neighbors. 
3. Apply a sigmoid
			[0.2,0.9,0.1,0.8]
4. Multiply the original feature maps by these weights.
As a result channel 2 (0.9) and 4 are emphasized (0.8) while channel 3 is largely suppressed (0.1).

## Channel Projection
A channel projection changes the **number of feature channels** while preserving the sequence length. Suppose the input tensor has shape
						(100, 64)
meaning
- 100 time steps
- 64 channels (features)
A projection layer can map it to
						(100,256)
Later another projection may reduce it back to 
						(100,64)
**Why expand?**
More channels mean a richer feature space. For example, instead of
```markdown
Temperature
Humidity
Pressure
```
you might create
```markdown
Temperature
Humidity
Pressure
Temp × Humidity 
Pressure gradients 
Local trends
```
**The common patter is**
```markdown
64 channels

↓

Expand

↓

256 channels

↓

Perform complex operations

↓

Project

↓

64 channels
```
