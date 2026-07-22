Let's understand how a custom layer in Keras works. You use custom layers for different scenarios, for example when creating the model architecture.
Let's understand the following snippet example:

```python
class HeadDense(tf.keras.layers.Layer):
    def __init__(self, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
    def build(self, input_shape):
        self.length = input_shape[1]
        self.dim = input_shape[2]
        self.dense = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.head_dim), activation = 'swish', bias_axes = 'd')
    def call(self, x):
        x = self.dense(x)
        return x

```
You have created that custom layer and then when creating the model architecture you called like:
```python
x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################

        ################ Final section ################
        x = HeadDense(head_dim)(x)
        x = GLUMlp(head_dim*2,head_dim)(x)
```

When I do:
```python
x = HeadDense(head_dim)(x)
```
I am actually doing **two separate operations**
```python
layer = HeadDense(head_dim) #Initializing the class "HeadDense"
x = layer(x) #Now, calling `Build()` and later `call()`
```
When **Build()** is called keras gets the input shape of the data. For example, Keras internally does:
```python
if not layer.built():
	layer.build(x.shape)
output = layer.call(x)
```
In **Build()** is when keras recieved the input data from *x* and assigns the dimensions or antoher important information.
With **call()** the actual *==forward pass==* happens.

**IMPORTANT** Only **Build()** receives the input shape.

## What would happen of a custom layer WITHOUT Build?
Custome layers can be created and not use **Build** and only **init** and **call**. Built is created or used only when **trainable weights** are defined in that custom layer. This is because Build requires the dimension of the input. A layer without Build can only perform operations of transformation but not of training. Examples of transformation can be vector reshape, concatenation, transpose.
### Layers WITHOUT build()
"Pure functions"
``` markdown 
output = f(input)
```
### Layers WITH build()
"Functions *and* learned parameters"
```markdown
output = f(input,w)
```
where ```w``` is created in *build()* 