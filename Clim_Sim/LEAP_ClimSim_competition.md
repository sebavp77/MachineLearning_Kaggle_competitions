# Link to competition
https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim

# Assembling the puzzle
#assembling_puzzle
The approach I will take to learn about this solution is backwards. I will start with the training notebook and move backwards collecting all pieces required to perform the training.

The model training is performed with
```python
history = model.fit(train_ds, verbose=2,
                    validation_data=val_1_ds,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps=val_steps_per_epoch,
                    epochs=10, batch_size=batch_size,
                    callbacks=[validation_callback(val_metric_list), lr_callback, WeightDecayCallback()])
```
From this line I need to create:
1. train_ds
2. val_1_ds
3. steps_per_epoch
4. val_steps_per_epoch
5. batch_size
6. callbacks: 
	1. validation_callback(val_metric_list)
	2. lr_callback
	3. weightDecayCallback
	
Let's start with the more simple variables and then continue with the ones that required more code or definitions.

## steps_per_epoch
This metric is related with the number of `training samples` and the `batch size`. Suppose you have 50.000 training samples and define batch_size =100, then steps_per_epoch=500 means that 500 batches = one epoch. This allows me to define what an epoch is.
It says: "One epoch will have this number of iterations (steps)"

## val_steps_per_epoch
It determines *how many validation batches are processed when validation is run*

**Let's see the relationship between "step_per_epoch" and "val_steps_per_epoch"**
Suppose
```python
steps_per_epoch = 5
validation_steps_per_epoch = 2
batch_size=62
epochs = 10
```
During training:
```python
Training:
Batch 1 (62 samples)
Batch 2 (62 samples)
Batch 3 (62 samples)
Batch 4 (62 samples)
Batch 5 (62 samples)
```
At this point, the epoch is over. **Only then** Keras runs validation
```python
Validation:
Validation batch 1 (62 samples)
Validation batch 2 (62 samples)
```
Then **the epoch ends completely**, and Epoch 2 begins.

## Callbacks
they allow you to **inject your own code at specific moments during the training process** without having to write the training loop yourself.
Consider `model.fit()` as a machine that repeatedly executes (==similar to what you have to write if you were working with **Pytorch**==)
```python
- start training -
  **on_train_begin()**
for each epoch
	- Before epoch starts -
	  **on_epoch_begin()**
	  for each training batch:
		  - Before batch -
		    **on_train_batch_begin()**
		  Forward pass
		  Compute loss
		  Backpropagation
		  Update weights
		  - After batch -
		    **on_train_batch_end()**
	----------------------------------
	 validation starts
	----------------------------------
	on_test_begin()
	for each validation batch:
		on_test_batch_begin()
		
		Forward pass only
		Compute validation loss
		Compute validation metrics
		
		on_test_batch_end()
	on_test_end()
	----------------------------------
	validation ends
	----------------------------------
	 - After epoch -
		**on_epoch_end()**
End training
**on_train_end()**
```
**A callback lets you attach your own functions to any of these events**
**Note** Validation only happens at the end of each **epoch**
**Note** Similarly, there is another set of callback for prediction:
* `on_predict_begin()`
* `on_predict_batch_begin()`
* `on_predict_batch_end()`
* `on_predict_end()`

You can create any **callback** to execute any desired function at one of the points marked by `**...**`. For this you need to create a class that inherits `tf.keras.callbacks.Callback`. Additionally, inside every callback 
`self.model`
is automatically set by Keras, so you can write 
`self.model.optimizer.learning_rate`, for example.

#### Example
```python
class WeightDecayCallback(tf.keras.callbacks.Callback):
	def __init__(self, wd_ratio=WD_RATIO):
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):

        self.model.optimizer.weight_decay = (
            self.model.optimizer.learning_rate
            * self.wd_ratio
        )
```
means
```markdown
Epoch starts

↓

callback runs

↓

weight decay updated

↓

training begins
```

In the first example, where the `.fit()` is performed, the callback `learning_rate` is not a function, different to the other two callbacks, this is because it is a Keras method constructed as:
```python
lr_callback = tf.keras.callbacks.LearningRateScheduler( lambda step: LR_SCHEDULE[step], verbose=0)
```
where `LR_SCHEDULE` is a vector defining the value of the learning rate at each batch.

#### Example 2
`validation_callback(val_metric_list)`
This is a user custom function used as callback. It follows the callback definitions we already covered, as
```python
class validation_callback(tf.keras.callbacks.Callback):
    def __init__(self, metric_list):
        self.metric_list = metric_list
        super().__init__()
    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch+1)%jump == 0 or epoch == 0:
            print('Metric: ------------------')
            metrics_1 = valid_sub_func(self.model, val_1_ds, val_1_labels_2, val_1_labels, val_1_q0002)
            metrics_2 = valid_sub_func(self.model, val_2_ds, val_2_labels_2, val_2_labels, val_2_q0002)
            metrics_3 = valid_sub_func(self.model, val_3_ds, val_3_labels_2, val_3_labels, val_3_q0002)
            metrics_4 = valid_sub_func(self.model, val_4_ds, val_4_labels_2, val_4_labels, val_4_q0002)
            eval = self.model.evaluate(val_4_ds, verbose = 0, batch_size = batch_size, steps = val_steps_per_epoch)
            print(f'evals: {eval}')
```
This callback is called at the end of each epoch (`on_epoch_end`). Keras automatically passes `epoch` and `logs`. The custom callback calls the function `valid_sub_func` at this point during the callback call with the arguments previously defined in the code: `val_x_ds`, `val_x_labels_2`, `val_x_labels`, `val_x_q0002` were `x` goes from 1 to 4.

## Model construction
When creating the `Model`, a bast number of architectures are already implemented and tested. For this competition, the winner model was:
```python
def get_model(input_len,dim=384, head_dim=2048):
    '''
    Creation of model architecture.
    Inputs:
    input_len : int. Lenght of the number of imput features
    dim: int. Number of elements in the spatial dimension for 1 time step
    head_dim:

    Ouput:
    TensorFlow model
    '''
    with strategy.scope(): #tell tensorflow that all models/variables created within should be managed by a distribution strategy (multi-GPU,TPU)
        inp1 = tf.keras.Input([input_len])
        x = inp1
        x = Reshape1(300)(x)
        x = tf.keras.layers.Dense(dim,use_bias=False)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x) #Normalizes each example independently in the batch, epsilon is for stability

        groups = 1
        conv_filter = 15
        ############ Block 1 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 2 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 3 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 4 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################

    	############ Block 5 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 6 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 7 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 8 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################

        ############ Block 9 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 10 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 11 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################
        ############ Block 12 ########################
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim,4,dim*4)(x)
        ############################################

        ################ Final section ################
        x = HeadDense(head_dim)(x)
        x = GLUMlp(head_dim*2,head_dim)(x)
        ###################################
        x_pred = tf.keras.layers.Dense(20)(x)
        x_confidence = tf.keras.layers.Dense(20)(x)

        x = Reshape2()(x_pred,x_confidence)

        #creates a model with input "inp1" and output "x"
        model = tf.keras.Model(inp1,x)
        return model
```
`TransformerEncoder` uses the idea of "attention is all your need". It applies **multi-head self-attention** followed by a **Glu-based feed-forward network (GLUMlp)**. Each sublayer is wrapped with a residual connection, learnable scaling (ScaleBias), and layer normalization, allowing the block to capture *long-range dependencies* while maintaining stable training.

`Conv1DBlockSqueezeformer` is a squeezeformer convolutional block that combines channel expansion, GLU gating, depthwise 1D convolution, efficient channel attention (ECA), and channel projection, to extract local sequential features efficiently. The convolutional path is followed by a GLUMlp, with both stages connected through residual connections, ScaleBias, and layer normalization to improve feature representation and optimization stability. 

## Obtaining training (train_ds) and validation (val_1_ds) datasets
When feeding training and validation data to a TensorFlow model, there are several possible approaches depending on the data format, the available computational resources, and the size of the dataset. A common workflow is to first convert the data into tensors and then create a `tf.data.Dataset`. The dataset pipeline can then be configured to perform operations such as batching, shuffling, caching, prefetching, and parallel data loading. This allows TensorFlow to efficiently feed data to the model during training while optimizing memory usage and CPU/GPU utilization.

This competition is particularly challenging because it involves an exceptionally large dataset, on the order of terabytes. As a result, the entire dataset cannot be loaded into memory at once. Instead, an efficient `tf.data.Dataset` pipeline is required to stream the data in batches, minimizing memory consumption and reducing I/O bottlenecks while maintaining high training throughput.

As mentioned before, we are going backwards in this notebook, starting from the last result and going to the first step that allowed us to obtain it.

### Obtaining train_ds
```python
train_ds, train_labels_ds, train_labels_ds_2, train_ds_q0002 = get_dataset(
    train_tffiles, batch_size, shuffle = 200000, split = 'train',
    to_repeat = True, cache = False)
```
`train_ds` along other variables are obtained by calling the function `get_dataset()` with input arguments:
- train_tffiles
- batch_size
- shuffle
- split
- to_repeat
- cache
```python
def get_dataset(tffiles, batch_size, shuffle, split = 'valid', to_repeat = False, drop_reminder = True, cache = False, valid = False):
'''
This function builds the complete TensorFlow input pipeline used during training or validation.
It reads compressed TFRecord files, decodes each example, applies all the requires preprocessing and feature engineering steps, extracts different sets of labels, batches the data, and finally returns TensorFlow datasets ready to be consumed by the neural network.

The preprocessing pipeline includes:
- Reading compressed TFRecord files
- Decoding serialized examples
- Adding temporal and spatial features
- Optional caching, shuffling, and dataset repetition
- Normalizing the input variables
- Constructing the final input features
- Extracting multiple versions of the target variables
- Padding and baching the data
- Prefetching batches to improve throughput 
  
Inputs:
- tffiles (list[str] or str): TFRecord file(s) containing the dataset
- batch_size (int): Number of samples per batch
- shuffle (int or False): Buffer size used for shuffling. If False or 0, no shuffling is performed
- to_repeat (bool): If True, repeats the dataset indefinitely
- drop_reminder (bool): Whether to discard the last batch if it contains fewer than batch_size samples
- cache (bool): If True, caches the dataset in memory after preprocessing
- valid (bool): passed to get_normed() to indicate whether validation-specific normalization should be used.

---
Outputs:
The function returns four tf.data.Dataset objects:
- ds: Final input dataset that will be fed into the neural network
- ds_label: Dataset containing the original labels
- ds_label_2: Dataset containing an alternative version of the labels
- ds_q002: Dataset containing the q002 state variable, laer used during custom validation


'''
	# --------------------------------------------------------
	# Read compressed TFRecord files. 
	# Multiple files are read in parallel to maximize I/O           #  throughput. 
	# --------------------------------------------------------
    ds = tf.data.TFRecordDataset(
        tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type = 'GZIP').prefetch(tf.data.AUTOTUNE)
        
    # --------------------------------------------------------      # Decode each serialized TFRecord example into tensors. 
    # calls the custom function "decode_tfrec" which decodes
    # the TFRecords
    # --------------------------------------------------------
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    # --------------------------------------------------------      # Add temporal and spatial information (latitude,               # longitude, 
    # cyclic time encodings, etc.) to each sample. 
    # --------------------------------------------------------
    ds = ds.map(lambda x: get_timespace(x, lat_hr, lon_hr, lat, lon, day_time_sin, day_time_cos, sec_time_sin, sec_time_cos), tf.data.AUTOTUNE)
	# --------------------------------------------------------      # Debug mode: 
	# Keep only the first 64 samples for quick testing. 
	# --------------------------------------------------------
    if DEBUG:
        ds = ds.take(64)
	# --------------------------------------------------------      # Cache the dataset in memory to avoid repeating                # preprocessing. 
	# This is useful only if the dataset fits in memory. 
	# -------------------------------------------------------
    if cache:
        ds = ds.cache()
        # Compute the number of cached samples.
        samples_num = ds.reduce(0, lambda x,_: x+1).numpy()
	# --------------------------------------------------------      # Randomly shuffle the dataset.                                 # The argument "shuffle" specifies the shuffle buffer size.     # ---------------------------------------------------------
    if shuffle:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration = True)
    # --------------------------------------------------------      # Repeat the dataset indefinitely.                              # Required when training with steps_per_epoch.                  # --------------------------------------------------------
    if to_repeat:
        ds = ds.repeat()

	# --------------------------------------------------------      # Extract the original labels before modifying the samples.     # --------------------------------------------------------
    ds_label = ds.map(get_labels, tf.data.AUTOTUNE)
    

	# -------------------------------------------------------       # Extract the q002 state variable used later during             # validation.                                                   # --------------------------------------------------------
	ds_q002 = ds.map(get_stateq002, tf.data.AUTOTUNE)
	# --------------------------------------------------------      # Normalize the predictor variables.                            # The "valid" flag controls whether validation                  # normalization should be applied.                              # --------------------------------------------------------
    ds = ds.map(lambda x: get_normed(x, valid), tf.data.AUTOTUNE)
    # --------------------------------------------------------      # Concatenate the normalized features into the final input      # vector.                                                       # --------------------------------------------------------
    ds = ds.map(get_concats, tf.data.AUTOTUNE)

	# --------------------------------------------------------      # Extract an alternative version of the labels after            # preprocessing.                                                # --------------------------------------------------------
    ds_label_2 = ds.map(get_labels_2, tf.data.AUTOTUNE)
	# --------------------------------------------------------      # Build the final input tensor that will be provided to the     # model.                                                        # --------------------------------------------------------
    ds = ds.map(get_final_output, tf.data.AUTOTUNE)

	# --------------------------------------------------------      # Batch the dataset.                                            # Variable-length inputs are padded to fixed dimensions         # before batching.                                              # --------------------------------------------------------
    ds = ds.padded_batch(
                batch_size, padding_values=(PAD, PAD),
        padded_shapes=([col_len+col_not_len],[col_num_y]), drop_remainder=drop_reminder)
	# -------------------------------------------------------       # Batch all auxiliary datasets using the same batch size.       # -------------------------------------------------------
    ds_label = ds_label.batch(batch_size)
    ds_label_2 = ds_label_2.batch(batch_size)
    ds_q002 = ds_q002.batch(batch_size)
	# --------------------------------------------------------      # Prefetch batches while the model is training to overlap       # data loading with computation.                                # -------------------------------------------------------
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # -------------------------------------------------------       # Return all datasets required during training and              # validation.                                                   # --------------------------------------------------------
    return ds, ds_label, ds_label_2, ds_q002
```
The above is a dense and extense function, which gives us the data ready to be fed into the model. Let's expand some parts of this function to understand it better.

**Why some function used lambda while others not?**
```markdown
A `lambda` is used when the function you want to pass does not have the signature (inputs) that `map()` expects.
```
*What does `Dataset,map()`expect?*
__Definition__ The `map()` method applies **one function to every element of the dataset**
Mathematically, if the dataset is $$D = {x_1,x_2,...,x_n}$$
and the mapping function is $$f(x)$$
then the output dataset is $${f(x_1),f(x_2),...,f(x_n)}$$
therefore, the called function **should** expect **one** input, which is what **map** provides.

*When does a lambda become necessary?*
if your function is, for example 
```python
def get_normed(x,valid):
...
```
it needs **two arguments**. The ==lambda== creates a ==new function==, instead of giving TensorFlow `get_normed`, you give it
```python
lambda x:
	get_normed(
		x, valid)
```
Now TensorFlow calls `lambda(x)` 

**What is `ds.reduce(0, lambda x,_: x+1).numpy()` doing?**
A reduction is an operation that combines all the elements of a collection into a single value. It iterates through all elements in the dataset doing whatever the function indicates.
*TensorFlow's `Dataset.reduce()`* the syntax is
```python
dataset.reduce(initial_state,reduce_function)
```
Conceptually
```python
state = initial_state 
for element in dataset: 
	state = reduce_function( state, element ) 
return state
```

The line `ds.reduce(0, lambda x,_: x+1).numpy()` is doing: start at 0, the function receives `x` and `_` where `x` is the variable storing whatever the function does (and initialized to 0 in this case) and `_` is the dataset, which is never used. So this line is counting how many elements are in the dataset. `.reduce` will give the last element.

*Why use this instead of `len(ds)` ?*
Datasets are **streams**, not lists. TensorFlow doesn't have the property `len()` because it doesn't know how many records are inside. For example, the dataset can be `.repeat()` the only way to know how many samples exist is to iterate over the dataset.

==Some other examples==
__sum__:
```python
dataset.reduce(
	0,lambda sum,e: sum+e)
```
`sum` is the variable storing the result (initialized to cero) and e is the value of the dataset at each element (which is walked by `.reduce`)
__max__
```python
dataset.reduce(
	-float("inf"),
	lambda max,e: tf.maximum(max,e) )
```
the initial values is defined to be the lowest possible such that the first value in the dataset is the new max. Later, `reduce` iterates through the dataset asking "is this new value larger than the before ?"

#### Additional functions used in this step
The `get_dataset()` function calls the following functions:
`decode_tfrec`, `get_timespace`, `get_labels`, `get_stateq002`, `get_normed`, `get_concats`, `get_labels_2`, `get_final_output`.
These functions are used to format the data according to the specific structure of this competition. We will go through them mentioning main features and relevant aspects about it.

##### `decode_tfrec()`
This function represents a very common design pattern in modern deep learning. The main steps this function performs are:
1. Read a serialized record (from disk)
2. Decode it into tensors
3. Transform the sample by computing additional features
4. Return a dictionary containing all the variables needed by the model.
This is a ==standard TensorFlow workflow==. The overall workflow is
```
TFRecord
	│ 
	▼
parse_single_example()
	│ 
	▼
decode tensors
	│ 
	▼
feature engineering
	│ 
	▼
normalization
	│ 
	▼
batch
	│ 
	▼
model
```
**Observation** Normally, TFRecords store ==tensors directly== as ==numeric arrays==, for example
```python
float_list
int64_list
bytes_list
```
In this function, however, each tensor has first been serialized using
```python
tf.io.serialize_tensor(...)
```
and ==stored as **string**==. Therefore, the code must later recover it using
```python
tf.io.parse_tensor(...)
```
This is mainly used when tensors have variable shapes or more complex structures, but it is not the most common storage format.

**Let's check the function:**
```python
def decode_tfrec(record_bytes):
	'''
	This function receives one serialized TFRecord example ("record_bytes", a sequence of bytes) and converts it into a Python dictionary containing TensorFlow tensors. It performs the inverse operation of writting a TFRecord.
	---
	Inputs:
	record_bytes: tf.string, one serialized TFRecord example.
	---
	Outputs:
	out: dictionary whose values are TensorFlow tensors.
'''
    schema = {} #Define the schema, each key of the schema is
		    #similar to a "column" of a dataset
	#-- It defines the variables to be read and the type --
	# `VarLenFeature` means: The feature may contain a 
	# variable number of elements (sparse tensor)
    schema["x1"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x2"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x3"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x4"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["time"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["gidx"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["idx"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["batch_index"] = tf.io.VarLenFeature(dtype=tf.int64)
    
    # -TensorFlow's fundamental parsing function-
    #it does: serialized bytes -> Dictionary
    #Now I have {"x1": SparseTensor(...),
	#            "x2": SparseTensor(...)}
    features = tf.io.parse_single_example(record_bytes, schema)
	
	# -Convert sparse tensors into dense tensors-
    data_cols = tf.sparse.to_dense(features["x1"])
    # -Recover the original tensor -
    # data_cols[0] is still "serialized bytes"
    # created with "tf.io.serialize_tensor()"
    # it does bytes -> Tensor
    data_cols = tf.io.parse_tensor(data_cols[0], out_type=tf.float64)
    #---------
    data_cols_not = tf.sparse.to_dense(features["x2"])
    data_cols_not = tf.io.parse_tensor(data_cols_not[0], out_type=tf.float64)
    #--------------
    data_cols_targets = tf.sparse.to_dense(features["x3"])
    data_cols_targets = tf.io.parse_tensor(data_cols_targets[0], out_type=tf.float64)
    #----------------
    data_cols_not_target = tf.sparse.to_dense(features["x4"])
    data_cols_not_target = tf.io.parse_tensor(data_cols_not_target[0], out_type=tf.float64)

    time = tf.sparse.to_dense(features["time"])
    gidx = tf.sparse.to_dense(features["gidx"])
    gidx = gidx[0]
    res = 0
    if gidx<0:
        gidx = -gidx-1
        res = 1
    idx = tf.sparse.to_dense(features["idx"])
    batch_index = tf.sparse.to_dense(features["batch_index"])

    out = {}
    out['X_col']  = data_cols
    out['X_col_not']  = data_cols_not
    out['y']  = tf.concat([data_cols_targets, data_cols_not_target], axis = 0)
    out['time']  = time
    out['gidx']  = gidx
    out['res'] = res
    return out
```

**Writing-Reading TFRecords**
In the function above, some variables were read with 
```python
sparse.to_dense() 
parse_tensor(...,out_type=)
```
while other variables were read with only
```python
parse.to_dense()
```
`parse.to_dense()` is used when the tensor was saved with a variable length of elements and then it wants to be converted to a dense vector.
`parse_tensor(...)` is an **additional** step that is required when the tensor was saved as a **serialized byte**. Let's see two examples to understand how a tensor is saved and read.
1. Method 1: store the tensor directly. Suppose you have `time = [1,2,3,4]` and you write it to tensor using
```
tf.train.Int64List(...)
```
in this case, the array is saved as tensor directly. It stores the numerical value. When reading, as the tensor already contains the numerical values you can read it directly with
```python
time = tf.sparse.to_dense(features["time"])
```
2. Method 2: serialize an entire tensor. Suppose now you have `x=[1,2,3,...,1e16]` and you write
```python
ser = tf.io.serialize_tensor(x)
```
In this case, the numerical value is **NOT** longer stored, but **one long byte string**, when reading it, TensorFlow see `b"\x08\..."`. To recover the original tensor you need to convert it back to a number.
```python
tf.io.parse_tensor(...out_type=...)
```
**When should I stored it as a byte string?**
In general, small data can be saved directly as numerical values, while long data is advisable to do it as a byte string.

##### `get_timespace`
Once we have read the TFRecords and obtain our `out` dictionary, the function `get_timespace(...)` performs **feature engineering** on the data creating new input variables from existing data. Let's take a look to the function
```python
def get_timespace(x, lat_hr, lon_hr, lat, lon, day_time_sin, day_time_cos, sec_time_sin, sec_time_cos):
	'''
	Inputs:
	x: dictionary produced by "decode_tfrec()"
	lookup tables: lat_hr, lon_hr, lat, lon,...
	---
	Outputs:
	Same directory as the input but: time removed and replaced 
	by latitude, longitude, cyclic day, cyclic second
	'''
    res = x['res'] # Read the resolution flag
    gidx = x['gidx'] # Read the grid index
    time = x['time'] # Read the time
    if res == 0: # res == 0 corresponds to the high resolution
        x_lat = lat_hr[gidx]
        x_lon = lon_hr[gidx]
    else:
        x_lat = lat[gidx]
        x_lon = lon[gidx]
	# ------------- Cyclic encoding ---------------
	# instead of January=1, December = 12
	# the code uses sin() and cos()
	# Because December -> January, should be close 
	# not far apart.
	# The same idea applies to minutes, hours, weeks
    day_sin = day_time_sin[(time[1]-1)*31+time[2]-1]
    day_cos = day_time_cos[(time[1]-1)*31+time[2]-1]
    sec_sin = sec_time_sin[time[3]]
    sec_cos = sec_time_cos[time[3]]
	# ---------------------------------------------
    x.pop('gidx') # removes the grind index
    x.pop('time') # removes time
	
	# ------------ Add new features ---------------
    x['lat']  = x_lat
    x['lon']  = x_lon
    x['day_sin']  = day_sin
    x['day_cos']  = day_cos
    x['sec_sin']  = sec_sin
    x['sec_cos']  = sec_cos
    return x
```

#####  `get_labels`, `get_stateq002`, `get_labels_2`
These functions access to a specific columns in the read dataset. For example `return x['concat_x`]

##### `get_normed`
One of the most important parts of the entire **preprocessing pipeline**. It does much more than ==normalizing==. It performs **feature engineering**, **target transformation**, **outlier compression**, **data type conversion**, and **feature construction**. 
The **goal** of this function is to transform the raw atmospheric variables into a representation that is easier for the neural network to learn.
Let's take a look to the function
```python
def get_normed(x, valid = False):
	'''
	This function does:
	1. Normalizes every predictor
	2. Normalizes the target
	3. Compresse extreme values
	4. Compute additional derived values
	5. Generates multiple representations of the same variable
	6. Removes the original raw variables
	   
	The general workflow is:
	Raw sample -> Normalize -> Feature engineering -> remove raw
	variables -> return transformed sample
	---
	Inputs:
	x: dataset
	---
	Outputs:
	x: transformed dataset
	'''
	# normalize the target: (x-mean)/standard_deviation
    y_norm =  (x['y'] - mean_y)*stds
    rescale_factor = 1.1
    if (x['res'] == 0) and not valid:
    # "tail-softening" apply it only to training data 
    # and not to valid(ation) data
	    # for values below the (rescaled) minimum threshold
	    # replace them with a logarithmically compressed 
	    # version
	    # this softens/limits the effect of extreme low outliers
        y_norm = tf.where(y_norm<norm_y_min*rescale_factor, norm_y_min*rescale_factor-tf.math.log(1-y_norm+norm_y_min*rescale_factor), y_norm)
        # the same idea as above but for outliers too big
        # instead of too low
        y_norm = tf.where(y_norm>norm_y_max*rescale_factor, norm_y_max*rescale_factor+tf.math.log(1+y_norm-norm_y_max*rescale_factor), y_norm)
       
	# Ensure float32 type. 
    y_norm = tf.cast(y_norm, tf.float32)
    # Standard z-score normalization
    x_col_not_norm = (x['X_col_not'] - mean_col_not)/std_col_not
	
	# takes 'X_col': flat vector encoding 9 variables X 60
	# vertical levels and reshape it to:
	# (batch/whatever, 9, 60)
    x_total_norm = tf.reshape(x['X_col'], [-1, 9, 60])
    # z-score normalization for all vertical levels
    x_total_norm = (x_total_norm - X_total_mean)/X_total_std
    # flattenes back to a 1D vector of length 9*60 = 540
    x_total_norm = tf.reshape(x_total_norm, [60*x_total_norm.shape[1]])
	# z-score but for each vertical level
    x_col_norm = (x['X_col'] - x_col_mean)/x_col_std
    
    # creates a log-transformed version of "x_col_norm"
    # Looks like a signed-log transform
    x_col_norm_log = tf.where((x_col_norm-x_col_norm_min+1)>=1, tf.math.log(x_col_norm-x_col_norm_min+1),
                                    -tf.math.log(1+1-(x_col_norm-x_col_norm_min+1)))
	
	# creates a new feature: wind
    wind = tf.reshape(x['X_col'], [-1, 9, 60])
    wind = wind[:, 4:6, :] # takes feature 4,5
    # computes wind speed magnitude (u**2 + v**2)^(1/2)
    # this is wind speed per level (60 levels)
    wind = (tf.math.reduce_sum(wind**2, axis = 1))**0.5
    wind = wind[0] # takes only the first element in the batch
    # normalizes the wind-speed 
    wind = ((wind - tf.math.reduce_mean((tf.reshape(x_col_mean, [9, 60])[4:6]), axis = 0))/
                    tf.math.reduce_sum((tf.reshape(x_col_std, [9, 60])[4:6]), axis = 0))

    cutoff = 30
    square_cutoff = cutoff**0.5
    # for values beyon +- 30, replace the linera value with 
    # a square-root-compressed value. Beyond cutoff, growth
    # is compressed to a squere-root rate instead of linear
    # it has a symmetric treatment for the negative tail.
    x_col_norm = tf.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
    x_col_norm = tf.where(x_col_norm<-cutoff, -tf.math.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)
	
	# Similar treatment described above for the wind feature
    wind = tf.where(wind>cutoff, wind**0.5+cutoff-square_cutoff, wind)
    wind = tf.where(wind<-cutoff, -tf.math.abs(wind)**0.5-cutoff+square_cutoff, wind)

	# casts all features to float32 precision
    x_col_not_norm = tf.cast(x_col_not_norm, tf.float32)
    x_total_norm = tf.cast(x_total_norm, tf.float32)
    x_col_norm = tf.cast(x_col_norm, tf.float32)
    x_col_norm_log = tf.cast(x_col_norm_log, tf.float32)
    wind = tf.cast(wind, tf.float32)

	# apply a second cutoff (softness) of the variables
    cutoff_2 = 86.0
    log_cutoff = tf.math.log(cutoff_2)
    x_col_norm = tf.where(x_col_norm>cutoff_2, tf.math.log(x_col_norm)+cutoff_2-log_cutoff, x_col_norm)
    x_col_norm = tf.where(x_col_norm<-cutoff_2, -tf.math.log(-x_col_norm)-cutoff_2+log_cutoff, x_col_norm)
	# "col_not_norm" didn't have the firs cutoff
    x_col_not_norm = tf.where(x_col_not_norm>cutoff_2, tf.math.log(x_col_not_norm)+cutoff_2-log_cutoff, x_col_not_norm)
    x_col_not_norm = tf.where(x_col_not_norm<-cutoff_2, -tf.math.log(-x_col_not_norm)-cutoff_2+log_cutoff, x_col_not_norm)
	# second cutoff, first cutoff was not applied.
    x_total_norm = tf.where(x_total_norm>cutoff_2, tf.math.log(x_total_norm)+cutoff_2-log_cutoff, x_total_norm)
    x_total_norm = tf.where(x_total_norm<-cutoff_2, -tf.math.log(-x_total_norm)-cutoff_2+log_cutoff, x_total_norm)
	#second cutoff, first cutoff was applied
    wind = tf.where(wind>cutoff_2, tf.math.log(wind)+cutoff_2-log_cutoff, wind)
    wind = tf.where(wind<-cutoff_2, -tf.math.log(-wind)-cutoff_2+log_cutoff, wind)
	# removes original feaures
    x.pop('X_col')
    x.pop('X_col_not')
    x.pop('y')
	
	# defines new features in the dataset
    x['normed_y'] = y_norm
    x['x_col_not_norm'] = x_col_not_norm
    x['x_total_norm'] = x_total_norm
    x['x_col_norm'] = x_col_norm
    x['x_col_norm_log'] = x_col_norm_log
    x['wind'] = wind
    x['epoch'] = 0
    return x
```

**Note:** `signed-log transform`: for values way greater than 1 take the log(value), and for values lower than 1 uses -log(2-value), which is the mirror/reflection making the function continuous and roughly antisymmetric around the shift point. This is a common trick to compress heavy-tailed distributions on both sides while keeping small values roughly linear.
![Example of log-transformed distribution](../figures/Pasted%20image%2020260718164501.png)

###### **Note: on Normalization and Clipping (cutoff)**
After `z-score` normalization, a well-behaved Gaussian feature should mostly go from `[-3,3]`. But real word data often has **heavy-tailed distributions** (occasional extreme values) that land at z-scores of 10, 50, 1000.
***If you those values into a neural network*** 
- Large-magnitude inputs dominate gradients and loss -> unstable training.
- The network uses its capacity trying to fit this extreme values instead of learning the bulk of the distribution
- Weight initializations and activation functions (tanh, sigmoid) are tuned assuming roughly unit-scale inputs.

**- The standard fix** is some form of **soft clipping**: instead of a hard cutoff (neglect any extreme value) you let values below/above a threshold continue growing, but at a **compressed rate**, e.g. sqrt, log. This preserves ordering and information allowing larger/smaller raw values still map to larger/smaller normalized values while preventing extreme values from having outsized.

***- How to choose the cutoff function?***
The election of the cutoff function depends on how "strong" you want to clip your data, for example a square root has a smaller effect on the data than a logarithmic function. One important to consider is that the piecewise function is **continuous**, for example `x_col_norm**0.5 + cutoff - square_cutoff` at `x=cutoff` we have that `cutoff**0.5 + cutoff - square_cutoff = cutoff` and there is no **jump** in the data.

***-How to choose the cutoff threshold?***
There is no universal formula for this and it depends on your data. One suggested approach is:
	1. Normalize (z-score) using train-set statistics
	2. Compute `min`,`max`, and `percentiles (99,99.9)` of each normalized feature
	3. Pick the cutoff around the point beyond which data density is very sparse. For example, at percentile 99.
	4. Verify by re-plotting the transformed distribution along the original distribution: new distribution more concentrated ([-10,10] and no jumps)
	5. Treat cutoff as **hyperparameters**, validate the model training stability and final performance are not overly sensitive to their exact values.

##### `get_concats`
This function groups (concatenates) several independent features into a single feature vector that can be fed into the neural network. If you have three vector $x_1$, $x_2$, $x_3$, their concatenation is $x = [x_1,x_2,x_3]$, it is simply placed one vector after another.

Let's consider the function. 
```python
def get_concats(x):
	#x['concat_x'] will be now one large vector containing
	#sequentially the concatenated vectors.
    x['concat_x'] = tf.concat([x['x_total_norm'],                            x['x_col_norm'], x['x_col_norm_log'],x['wind']],                axis = 0)
    #for now this is redundant
    x['concat_x_col_not'] = tf.concat([x['x_col_not_norm']], axis = 0)
    #this concatenates normalied target plus longitude,
    #latitude, time.
    #According to this the model won't only predict y
    #but also: latitude,longitude (where), time(when)
    #"where and when the observation occurred".
    #The use of [None]: if x['lat'] is 42.5 ( a scalar )
    #its shape is (), but tf.concat() CANNOT concatenate
    #scalars with vectors so x['lat'][None] creates [42.5]
    #and its shape is now (1,)
    x['concat_target'] = tf.concat([x['normed_y'], x['lat'][None], x['lon'][None],x['day_sin'][None], x['day_cos'][None], x['sec_sin'][None], x['sec_cos'][None]], axis = 0)
	#removes original variables from the dataset
	#Now the input dataset to fed the NN is ony single tensor
	#instead of a dataset with multiple columns (features)
	# X
	# -concat_x
	# -concat_x_col_not
	# -concat_target
	# -epoch 
	# Most dense networks and Transformers expect exactly this.
    x.pop('x_total_norm')
    x.pop('x_col_norm')
    x.pop('x_col_norm_log')
    x.pop('x_col_not_norm')
    x.pop('lat')
    x.pop('lon')
    x.pop('day_sin')
    x.pop('day_cos')
    x.pop('sec_sin')
    x.pop('sec_cos')
    x.pop('wind')
    return x
```

##### `get_final_output`.
This function makes another concatenation. It takes the output dataset from `get_concats(x)` and concatenates all features previously divided into `concat_x` and `concat_x_col_not` and defines the variable `concat_target`
```python
def get_final_output(x):
    x_concat = tf.concat([x['concat_x'], x['concat_x_col_not']], axis = 0)
    concat_target = x['concat_target']
    return x_concat, concat_target
```

### Obtaining validation data (val_1_ds)
validation data is obtained by calling the function `get_dataset(...)`
```python
val_1_ds, val_1_labels_ds, val_1_labels_ds_2, val_1_ds_q0002 = get_dataset(
    val_1_tffiles, val_batch_size, shuffle = False, cache = True)
```
`gest_dataset(...)` and all calls to other functions inside it have already been explained in the above section (getting train_ds). Therefore, the explanation of the validation dataset will refer to that section. It is worth noting that the difference from the training dataset lies in the use of the `TFRecords` used.

## Getting additional information
In the previous section the model structure and the training and validation data were studied. During this process, several variables were used, such as the mean and standard deviation of the data, latitude, longitude and time. Additionally, the datasets were obtained from `TFRecords`. In this section we will explore the original structure of these variables and how the `TFRecords`  and statistical information are obtained.

