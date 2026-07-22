
# Import libraries
```
import tensorflow as tf
import numpy as np
```
# Load and prepare data
```
X = np.load...
y = np.load ..
```
Suppose we have features `X` and targets `y` . You get the data.
If not done yet, you split the data into *training* and *validation* (70/30) datasets.
## Optionally: create TensorFlow datasets
```
train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_ds = train_ds.shuffle(#rows).batch(32)

val_ds = tf.data.Dataset.from_tensor_slices((X_val,y_val))
val_ds = val_ds.batch(32)
```
The above snippet transforms the dataset into a TensorFlow dataset, later it shuffles the samples (#rows) and creates batches of data, in this case of 32 elements. It does that for both, the training and validation dataset.

# Define the model
Here is where you create your model, the possibilities are only limited but your imagination.
For example, a simple feed-forward neural network
```
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

# Choose the loss function
The loss function can be a designed loss function or your can use one available in keras. For example, 
```
loss_fn = tf.keras.losses.MeanSquaredError()
```

# Choose an optimizer
The optimizer updates the weights using gradients. For example:
```
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
```

# Select metrics
Metrics measure how well your model performs with respect to the *validation* dataset. Metrics are not trained but reported during training  to see model evolution. Example of metrics as *'mae'*, *'accuracy'* 

# Compile the model
This step connects all previous steps. It connects the model, loss, optimizer and metrics.
```
model.compile(
	optimizer = optimizer,
	loss= loss_fn,
	metrics = ['mae'])
```

# Train the model
Now, you have created a model, you can train it:
```
history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=100)
```
This step conceptually does:
```
for epoch:
	for batch: #If I have 200 batches of 32 elements, for example
		y_pred = model(X_batch) #Performs a prediction using my model architecture
		loss = loss_fn(y_batch,y_pred) #Computes the loss using my defined loss fn
		>compute gradients (gradient descent)<
		>update weights according to the gradient<
```

# Evaluate model
Now, I use the data saved for validation
```
model.evaluate(X_val, y_val)

>>output:<<
>>loss: 0.042
>>mane: 0.15
```

# Make predictions
Now, I have a trained model that I can use to make predictions:
```
predictions = model.predict(X_new)
```

# Save the trained model
```
model.save("my_model.keras")
```

# Workflow summary:
```
Load data
    ↓
Preprocess
    ↓
Build model
    ↓
Compile(loss + optimizer + metrics)
    ↓
model.fit(...)
    ↓
Monitor train/validation loss
    ↓
Tune hyperparameters
    ↓
Evaluate
    ↓
Predict
    ↓
Save model
```
## Hyperparameters you usually tune
```
|Hyperparameter|Typical values|
|---|---|
|Learning rate|1e-2, 1e-3, 1e-4|
|Batch size|16, 32, 64, 128|
|Epochs|10–1000|
|Hidden layers|1–10|
|Neurons/layer|16–1024|
|Optimizer|Adam, SGD, RMSprop|
|Dropout|0.1–0.5|
|Weight decay|1e-6–1e-2|
```
