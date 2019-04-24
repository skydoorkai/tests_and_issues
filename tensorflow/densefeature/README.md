# Issue for tf.keras.layers.DenseFeatures

`tf.keras.layers.DenseFeatures` is a new keras layer added in tf2.0a0.  This is the first layer for tf.keras.Model that accepts dict of tensors as the input.

If `tf.keras.layers.DenseFeatures` is used as the first layer in a keras model created by subclass `tf.keras.Model`, `model.build(input_shape)` will:

1. fail [here](https://github.com/tensorflow/tensorflow/blob/3c676a1cf6cc402f6ed7ccc9f52f15d62e4bb3ea/tensorflow/python/keras/engine/network.py#L807) if `dict of TensorShapes` is used as `input_shape`, as `input_shape` requires to be a TensorShape or a list/tuple of TensorShapes;

2. or fails [here](https://github.com/tensorflow/tensorflow/blob/3c676a1cf6cc402f6ed7ccc9f52f15d62e4bb3ea/tensorflow/python/keras/engine/network.py#L855) if `list of TensorShapes` is used as `input_shape`. This is because a list of corresponding placeholders are created, and `DenseFeaturess.call` is called with this list, but `DenseFeaturess.call` can only accept dict as input.



### Sample test code
[example.py](example.py)

```
import tensorflow as tf

feature_columns = [tf.feature_column.numeric_column(header) for header in ['c1', 'c2']]

class TestModel(tf.keras.Model):
    def __init__(self, feature_columns):
        super(TestModel, self).__init__()
        self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.dense_layer = tf.keras.layers.Dense(8)

    def call(self, inputs):
        x = self.feature_layer(inputs)
        return self.dense_layer(x)

model = TestModel(feature_columns=feature_columns)
shape = {'c1': (1,1), 'c2': (1,1)}
model.build(shape)
print(model.trainable_variables)
```

## Repro the failure 2

```
docker build -t test_densefeature_bad -f Dockerfile.bad .
docker run -it --rm test_densefeature_bad
```

it will fail with error message

```
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/keras/engine/network.py", line 792, in build
    'input type: {}'.format(type(input_shape)))
ValueError: Specified input shape is not one of the valid types. Please specify a batch input shape of type tuple or list of input shapes. User provided input type: <class 'dict'>
```

## Possible fix 
Adding `dict` support in `Network.build(input_shape)` would fix the issue.
See  the provided file [network.py](network.py).

And the repro step for the fix

```
docker build -t test_densefeature_pass -f Dockerfile.pass .
docker run -it --rm test_densefeature_pass
```
