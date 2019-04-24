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



