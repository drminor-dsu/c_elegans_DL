import tensorflow as tf
from tensorflow import keras


class TransformerEncoder(keras.layers.Layer):
	def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
		super().__init__(**kwargs)
		self.embed_dim = embed_dim
		self.dense_dim = dense_dim
		self.num_heads = num_heads

		self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads,
														 key_dim=embed_dim)
		self.dense_proj = keras.Sequential(
			[keras.layers.Dense(dense_dim, activation='relu'),
			 keras.layers.Dense(embed_dim)]
		)
		self.layernorm_1 = keras.layers.LayerNormalization()
		self.layernorm_2 = keras.layers.LayerNormalization()

	def call(self, inputs, mask=None):
		if mask is not None:
			mask = mask[:, tf.newaxis, :]
		attention_output = self.attention(inputs, inputs, attention_mask=mask)
		proj_input = self.layernorm_1(inputs + attention_output)
		proj_output = self.dense_proj(proj_input)

		return self.layernorm_2(proj_input + proj_output)

	def get_config(self):
		config = super().get_config()
		config.update({
			'embed_dim': self.embed_dim,
			'num_heads': self.num_heads,
			'dense_dim': self.dense_dim
		})

		return config


if __name__ == '__main__':
	vocab_size = 20000
	embed_dim = 256
	num_heads = 2
	dense_dim = 32

	inputs = keras.Input((None,), dtype='int64')
	x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
	x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
	x = keras.layers.GlobalMaxPooling1D()(x)
	x = keras.layers.Dropout(0.5)(x)
	outputs = keras.layers.Dense(1, activation='sigmoid')(x)

	model = keras.Model(inputs, outputs)
	model.compile(optimizer='rmsprop',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])
	model.summary()
