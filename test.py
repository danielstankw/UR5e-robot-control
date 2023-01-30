import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV, train_test_split

model = keras.models.load_model('/home/danieln7/Desktop/ML/robot_1_full')
print(model.summary())
# model.compile()

# def make_model(n_inputs, metrics=None, output_bias=None):
#     if output_bias is not None:
#         output_bias = tf.keras.initializers.Constant(output_bias)
#
#     nn = keras.Sequential()
#     nn.add(keras.layers.Dense(16, input_shape=(n_inputs,), activation='relu'))
#     nn.add(keras.layers.Dense(32, activation='relu'))
#     nn.add(keras.layers.Dense(64, activation='relu'))
#     nn.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))
#     # testing to reduce overfit
#     #     nn.add(keras.layers.Dropout(0.1))
#
#     nn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#                loss='binary_crossentropy',
#                metrics=metrics)
#     # metrics simply tell us what we will be able to see in the log and on plot
#     # they do are NOT used for optimization!
#
#     return nn
#
#
# df = pd.read_csv('./robotdatacollection/ep1.csv')
#
# feature_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My']
# # feature_names = ['fz']
# #
# # feature_names = ['fx', 'fy', 'fz', 'mx', 'my']
#
# X_df = df[feature_names]
# X = X_df.to_numpy()
# y = df.Case.to_numpy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# n_inputs = X_train.shape[1]  # (77862, 3)
# print('Input shape',n_inputs)
# model = make_model(n_inputs, metrics='AUC')
# print(model.summary())
#
# model.save('testing_model')