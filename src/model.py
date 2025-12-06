# creates the model using the tensorflow sublassing api

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, TimeDistributed, Layer, GRU, Dropout, LayerNormalization
from tensorflow.keras.regularizers import l2

# attention layer
@register_keras_serializable(package="custom_layer")
class Attention(Layer):

    """custom attention layer to help model pay attention to specific frames"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation='tanh')
    
    def call(self, inputs):
        # inputs: (batch_size, timesteps, features)
        score = self.W(inputs)  # (batch_size, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)  # attention weights
        context = tf.reduce_sum(weights * inputs, axis=1)  # weighted sum
        return context

@register_keras_serializable(package="custom_model")
class GeometryDashAgent(Model):

    """a CNN/LSTM hybrid model to play geometry dash"""

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # model architecture

        # conv layers -> TimeDistributed to add timestep dimension
        self.conv1 = Conv2D(
            filters=32, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding="same", 
            activation="relu",
        )
            
        self.conv2 = Conv2D(
            filters=32,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same",
            activation="relu",
        )

        # layer normalization
        self.ln = LayerNormalization()

        # flatten -> timedistributed globalaveragepooling2d to preserve timesteps for lstm
        self.td_gap2d = TimeDistributed(GlobalAveragePooling2D()) # NOTE: switch to Flatten() with td if bad performance

        # attention layer
        self.attention = Attention()

        # gru layers (maybe lstm later???)
        # change to lstm if performance is bad
        self.gru1 = GRU(
            units=128,              
            return_sequences=True,   
            activation='tanh',      
            recurrent_activation='sigmoid',
            recurrent_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5),
        )

        self.gru2 = GRU(
            units=128,
            return_sequences=False, # only true if stacking more gru/lstm
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5)
        )

        # dropout
        self.dropout = Dropout(0.2)

        # dense layers
        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))
        self.dense2 = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))

        self.out = Dense(4, activation="softmax") # 3 classes -> wait, jump, hold, release

    def call(self, x):

        x = TimeDistributed(self.conv1)(x)   
        x = TimeDistributed(self.conv2)(x)
        x = TimeDistributed(self.ln)(x)     

        x = self.td_gap2d(x)      

        x = self.gru1(x)               
        x = self.gru2(x)               

        x = tf.expand_dims(x, axis=1)  
        x = self.attention(x)          

        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.out(x)     

        return x
