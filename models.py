import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


class CustomSAEMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_sae_metric', **kwargs):
        super(CustomSAEMetric, self).__init__(name=name, **kwargs)
        self.total_score = self.add_weight(name='total_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred_class = tf.cast(y_pred_class, tf.float32)
        true_labels = y_true - 1
        pred_labels = y_pred_class - 1
        
        correct_nonzero = tf.where(
            tf.logical_and(
                tf.equal(true_labels, pred_labels),
                tf.not_equal(true_labels, 0)
            ),
            tf.ones_like(true_labels),
            tf.zeros_like(true_labels)
        )
        
        wrong_sign = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.not_equal(true_labels, 0),
                    tf.not_equal(pred_labels, 0)
                ),
                tf.equal(true_labels, -pred_labels)
            ),
            -tf.ones_like(true_labels),
            tf.zeros_like(true_labels)
        )
        
        pred_nonzero_actual_zero = tf.where(
            tf.logical_and(
                tf.equal(true_labels, 0),
                tf.not_equal(pred_labels, 0)
            ),
            -0.05 * tf.ones_like(true_labels),
            tf.zeros_like(true_labels)
        )
        
        batch_score = correct_nonzero + wrong_sign + pred_nonzero_actual_zero
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            batch_score = tf.multiply(batch_score, sample_weight)
        
        self.total_score.assign_add(tf.reduce_sum(batch_score))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return tf.divide(self.total_score, self.count)
    
    def reset_state(self):
        self.total_score.assign(0.0)
        self.count.assign(0.0)


def create_ae_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-3, batch_size=32
):
    inp = tf.keras.layers.Input(shape=(num_columns,), batch_size=batch_size)
    x0 = tf.keras.layers.BatchNormalization()(inp)

    
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation("swish")(encoder)
    
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name="decoder")(decoder)
    
    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation("swish")(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)
    out_ae = tf.keras.layers.Dense(num_labels, activation="softmax", name="ae_action")(
        x_ae
    )
    
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    
    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
    
    out = tf.keras.layers.Dense(num_labels, activation="softmax", name="action")(x)
    
    model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "decoder": tf.keras.losses.MeanSquaredError(),
            "ae_action": tf.keras.losses.SparseCategoricalCrossentropy(),
            "action": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "decoder": tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            "ae_action": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                CustomSAEMetric(name="custom_metric")
            ],
            "action": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                CustomSAEMetric(name="custom_metric")
            ],
        },
    )

    return model


def create_base_regression_model(num_columns, hidden_units, learning_rate=1e-3, batch_size=32):
    inputs = tf.keras.layers.Input(shape=(num_columns,), batch_size=batch_size)
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    for units in hidden_units:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
    
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_binary_classification_model(num_columns, hidden_units, learning_rate=1e-3, batch_size=32):
    inputs = tf.keras.layers.Input(shape=(num_columns,), batch_size=batch_size)
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    for units in hidden_units:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model
