#!/usr/bin/env python3

from tensorflow import keras as K

def input_layers(max_moves):
    color_input_layer = K.layers.Input(shape=(1,), dtype='float32')
    board_input_layer = K.layers.Input(shape=(8, 8))
    moves_input_layer = K.layers.Input(shape=(1, max_moves))

    board_input_layer_with_channel = K.layers.Reshape(target_shape=(8, 8, 1))(board_input_layer)
    board_conv_layer_0 = K.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(board_input_layer_with_channel)
    board_conv_layer_1 = K.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(board_conv_layer_0)
    board_pool_layer = K.layers.MaxPooling2D(pool_size=(2, 2))(board_conv_layer_1)
    board_pool_flat = K.layers.Flatten()(board_pool_layer)

    moves_lstm_layer = K.layers.LSTM(units=128, return_sequences=True)(moves_input_layer)
    moves_lstm_flat = K.layers.Flatten()(moves_lstm_layer)

    merged_inputs = K.layers.concatenate([color_input_layer, board_pool_flat, moves_lstm_flat])

    return (
        merged_inputs,
        color_input_layer,
        board_input_layer,
        moves_input_layer
    )

def hidden_layers(max_moves, inputs):
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(inputs)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    output_layer = K.layers.Dense(max_moves, activation='softmax')(hidden_layer)
    return output_layer

def compile_model(output_layer, color_input_layer, board_input_layer, moves_input_layer,
                  metrics=['accuracy']):
    m = K.models.Model(
        inputs=[color_input_layer, board_input_layer, moves_input_layer],
        outputs=output_layer
    )
    m.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(), metrics=metrics)
    return m
