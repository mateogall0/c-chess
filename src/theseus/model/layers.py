#!/usr/bin/env python3

from tensorflow import keras as K

def input_layers(max_moves):
    color_input_layer = K.layers.Input(shape=(1,), dtype='int32')
    board_input_layer = K.layers.Input(shape=(8, 8))
    moves_input_layer = K.layers.Input(shape=(1, max_moves))

    board_flattened = K.layers.Flatten()(board_input_layer)
    moves_flattened = K.layers.Flatten()(moves_input_layer)

    hidden_layer_board = K.layers.Dense(64, activation='relu')(board_flattened)
    hidden_layer_moves = K.layers.Dense(max_moves, activation='relu')(moves_flattened)
    hidden_layer_color = K.layers.Dense(32, activation='relu')(
        K.layers.Flatten()(K.layers.Embedding(2, 32)(color_input_layer))
    )

    merged_inputs = K.layers.concatenate(
        [hidden_layer_color, hidden_layer_board, hidden_layer_moves]
    )
    return (
        merged_inputs,
        color_input_layer,
        board_input_layer,
        moves_input_layer
    )

def hidden_layers(max_moves, inputs):
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(inputs)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
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
