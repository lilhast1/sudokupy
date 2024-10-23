#!/usr/bin/env python3
import tensorflow as tf

keras_model = tf.keras.models.load_model('final_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('android_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print('success')