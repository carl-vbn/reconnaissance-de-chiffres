# Importation de la bibliothèque TensorFlow permettant d'automatiser l'apprentissage et de NumPy facilitant le traitement de données
import tensorflow as tf
import numpy as np

# Téléchargement d'exemples de chiffres manuscrits connus déja numérisés (Banque de donnée MNIST)
(x_train, y_train), test_data = tf.keras.datasets.mnist.load_data()

# Définition du réseau de neurones
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Définition de la fonction d'erreur (SparseCategoricalCrossentropy) et de l'algorithme d'apprentissage (Adam)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Début de l'apprentissage 
model.fit(x=x_train,y=y_train,batch_size=128,epochs=6,validation_data=test_data)

# Enregistrement des paramètres dans un fichier
model.save('model.h5')