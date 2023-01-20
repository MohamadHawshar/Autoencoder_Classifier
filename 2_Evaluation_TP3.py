from numpy.core.fromnumeric import argmax
# ===========================================================================
# TP3 : INF7370 - Hiver 2022
#
# Mohamad Hawchar : HAWM20039905
# Nassir Ade-Dayo Adekoudjo: ADEA04089904

#===========================================================================

#===========================================================================
# Dans ce script, on évalue l'autoencodeur entrainé dans 1_Modele.py sur les données tests.
# On charge le modèle en mémoire puis on charge les images tests en mémoire
# 1) On évalue la qualité des images reconstruites par l'autoencodeur
# 2) On évalue avec une tache de classification la qualité de l'embedding
# 3) On visualise l'embedding en 2 dimensions avec un scatter plot


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Utilisé pour normaliser l'embedding
from sklearn.preprocessing import StandardScaler

from keras import backend as K
# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, LeakyReLU, Dense
from sklearn.preprocessing import StandardScaler


import time
# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);


# ==========================================
# ==================MODÈLE==================
# ==========================================

# Chargement du modéle (autoencodeur) sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
autoencoder = load_model(model_path)

# Configuration des  images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
image_scale = 120
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images
mainDataPath = "vache_elephant/"

# On évalue le modèle sur les images tests
datapath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 400 # 400 images
number_images_class_0 = 200 # 200 images pour la classe du chiffre 2
number_images_class_1 = 200 # 200 images pour la classe du chiffre 7

# Les étiquettes (classes) des images
labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# La taille des images
image_scale = 120

# La couleur des images
images_color_mode = "rgb"  # grayscale ou rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images test
data_generator = ImageDataGenerator(rescale=1. / 255)

generator = data_generator.flow_from_directory(
    datapath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size= number_images, # nombre d'images total à charger en mémoire
    class_mode='binary',
    shuffle=False) # pas besoin de bouleverser les images

(x , y) = generator.next()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 2) Reconstruire les images tests en utilisant l'autoencodeur entrainé dans la première étape.
# Pour chacune des classes: Afficher une image originale ainsi que sa reconstruction.
# Afficher le titre de chaque classe au-dessus de l'image
# Note: Les images sont normalisées (entre 0 et 1), alors il faut les multiplier
# par 255 pour récupérer les couleurs des pixels
#
# ***********************************************
imgs = autoencoder.predict(x)
print(imgs.shape)

n = 2  # How many digits we will display
plt.figure(figsize=(2, 3), dpi = 200)
j = 0
for i in (3,340):
    # Display original
    ax = plt.subplot(2, n, j + 1)
    plt.imshow(x[i])
    ax.set_title( "cow" if y[i]//1 == 1 else "elephant")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, j + 1 + n)
    plt.imshow(imgs[i])
    ax.set_title(("cow" if y[i]//1 == 1 else "elephant") + " \n recon")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    j = j+1
plt.show()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 3) Définire un modèle "encoder" qui est formé de la partie encodeur de l'autoencodeur
# Appliquer ce modèle sur les images afin de récupérer l'embedding
# Note: Il est "nécessaire" d'appliquer la fonction (flatten) sur l'embedding
# afin de réduire la représentation de chaque image en un seul vecteur
#
# ***********************************************

# Défintion du modèle
input_layer_index = 0 # l'indice de la première couche de l'encodeur (input)
output_layer_index = 12 # l'indice de la dernière couche (la sortie) de l'encodeur (dépend de votre architecture)
# note: Pour identifier l'indice de la dernière couche de la partie encodeur, vous pouvez utiliser la fonction "model.summary()"
# chaque ligne dans le tableau affiché par "model.summary" est compté comme une couche
encoder = Model(autoencoder.layers[input_layer_index].input, autoencoder.layers[output_layer_index].output)
encoder.summary()
imgs_encoder = encoder.predict(x)
embedding = Flatten(input_shape=image_shape)(imgs_encoder)

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 4) Normaliser le flattened embedding (les vecteurs recupérés dans question 3)
# en utilisant le StandardScaler
# ***********************************************
scaler = StandardScaler()
scaler.fit(embedding)
normalized_embedding = scaler.transform(embedding)



# # ***********************************************
# #                  QUESTIONS
# # ***********************************************
# #
# # 5) Appliquer un SVM Linéaire sur les images originales (avant l'encodage par le modèle)
# # Entrainer le modèle avec le cross-validation
# # Afficher la métrique suivante :
# #    - Accuracy
# # ***********************************************


from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import confusion_matrix

clf_original = svm.SVC(kernel='linear',
              C=1, 
              #probability=True
              )

x_flatten = Flatten(input_shape=image_shape)(x)

#normalization
scaler.fit(x_flatten)
x_flatten_normalized = scaler.fit_transform(x_flatten)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# cross validation helps in detecting overfitting.
print("question 5 validation") 
print(clf_original)
accuracy  = cross_val_score(clf_original, x_flatten_normalized,y, cv = cv, scoring='accuracy')
print(accuracy)
print(np.mean(accuracy))


# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 6) Appliquer un SVC Linéaire sur le flattened embedding normalisé
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
# ***********************************************

from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
clf_embedding = svm.SVC(kernel='linear',
              C=1, 
              )
print("question 6 validation:")
print(clf_embedding)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
accuracy_embedding  = cross_val_score(clf_embedding, normalized_embedding,y, cv =cv, scoring='accuracy')
print(accuracy_embedding)
print(np.mean(accuracy_embedding))



# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 7) Appliquer TSNE sur le flattened embedding afin de réduire sa dimensionnalité en 2 dimensions
# Puis afficher les 2D features dans un scatter plot en utilisant 2 couleurs(une couleur par classe)
# ***********************************************
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

enb_reduce = TSNE(n_components=2).fit_transform(embedding)

plt.figure(dpi=200)
for i in range(400):

  if y[i]==1:
    plt.scatter(enb_reduce[i,0], enb_reduce[i,1], color = 'black' ,alpha=.4, s=3**2)
  else:
    plt.scatter(enb_reduce[i,0], enb_reduce[i,1], color = 'blue', alpha=.4, s=3**2)

plt.show()


