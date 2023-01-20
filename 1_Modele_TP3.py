
# ===========================================================================
# TP3 : INF7370 - Hiver 2022
#
# Mohamad Hawchar : HAWM20039905
# Nassir Ade-Dayo Adekoudjo: ADEA04089904

#===========================================================================

# #===========================================================================
# Ce modèle est un Autoencodeur Convolutif entrainé sur l'ensemble de données MNIST afin d'encoder et reconstruire les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 1 000 images | classe '7': images 1 000 images
# validation   : classe '2':   200 images | classe '7': images   200 images
# test         : classe '2':   200 images | classe '7': images   200 images
# ------------------------------------------------

# >>> Ce code fonctionne sur MNIST.
# >>> Vous devez donc intervenir sur ce code afin de l'adapter aux données du TP3.
# >>> À cette fin repérer les section QUESTION et insérer votre code et modification à ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Le Model à compiler
from keras.models import Model

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats

from tensorflow.keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, LeakyReLU, Dense
from sklearn.preprocessing import StandardScaler
# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
# Configuration du GPU
import tensorflow as tf

# Affichage des graphes
import matplotlib.pyplot as plt

from keras import backend as K
import time

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_ds_size
# - validation_ds_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "vache_elephant/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le nom du fichier du modèle à sauvegarder
model_path = "Model.hdf5"

# Le nombre d'images d'entrainement
training_ds_size = 2160  # total 2000 (1000 classe: 2 et 1000 classe: 7)
validation_ds_size = 240  # total 400 (200 classe: 2 et 200 classe: 7)


# Configuration des  images
image_scale = 120  # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32  # le nombre d'images entrainées ensemble: un batch
fit_epochs = 100  # Le nombre d'époques

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) encoder
# 3) decoder
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie d'encodage (qui extrait les features des images et les encode)
def encoder(input):
    # 1- couche de convolution avec nombre de filtre  (exp 32)  avec la taille de la fenêtre de ballaiage exp : 3x3
    # 2- fonction d'activation exp: sigmoid, relu, tanh ...
    # 3- couche d'echantillonage (pooling) pour reduire la taille avec la taille de la fenêtre de ballaiage exp :2x2

    # **** On répète ces étapes tant que nécessaire ****
    
    #Pour notre modèle, nous utilisons une architecture semi-métrique d'encodeur-décodeur.
    # Nous avons expérimenté de nombreuses versions différentes de cette architecture et nous sommes arrivés à la conclusion que cette architecture est la meilleure
    #    pour capturer les caractéristiques des deux classes.
    # Explication des choix de l'architecture :
    # -Trois blocs de "convolution-sampling" : Nous avons essayé d'utiliser deux blocs au lieu de trois au début,
    #    le résultat obtenu n'était pas assez satisfaisant, nous avons donc augmenté à trois blocs
    #    dans le but de réduire beaucoup plus l'information spatiale (de 120x120 à 15x15) et de faire en sorte que l'algorithme se base et se concentre sur les caractéristiques 
    #    calculées pour représenter les classes.
    # -Poids dans l'ordre ([128,64,32] pour l'encodeur et [32,64,128] pour le décodeur : Nous avons expérimenté avec beaucoup de valeurs différentes,
    #    et nous avons conclu que la dimensionnalité totale et les paramètres à la fin de la phase d'encodage sont très importants pour la généralisation,
    #    c'est pourquoi le nombre de filtres diminue avec la dimensionnalité de l'image (pour la partie encodeur). Notre objectif est que le modèle ne garde que les quelques
    #    caractéristiques qui représentent les classes dans la couche d'intégration, ce choix rend l'image reconstruite plus floue car les paramètres de la couche d'intégration
    #    sont peu nombreux. Cependant, ce choix a permis d'augmenter le score de l'évaluation de l'encodeur.
    # -Dropout : Nous avons utilisé le dropout pour empêcher le modèle de mémoriser les données d'entraînement afin de lutter contre l'overfitting.
    # -Pas de normalisation par lot : l'utilisation de la normalisation par lot peut aider à accélérer le taux d'apprentissage, mais dans notre cas,
    #    nous n'avons pas vu la nécessité de l'utiliser car les pixels sont déjà normalisés entre les valeurs 0 et 255.
    # -Le leaky rely : les avantages du leaky relu sont qu'il augmente le temps d'apprentissage et qu'il résout le problème du dying relu, 
    #    il est également populaire pour les tâches d'autoencodage et de génération.
    #1
    x = Conv2D(32, (3, 3), padding='same')(input) 
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    # #2
    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    # #3
    x = Conv2D(128, (3, 3), padding='same')(x) 
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)


    # encoded : La sortie de l'encodeur consistue l'embedding (ou les descripteurs extraites par l'encodeur)
    return encoded


# Partie de décodage (qui reconstruit les images à partir de leur embedding ou la sortie de l'encodeur)
def decoder(encoded):
    # 1- couche de convolution avec nombre de filtre  (exp 32) (avec la taille de la fenêtre de ballaiage exp : 3x3)
    # 2- fonction d'activation exp: sigmoid, relu, tanh ...
    # 3- couche de suréchantillonnage pour augmenter la taille (avec la taille de la fenêtre de ballaiage exp :2x2)

    # l'enocdeur diminue la taille de l'image  et augmente le nombre de filtres de convolution progressivement
    # le decodeur augmente la taille de l'image et diminue le nombre de filtres de convolution progressivement

    # **** On répète ces étapes tant que nécessaire ****
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    

    # la dernier couche doit avoir les mêmes dimensions de l'image d'entré (input)
    # (C'est très important, car nous somme entrain de reconstruire l'image d'entrée)
    x = Conv2D(image_channels, (3, 3), padding='same')(x)

    # la dernière couche doit passer par un sigmoide car les pixels des images sont
    # normalisées entre 0 et 1 et l'autoencodeur essaie de prédire chaque pixel par une valeur entre 0 et 1
    decoded = Activation('sigmoid')(x)
    return decoded


# Déclaration du modèle:
# La sortie de l'encodeur sert comme entrée à la partie decodeur
model = Model(input_layer, decoder(encoder(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétres ...)
model.summary()

# Compilation du modèle :
# loss: On définit la fonction de perte (généralement on utilise le MSE pour les autoencodeurs standards)
# optimizer: L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# metrics: La valeur à afficher durant l'entrainement, metrics=['mse']
# On suit le loss (ou la difference) de l'autoencodeur entre les images d'entrée et les images de sortie
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# Les images sont normalisées (leurs pixels divisées par 255)
training_data_generator = ImageDataGenerator(rescale=1. / 255,
                                             validation_split=0.1)

# validation_data_generator: charge les données de validation en mémoire
# Les images sont normalisées (leurs pixels divisées par 255)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode =images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size = training_ds_size, # nombre d'images total à entrainer
    subset='training',
    class_mode ="input") # Comme nous somme entrain de reconstruire les images, alors
                        # la classe de chacune des pixels de sorite est le pixel d'entrée elle même(Input pixel)

# validation_generatory: indique la méthode de chargement des données de validation
validation_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode =images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size = validation_ds_size, # nombre d'images total à valider
    subset='validation',
    class_mode ="input") # Comme nous somme entrain de reconstruire les images, alors
    
                        # la classe de chacune des pixels de sorite est le pixel d'entrée elle même(Input pixel)

# On charge les données d'entrainement en mémoire
(x_train, y_train) = training_generator.next()
# On charge les données de validation en mémoire
(x_val, y_val) = validation_generator.next()


# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec le minimum loss sur les données de validation (monitor='val_loss')
# Note: on sauvegarde le modèle seulement quand le validation loss (la perte) diminue
# le loss ici est la difference entre les images originales (input) et les images reconstruites (output)
modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# entrainement du modèle
# On remarque que pour la fonction "fit", la valeur de "x" (données d'entrainement) et celles de "y" (étiquettes) sont les mêmes
# C'est parce qu'on est entrain de reconstruire les pixels de l'image d'entrée
start = time.time()
autoencoder = model.fit(x_train, x_train,
                       epochs=fit_epochs, # nombre d'epochs
                       batch_size=fit_batch_size, # nombre d'images entrainées ensemble
                       verbose=1, # mets cette valeur à 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint], # les fonctions à appeler à la fin de chaque epoch (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
                       shuffle=False, # On ne boulverse pas les données
                       validation_data=(x_val, x_val)) # données de validation
execution_time = (time.time() - start)/60
# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
# ***********************************************
print("Total time: ", execution_time, " minutes")



# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe de  perte par époque (loss over epochs)
#
# ***********************************************
plt.plot(autoencoder.history['loss'])
plt.plot(autoencoder.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()