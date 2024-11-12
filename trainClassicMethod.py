import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import os
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# pathFeatures_Full = "/home/pc/Documents/Tensin/Codes/TrasnLow/filesPreprosesing/fileRefined_UCF_Crime_I3D"
pathFeatures_Full = "/home/pc/Documents/Tensin/Codes/TrasnLow/fileRefineProsesing/fileRefine_UCF_Crime_I3D_Part2"
pathSaveNPY = "/home/pc/Documents/Tensin/Codes/TrasnLow/filesPreprosesing/featuresTrain"
labelsAbnormal = ['Arrest', 'Arson', 'Assault',
                  'Abuse']

colors = [        '#1f77b4', '#ff7f0e', '#2ca02c', 
                  '#d62728']


# labelsAbnormal = ['Abuse', 
#                 'Arrest', 
#                 'Arson',
#                 'Assault', 
#                 'Burglary', 
#                 'Explosion',
#                 'Fighting', 
#                 'RoadAccidents', 
#                 'Robbery',
#                 'Shooting', 
#                 'Shoplifting', 
#                 'Stealing',
#                 'Vandalism']

# colors = [
#     '#1f77b4',  # azul
#     '#ff7f0e',  # naranja
#     '#2ca02c',  # verde
#     '#d62728',  # rojo
#     '#9467bd',  # morado
#     '#8c564b',  # marrón
#     '#e377c2',  # rosa claro
#     '#7f7f7f',  # gris
#     '#bcbd22',  # verde amarillento
#     '#17becf',  # cyan
#     '#a55194',  # violeta oscuro
#     '#ffff00',  # durazno claro
#     '#4b4b4b'   # gris oscuro
# ]


print(f"[*] Database UCF-Crime classes: {labelsAbnormal}")
validationSize = 0.10 # Porcentaje de validacion
N = 300



# Preparacion Traning

# ---> NCA para el set de entrenamiento
namePlotSave = "clusters_NCA_Train_13Class_without.png"
pathSavePlot = os.path.join(pathFeatures_Full, "Cluster")
os.makedirs(pathSavePlot, exist_ok=True)
os.makedirs(pathSaveNPY, exist_ok=True)
pathSavePlot = os.path.join(pathSavePlot, namePlotSave)
features_all = []
labels_all = []


for index, nameClass in enumerate(labelsAbnormal):
    # pathLabels = os.path.join(pathFeatures_Full, (nameClass + "_Features"), (nameClass + "_labales_clipLevel.npy"))
    pathFeatures = os.path.join(pathFeatures_Full, "Train", nameClass)
    featuresName = os.listdir(pathFeatures)
    featuresName = sorted(featuresName)
    featuresConca = []


    for indexClip, feature_file in enumerate(featuresName):
        pathFeaturesVideo = os.path.join(pathFeatures, feature_file)
        featuresVideo = np.load(pathFeaturesVideo)
        featuresConca.append(featuresVideo)
        print(nameClass, feature_file, featuresConca[indexClip].shape)


    valitionVideos = -1 * int(validationSize * len(featuresConca))
    featuresConca = featuresConca[:valitionVideos]
    featuresConca = np.concatenate(featuresConca, axis=0)

    print(f" {len(featuresConca)} características de I3D de {featuresConca.shape[1]}x{featuresConca.shape[2]}  para clase de {nameClass}. ")
    indices_aleatorios = np.random.choice(featuresConca.shape[0], N, replace=False)
    featuresConca = featuresConca[indices_aleatorios]
    labalesNpy = [index] * featuresConca.shape[0]
    labels_all.append(labalesNpy)
    features_all.append(featuresConca)
    
    print(f" {len(featuresConca)} características de I3D de {featuresConca.shape[1]}x{featuresConca.shape[2]}  para clase de {nameClass}. ")
    # print(nameClass, labels_all[index][-5:], featuresConca.shape[0])

labels_all = np.concatenate(labels_all)
features_all = np.concatenate(features_all, axis=0)
features_reduced = np.mean(features_all, axis=1) # N x 2048

n_components = 2
# random_state = 42

print(f'[*] Loaded {len(labels_all)} labales for training, {labels_all}')
print(f'[*] Loaded {features_reduced.shape} videos for training')

# nca = make_pipeline(StandardScaler(),
#                     NeighborhoodComponentsAnalysis(n_components=n_components,
#                                                    random_state=random_state))
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=n_components))
print(features_reduced.shape)
features_nca = nca.fit_transform(features_reduced, labels_all)
print("Dimensiones de los datos reducidos en NCA:", features_nca.shape)

# Entrenar el modelo SVM en el conjunto de entrenamiento
# svm_model = SVC(kernel='linear', random_state=random_state)
svm_model = SVC(kernel='linear')
svm_model.fit(features_reduced, labels_all)

# Entrenar el modelo KNN en el conjunto de entrenamiento
knn_model = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos (n_neighbors)
knn_model.fit(features_reduced, labels_all)

custom_cmap = ListedColormap(colors)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_nca[:, 0], features_nca[:, 1], c=labels_all, cmap=custom_cmap, alpha=0.7)
for i, nombre_etiqueta in enumerate(labelsAbnormal):
    plt.scatter([], [], color=colors[i], label=nombre_etiqueta)
plt.title("Proyección de datos reducidos con Neighborhood Components Analysis (NCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend(title="Etiquetas de clase")
plt.savefig(pathSavePlot)
plt.show()



# Preparacion Testing

def confusionMat (gt, top_Scores):
    cm =confusion_matrix(gt,top_Scores)
    cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
    mean_avg_acc = np.sum(cm.diagonal()) / np.sum(cm)
    mean_avg_acc_norm=np.sum(cmn.diagonal())/ cm.shape[0]
    
    return cm, cmn, mean_avg_acc, mean_avg_acc_norm

def plot_confusion_matrices(cm, cmn, labels, mean_avg_acc, mean_avg_acc_norm, epoc = None, pathSave_Data = "", ModeName = "Test"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], xticklabels=labels, yticklabels=labels, cmap='Blues')
    axes[0].set_title('Confusion Matrix (mAA: ' + str(mean_avg_acc) + ')' )
    axes[0].set_ylabel('Predicted Label')
    axes[0].set_xlabel('Truel Label')

    sns.heatmap(cmn, annot=True, fmt='.2f', ax=axes[1], xticklabels=labels, yticklabels=labels, cmap='Blues')
    axes[1].set_title('Normalized Confusion Matrix (mAA: ' + str(mean_avg_acc_norm) + ')' + "-" + ModeName )
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('Truel Label')

    plt.tight_layout()
    # plt.savefig(pathSave_Data + "/" + str(epoc) + "_" + ModeName + "_" + str(mean_avg_acc_norm)+ '.png')
    # plt.savefig(pathSave_Data + "/Segement_" + str(epoc) + "_" + ModeName + "_" + '.png')
    plt.show()

# ---> SVM

pathFeatures_Full = "/home/pc/Documents/Tensin/Codes/TrasnLow/filesPreprosesing/fileRefined_UCF_Crime_I3D"
# Cargar y preparar el conjunto de prueba
features_test_all = []
labels_test_all = []

for index, nameClass in enumerate(labelsAbnormal):
    pathFeaturesTest = os.path.join(pathFeatures_Full, "Test", nameClass)
    featuresNameTest = sorted(os.listdir(pathFeaturesTest))
    featuresConcaTest = []
    for feature_file in featuresNameTest:
        pathFeaturesVideoTest = os.path.join(pathFeaturesTest, feature_file)
        featuresVideoTest = np.load(pathFeaturesVideoTest)
        featuresConcaTest.append(featuresVideoTest)

    featuresConcaTest = np.concatenate(featuresConcaTest, axis=0)
    labels_test_all.append([index] * featuresConcaTest.shape[0])
    features_test_all.append(featuresConcaTest)

labels_test_all = np.concatenate(labels_test_all)
features_test_all = np.concatenate(features_test_all, axis=0)
features_reduced_test = np.mean(features_test_all, axis=1)  # N x 2048


print(f'[*] Loaded {len(labels_test_all)} labales for testing, {labels_test_all}')
print(f'[*] Loaded {features_reduced_test.shape} videos for testing')

# Hacer predicciones en el conjunto de prueba
predictions = svm_model.predict(features_reduced_test)

# Calcular precisión del modelo
accuracy = accuracy_score(labels_test_all, predictions)
print(f"Precisión del modelo SVM en el conjunto de prueba: {accuracy:}")

cm, cmn, mean_avg_acc, mean_avg_acc_norm = confusionMat (labels_test_all, predictions)
plot_confusion_matrices(cm, cmn, labelsAbnormal, mean_avg_acc, mean_avg_acc_norm, 0, ModeName="Test")
print(f"mAA value: {mean_avg_acc}, mAA Norm value: {mean_avg_acc_norm}")


# --->K-NN

# Cargar y preparar el conjunto de prueba
features_test_all = []
labels_test_all = []

for index, nameClass in enumerate(labelsAbnormal):
    pathFeaturesTest = os.path.join(pathFeatures_Full, "Test", nameClass)
    featuresNameTest = sorted(os.listdir(pathFeaturesTest))
    featuresConcaTest = []
    for feature_file in featuresNameTest:
        pathFeaturesVideoTest = os.path.join(pathFeaturesTest, feature_file)
        featuresVideoTest = np.load(pathFeaturesVideoTest)
        featuresConcaTest.append(featuresVideoTest)

    featuresConcaTest = np.concatenate(featuresConcaTest, axis=0)
    labels_test_all.append([index] * featuresConcaTest.shape[0])
    features_test_all.append(featuresConcaTest)

labels_test_all = np.concatenate(labels_test_all)
features_test_all = np.concatenate(features_test_all, axis=0)
features_reduced_test = np.mean(features_test_all, axis=1)  # N x 2048

print(f'[*] Loaded {len(labels_test_all)} labels for testing, {labels_test_all}')
print(f'[*] Loaded {features_reduced_test.shape} videos for testing')

# Hacer predicciones en el conjunto de prueba
predictions = knn_model.predict(features_reduced_test)

# Calcular precisión del modelo
accuracy = accuracy_score(labels_test_all, predictions)
print(f"Precisión del modelo KNN en el conjunto de prueba: {accuracy:}")

cm, cmn, mean_avg_acc, mean_avg_acc_norm = confusionMat (labels_test_all, predictions)
plot_confusion_matrices(cm, cmn, labelsAbnormal, mean_avg_acc, mean_avg_acc_norm, 0, ModeName="Test")
print(f"mAA value: {mean_avg_acc}, mAA Norm value: {mean_avg_acc_norm}")
