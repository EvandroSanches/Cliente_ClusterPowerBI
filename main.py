from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import make_column_transformer
from pandasgui import show
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

k = 3
batch_size = 4
epochs = 100

def CarregaDados():
    df = pd.read_csv('Data/customer_segmentation.csv')

    imputer = SimpleImputer()
    label_encoder = OrdinalEncoder()
    normalizador = MinMaxScaler()
    pca = PCA(n_components=2)

    x = np.asarray(df.Income)
    x = np.expand_dims(x, axis=1)
    imputer.fit(x)
    df.Income = imputer.transform(x)

    df = df.drop(['ID'], axis=1)
    df = df.drop(['Dt_Customer'], axis=1)
    df = df.drop(['Z_CostContact'], axis=1)
    df = df.drop(['Z_Revenue'], axis=1)
    df = df.drop(['Response'], axis=1)

    label = np.asarray(df.Education)
    label = np.expand_dims(label, axis=1)

    df.Education = label_encoder.fit_transform(label)
    encoder = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['Marital_Status']),
                remainder='passthrough', sparse_threshold=False)
    df = encoder.fit_transform(df)

    previsores = normalizador.fit_transform(df)
    print(previsores.shape)

    previsores = AutoEncoder_(previsores)
    #previsores = pca.fit_transform(previsores)

    return previsores

def elbow():
    #Carregando dados
    dados = CarregaDados()

    #Variaveis de valores de K e Score
    valores_k = []
    inertias = []

    #Armazenando score para cada valor de K em um intervalo do 1 a 15
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(dados)
        valores_k.append(i)
        inertias.append(kmeans.inertia_)

    #Visualizando grafico
    plt.plot(valores_k, inertias)
    plt.title('Validação de ELbow')
    plt.xlabel('Valores de K')
    plt.ylabel('Inertia')
    plt.show()

def AutoEncoder_(previsores):
    previsores_treino = previsores[:1500]
    previsores_teste = previsores[1500:]

    #Definindo callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    md_encoder = ModelCheckpoint(filepath='Encoder.0.1', save_best_only=True, verbose=1)

    #Carregando modelo de Autoencoder caso existente ou treinando novo caso inexistente
    try:
        encoder = load_model('Encoder.0.1')
    except:
        encoder = Sequential()
        encoder.add(Dense(units=16, activation='relu', input_dim=31))
        encoder.add(Dense(units=8, activation='relu'))
        encoder.add(Dense(units=4, activation='relu'))
        encoder.add(Dense(units=8, activation='relu'))
        encoder.add(Dense(units=16, activation='relu'))
        encoder.add(Dense(units=31, activation='sigmoid'))

        encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        encoder.fit(previsores_treino, previsores_treino, batch_size=batch_size, epochs=epochs, callbacks=[es, rlp, md_encoder], validation_data=(previsores_teste, previsores_teste))

    #Definindo encoder a partir do modelo de encode/decode
    dimensao_original = Input(shape=(31,))
    camada_encoder1 = encoder.layers[0]
    camada_encoder2 = encoder.layers[1]
    camada_encoder3 = encoder.layers[2]

    encode = Model(dimensao_original, camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))

    resultado = encode.predict(previsores)

    return resultado

def validacao_SR():
    #Carregando dados
    dados = CarregaDados()

    #Variaveis de valores de K e Score
    valores_k = []
    SR = []

    #Armazenando score para cada valor de K em um intervalo do 2 a 15 (K = 1 gera erro)
    for i in range(2,15):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(dados)
        valores_k.append(i)
        SR.append(silhouette_score(dados, kmeans.labels_))

    #Visualizando grafico
    plt.plot(valores_k, SR)
    plt.title('Validação SR')
    plt.xlabel('Valores de K')
    plt.ylabel('SR Score')
    plt.show()


def Cleturing():
    #Carregando dados
    dados = CarregaDados()

    #Clusterizando dados
    kmeans = KMeans(n_clusters=k, random_state=0).fit(dados)

    #Atribuindo clusterização aos dados
    cluster_map = pd.read_csv('Data/customer_segmentation.csv')
    cluster_map['Grupo'] = kmeans.labels_

    #Definindo limites do eixo X e Y do grafico
    x_min, x_max = dados[:,0].min() - 0.3, dados[:,0].max() + 0.3
    y_min, y_max = dados[:,1].min() - 0.3, dados[:,1].max() + 0.3

    #Definindo centroids
    centroids = kmeans.cluster_centers_

    #Plotando grafico com dados e centroids
    plt.scatter(dados[:,0], dados[:,1], c=kmeans.labels_)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=169, linewidths=3, color='r', zorder=8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    #Exebindo dados com coluna de clusterização

    cluster_map.rename(columns={'Income':'Renda', 'Kidhome':'Crianças', 'Teenhome':'Adolescentes', 'MntWines':'Vinhos', 'MntFruits':'Frutas', 'MntMeatProducts':'Carnes',
                                 'MntFishProducts':'Produtos de Pesca', 'MntSweetProducts':'Doces', 'MntGoldProds':'Produtos de Ouro', 'NumDealsPurchases':'Compras por Promoções',
                                'NumWebPurchases':'Compras no Site', 'NumCatalogPurchases':'Compras por Catálogo', 'NumStorePurchases':'Compras na Loja', 'NumWebVisitsMonth':'Visitas no Site'}, inplace=True)

    cluster_map.to_csv('Dados_Clusterizados')

    show(cluster_map)

elbow()
validacao_SR()
Cleturing()
