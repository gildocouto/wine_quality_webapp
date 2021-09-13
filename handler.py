import pandas as pd
import pickle, os
from flask import Flask, request

from wine_quality.WineQuality import WineQuality

#Load modelo ML
model = pickle.load(open('model/model_wine_quality.pkl','rb'))

# Instanciar o Flask
app = Flask( __name__ )

#end-point do flask
@app.route( '/predict',methods=['POST'] ) #método tipo post porque vai enviar dados, se fosse get podia só recuperar dados
#Essa rota (route) vai redirecionar para a função abaixo. Toda que vez que acessar esse end-point essa função (predict) vai rodar
def predict():
    test_json = request.get_json()

    #Coletor de dados
    if test_json: #Saber se retornou algum dados, se não é vazio.
        if isinstance( test_json, dict):# Ver se é valor único, se for uma linha somente
            df_raw = pd.DataFrame( test_json, index=[0] )
        else:
            df_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

    #Preparação dos dados (acrescentar código de preparação de dados)

    #Predição
    pred = model.predict( df_raw)

    df_raw['prediction'] = pred

    return df_raw.to_json( orient='records') #Retornar resultado da predição no mesmo formato que veio (json) e mesma orientação (orient)

if __name__ == '__main__':
    # Iniciar Flask
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0', port=port)


