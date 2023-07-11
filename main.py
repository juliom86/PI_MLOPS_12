from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    df = pd.read_csv('leng.csv')
    l = df.loc[df.language == idioma]
    l = l['number'].to_list()
    return {'idioma':idioma, 'cantidad':l[0]}

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    df = pd.read_csv('dur.csv')
    d = df.loc[df.title == pelicula]
    c = d['runtime'].to_list()[0]
    m = d['year'].to_list()[0]

    return {'pelicula':pelicula, 'duracion':c, 'anio':m}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    df = pd.read_csv('franq.csv')
    d = df.loc[df.belongs_to_collection == franquicia]
    c = d['count'].to_list()[0]
    m = d['mean'].to_list()[0]
    s = d['sum'].to_list()[0]
    return {'franquicia':franquicia, 'cantidad':c, 'ganancia_total':s, 'ganancia_promedio':m}


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    df = pd.read_csv('paispeli.csv')
    d = df.loc[df.country == pais]
    d = d.num_movies.to_list()
    return {'pais':pais, 'cantidad':d[0]}


@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora):
    '''Ingresas la productora, retornando la ganancia toal y la cantidad de peliculas que produjeron'''
    df = pd.read_csv('prod.csv')
    d = df.loc[df.companies == productora]
    c = d['Number'].to_list()[0]
    m = d['Average'].to_list()[0]
    s = d['Total'].to_list()[0]
    return {'productora':productora, 'ganancia_total':s, 'cantidad':c}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno.
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
    df = pd.read_csv('dir.csv')
    df1 = pd.read_csv('dir_pel.csv')
    d = df.loc[df.director == nombre_director]
    r = d['return'].to_list()

    b = df1[(df1['director'] == nombre_director) & (df1['return'].notnull())]
    g = b['title'].to_list()
    a = b['year'].to_list()
    rr = b['return'].to_list()
    bd = b['budget'].to_list()
    rv = b['revenue'].to_list()
    return {'director':nombre_director, 'retorno_total_director':r,
    'peliculas':g, 'anio':a, 'retorno_pelicula':rr,
    'budget_pelicula':bd, 'revenue_pelicula':rv}

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo):
    '''Ingresasun nombre de pelicula y te recomienda las similares en una lista'''
    i = pd.read_csv("titulos.csv").iloc[:6000]
    tfidf = TfidfVectorizer(stop_words="english")
    i["overview"] = i["overview"].fillna("")

    tfidf_matriz = tfidf.fit_transform(i["overview"])
    coseno_sim = linear_kernel(tfidf_matriz, tfidf_matriz)

    indices = pd.Series(i.index, index=i["title"]).drop_duplicates()
    idx = indices[titulo]
    simil = list(enumerate(coseno_sim[idx]))
    simil = sorted(simil, key=lambda x: x[1], reverse=True)
    simil = simil[1:11]
    movie_index = [i[0] for i in simil]

    lista = i["title"].iloc[movie_index].to_list()[:5]

    return {'lista recomendada': lista}
