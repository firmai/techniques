import pandas as pd
import numpy as np

def clean_data():
    ''' clean data and split dataset into two subsets '''
    df = pd.read_table('data.txt')
    df = df.drop(columns='Unnamed: 0', axis=1)
    df = df[df['country'].isnull() == False]
    country_list = []
    for i in range(df.shape[0]):
        if (df['country'].tolist()[i].isdigit() == False) and (df['country'].tolist()[i].isalpha() == True):
            country_list.append(df['country'].tolist()[i])
    df = df[df.country.isin(country_list)]

    df['height'] = pd.to_numeric(df['height'],errors='coerce')
    df['width'] = pd.to_numeric(df['width'],errors='coerce')
    df = df[df['height'].isnull() == False]
    df = df[df['width'].isnull() == False]

    material_string = df['material']
    material_string = material_string.str.lower()
    df['material'] = material_string

    famous_artists = ['Pablo Picasso', 'Andy Warhol', 'Gustav Klimt', 'Paul Cezanne', 'Edvard Munch', 'Vincent van Gogh', 'Mark Rothko']
    df_top_artists = df[df.artist.apply(lambda x: x in top_artists)]
    df_famous_artists = df[df.artist.apply(lambda x: x not in top_artists)]
    return df_famous_artists, df_famous_artists, df

def clean_small_artist_data(df_small_artists):
    df_small_artists['if_oil_on_canvas'] = df_small_artists.material.apply(lambda x: x=='oil_on_canvas').astype(int)
    df_small_artists['if_sells_2000'] = df_small_artists.price.apply(lambda x: x > 2000).astype(int)
    df_small_artists = pd.get_dummies(df_small_artists, columns=['country', 'FaceCount', 'dominantColor'])
    df_small_artists = df_small_artists.drop(columns=['yearOfBirth', 'yearOfDeath', 'link', 'source', 'soldtime', 'year', 'name', 'material', 'artist', 'price'])
    return df_small_artists

def clean_famous_artist_data(df_famous_artists):
    df_famous_artists['if_sells_20000'] = df_famous_artists.price.apply(lambda x: x > 20000).astype(int)
    df_famous_artists['if_oil_on_canvas'] = df_famous_artists.material.apply(lambda x: x == 'oil_on_canvas').astype(int)
    df_famous_artists['if_prints'] = df_famous_artists.material.apply(lambda x: x == 'prints').astype(int)
    df_famous_artists['other_materials'] = df_famous_artists.material.apply(lambda x: x!='prints' and x!='oil_on_canvas').astype(int)
    df_famous_artists = pd.get_dummies(df_famous_artists, columns=['FaceCount', 'dominantColor', 'artist'])
    df_famous_artists = df_famous_artists.drop(columns=['year', 'link', 'source', 'soldtime', 'material', 'price', 'name', 'yearOfBirth', 'yearOfDeath', 'country'])
    return df_famous_artists
