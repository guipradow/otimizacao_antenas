import pandas as pd
import math

def calcula_habitantes(lista_antenas):

    # Inicia a class que contém as informações da vila
    class Vila:
        def __init__(self, x, y, hab):
            self.x = x
            self.y = y
            self.hab = hab


    # Inicia a class que contém as informações das antenas
    class Antena:
        def __init__(self, x, y, custo):
            self.x = x
            self.y = y
            self.custo = custo

        def cobertura(self, vilas):
            x = self.x
            y = self.y
            dentro = []
            for vila in vilas.values():
                distancia = math.sqrt((vila.x - x)**2 + (vila.y - y)**2)
                if distancia <= 72:
                    dentro.append(vila)
            return dentro


    # Lê a guia contendo a informação das vilas do arquivo xlsx
    df_vilas = pd.read_excel(f'./files/cobertura_exemplo.xlsx', sheet_name='Vilarejos')

    # Converte a coluna Vilarejo de int para str
    df_vilas['Vilarejo'] = df_vilas['Vilarejo'].astype('str')

    # Transforma o df da linha anterior em um dicionário
    dict_vilas = df_vilas.set_index('Vilarejo').T.to_dict('list')

    # Lê a guia contendo a informação das antenas do arquivo xlsx
    df_antenas = pd.read_excel(f'./files/cobertura_exemplo.xlsx', sheet_name='Antenas')

    # Transforma o df da linha anterior em um dicionário
    dict_antenas = df_antenas.set_index('Antena').T.to_dict('list')

    # Cria um dicionário contendo objetos da classe Vila
    vilas = {key: Vila(*values) for key, values in dict_vilas.items()}

    # Cria um dicionário contendo objetos da classe Antena
    antenas = {key: Antena(*values) for key, values in dict_antenas.items()}

    lista_antenas_ = [Antena(*dict_antenas[key]) for key in lista_antenas]
    cobertura_ = set(sum([antena.cobertura(vilas) for antena in lista_antenas_], []))
    habitantes = sum(vila.hab for vila in cobertura_)

    return habitantes
