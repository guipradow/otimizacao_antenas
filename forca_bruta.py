from funcao_objetivo import calcula_habitantes
from itertools import combinations
from funcao_objetivo import calcula_habitantes
import pandas as pd

# Lista de antenas disponiveis
antenas_disponiveis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# Inicia contador de combinação
n_comb = 0
# Inicia melhor solução com valor 0
best_fitness = 0
# Inicia a melhor combinação com lista vazia
best_solution = []

# Loop para combinações selecionando de 1 a 7 letras, sem repetição
for i in range(1, 8):
    # Gera as combinações de tamanho i
    combs = combinations(antenas_disponiveis, i)
    # Loop sobre as combinações
    for comb in combs:
        # Incrementa o número de combinações
        n_comb += 1
        solucao = calcula_habitantes(comb)[0]
        if solucao > best_fitness:
            best_fitness = solucao
            best_solution = [1 if antena in comb else 0 for antena in antenas_disponiveis]
        # Imprime o número da combinação e a combinação atual
        # print(f'Comb {n_comb}: {comb}')
        # Adiciona a combinação à lista de combinações
       
  # Dict resposta -> xlsx
    resposta = {key: value for key, value in zip(antenas_disponiveis, best_solution)}
    df_resposta = pd.DataFrame(resposta.items(), columns=['Antena', 'Antenas a construir'])
    df_resposta.to_excel('./solutions/resposta_fb.xlsx', index=False)
