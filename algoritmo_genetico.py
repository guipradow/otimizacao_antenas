from deap import base, creator, tools
from deap.algorithms import varAnd
from funcao_objetivo import calcula_habitantes
import random
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# Lista de antenas disponíveis
antenas_disponiveis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# Constantes do algoritmo genético
NGEN, CXPB, MUTPB = 100, .7, .3
mu, lambda_ = 5, 10
HOF_SIZE = 1

# Configura a semente para os geradores pseudorandômicos
#RANDOM_SEED = 42
#random.seed(RANDOM_SEED)

# Declaração da classe Toolbox que contém os operadores evolucionários
toolbox = base.Toolbox()

# Define a estratégia de maximização de objetivo único
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Cria a classe de índividuo baseada em lista
creator.create("Individual", list, fitness=creator.FitnessMax)

# Cria um operador para gerar indivíduos
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                toolbox.attr_bool, 9)

# Cria um operador para gerar população baseada em lista de indivíduos
toolbox.register("populationCreator", tools.initRepeat, list,
                 toolbox.individualCreator)

# Cria função fitness
def fitness_function(individual):
    lista_antenas = [antenas_disponiveis[i] for i in range(len(individual)) \
        if individual[i] == 1]
    return calcula_habitantes(lista_antenas)

# Define a função fitness como forma de avaliação
toolbox.register('evaluate', fitness_function)

# Define os operadores genéticos
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=.05)

# Define a estratégia de seleção
toolbox.register('select', tools.selRoulette)


# Define a rotina do algoritmo genético
def main():
    # Calcula estatísticas
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)
    stats.register('max', np.max)

    # Cria a população inicial (geração 0)
    pop = toolbox.populationCreator(n=15)

    # Define o objeto hall-of-fame
    hof = tools.HallOfFame(HOF_SIZE)

    # Define informações do logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Avalia toda a população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Atualiza o hall of fame com os indivíduos da população
    hof.update(pop)
    hof_size = len(hof.items) if hof.items else 0

    # Inclui as estatísticas
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # Inicia o processo evolutivo
    for g in range(1, NGEN):
        # Seleciona os indivíduos para a próxima geração
        offspring = toolbox.select(pop, len(pop) - hof_size)
        offspring = varAnd(offspring, toolbox, CXPB, MUTPB)
        
        # Clona os indivíduos selecionados
        offspring = list(map(toolbox.clone, offspring))

        # Avalia os indivíduos com fitness inválido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Adiciona os melhores indivíduos novamente à população
        offspring.extend(hof.items)

        # Atualiza o hall of fame com os filhos gerados
        hof.update(offspring)

        # Substitui a população atual pelos filhos gerados
        pop[:] = offspring

        # Atualiza as estatísticas da população atual ao logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=g, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook, hof


if __name__ == '__main__':
    # Ativando multiprocessamento
    pool = multiprocessing.Pool()
    toolbox.register('map', pool.map)
    
    # Inicia figura para plotagem das execuções do algoritmo genético
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i in range(10):
        pop, logbook, hof = main()
        # Printa a melhor solução encontrada
        print(f'''
Melhor solução encontrada: {hof.items[0]} com  uma cobertura de
{hof.items[0].fitness.values[0]:.0f} habitantes
''')
        
        # Extrai as estatísticas
        maxFitnessValues, meanFitnessValues = logbook.select('max', 'avg')

        # Plota as estatísticas
        plt.plot(maxFitnessValues)
    plt.xlabel('Geração')
    plt.ylabel('Cobertura (nº habitantes)')
    plt.title('Cobertura através das gerações')
    plt.show()
    pool.close()

