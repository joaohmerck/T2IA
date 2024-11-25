import numpy as np
import random
from rede_neural import RedeNeural
from utils.confere_vencedor import confere_vencedor

class AlgoritmoGenetico:
    def __init__(self, populacao_tamanho, input_size, hidden_size, output_size, minimax):
        self.populacao_tamanho = populacao_tamanho
        self.minimax = minimax
        self.population = [RedeNeural(input_size, hidden_size, output_size) for _ in range(populacao_tamanho)]

    def evolve_population(self, generations, dificuldade):
        for generation in range(generations):
            print(f"Treinando geração {generation + 1}/{generations}")
            scores = [self.fitness(rede, dificuldade) for rede in self.population]
            sorted_population = [rede for _, rede in sorted(zip(scores, self.population), key=lambda x: -x[0])]

            next_generation = []
            elite_count = self.populacao_tamanho // 4
            next_generation.extend(sorted_population[:elite_count])

            while len(next_generation) < self.populacao_tamanho:
                parent1, parent2 = random.sample(sorted_population[:elite_count], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            self.population = next_generation
            best_fitness = max(scores)
            print(f"Melhor aptidão da geração {generation + 1}: {best_fitness}")
        print("Treinamento concluído!")
        return sorted_population[0]  # Retorna a melhor rede neural

    def fitness(self, rede, dificuldade):
        score = 0
        games_to_play = 15  # Jogar várias partidas para uma média de desempenho

        for _ in range(games_to_play):
            result = self.play_game(rede, dificuldade)
            if result == 1:
                score += 20  # Pontuação mais alta para vitória
            elif result == -1:
                score -= 10  # Penalização para derrota
            elif result == 0:
                score += 5   # Pequena recompensa para empates

        return score

    def play_game(self, rede, dificuldade):
        board = [0] * 9
        current_player = 1  # 1 para a rede neural, -1 para Minimax
        for _ in range(9):
            if current_player == 1:
                jogada = self.rede_jogada(rede, board)
                board[jogada] = 1
            else:
                jogada = self.minimax.escolhe_jogada(board, dificuldade)
                board[jogada] = -1
            current_player *= -1

            if confere_vencedor(board) == 1:
                return 1  # Vitória da rede neural
            elif confere_vencedor(board) == -1:
                return -1  # Vitória do Minimax

        return 0  # Empate

    def rede_jogada(self, rede, board):
        board_input = np.array(board).reshape(-1)
        output = rede.forward(board_input)
        possible_jogadas = [i for i, x in enumerate(board) if x == 0]
        return possible_jogadas[np.argmax([output[i] for i in possible_jogadas])]

    def crossover(self, parent1, parent2):
        child_weights_input_hidden = (parent1.weights_input_hidden + parent2.weights_input_hidden) / 2
        child_weights_hidden_output = (parent1.weights_hidden_output + parent2.weights_hidden_output) / 2
        child = RedeNeural(parent1.weights_input_hidden.shape[0], parent1.weights_input_hidden.shape[1], 1)
        child.set_weights(child_weights_input_hidden, child_weights_hidden_output)
        return child

    def mutate(self, rede):
        mutation_rate = 0.1
        rede.weights_input_hidden += mutation_rate * np.random.uniform(-1, 1, rede.weights_input_hidden.shape)
        rede.weights_hidden_output += mutation_rate * np.random.uniform(-1, 1, rede.weights_hidden_output.shape)