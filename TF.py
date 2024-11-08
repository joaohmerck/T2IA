import numpy as np
import random

# Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = self.sigmoid(output_layer_input)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def set_weights(self, weights_input_hidden, weights_hidden_output):
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output

# Algoritmo Minimax com Memoização e Modos de Dificuldade
class MinimaxTrainer:
    def __init__(self):
        # Dicionário para armazenar estados de tabuleiro já avaliados
        self.memo = {}

    def minimax_move(self, board, is_maximizing):
        # Converter o tabuleiro em uma tupla para usá-lo como chave no dicionário de memoização
        board_tuple = tuple(board)
        
        # Se o estado já foi avaliado antes, retornar o resultado armazenado
        if board_tuple in self.memo:
            return self.memo[board_tuple]

        # Verificar o vencedor
        winner = check_winner(board)
        if winner != 0:  # Se há um vencedor
            return 10 if winner == -1 else -10  # Minimax ganha = 10, perde = -10
        if all(x != 0 for x in board):  # Empate
            return 0

        best_score = float('-inf') if is_maximizing else float('inf')
        best_move = None

        for i in range(9):
            if board[i] == 0:
                # Fazer a jogada temporária
                board[i] = -1 if is_maximizing else 1
                score = self.minimax_move(board, not is_maximizing)  # Recursão
                board[i] = 0  # Desfazer a jogada

                # Atualizar a melhor pontuação
                if is_maximizing:
                    if score > best_score:
                        best_score = score
                        best_move = i
                else:
                    if score < best_score:
                        best_score = score
                        best_move = i

        # Armazenar o melhor resultado para o estado atual no dicionário de memoização
        self.memo[board_tuple] = best_move if best_move is not None else best_score
        return best_move if best_move is not None else best_score

    def choose_move(self, board, mode):
        if mode == 'easy':
            use_minimax = random.random() < 0.25
        elif mode == 'medium':
            use_minimax = random.random() < 0.50
        elif mode == 'hard':
            use_minimax = True
        else:
            raise ValueError("Modo desconhecido: Use 'easy', 'medium' ou 'hard'")

        if use_minimax:
            return self.minimax_move(board, True)
        else:
            return random.choice([i for i, x in enumerate(board) if x == 0])

# Algoritmo Genético (AG)
class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size, minimax_trainer):
        self.population_size = population_size
        self.minimax_trainer = minimax_trainer
        self.population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]

    def evolve_population(self, generations, mode):
        for generation in range(generations):
            print(f"Treinando geração {generation + 1}/{generations}")
            scores = [self.fitness(network, mode) for network in self.population]
            sorted_population = [network for _, network in sorted(zip(scores, self.population), key=lambda x: -x[0])]

            next_generation = []
            elite_count = self.population_size // 4
            next_generation.extend(sorted_population[:elite_count])

            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(sorted_population[:elite_count], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            self.population = next_generation
            best_fitness = max(scores)
            print(f"Melhor aptidão da geração {generation + 1}: {best_fitness}")
        print("Treinamento concluído!")
        return sorted_population[0]  # Retorna a melhor rede neural

    def fitness(self, network, mode):
        score = 0
        games_to_play = 15  # Jogar várias partidas para uma média de desempenho

        for _ in range(games_to_play):
            result = self.play_game(network, mode)
            if result == 1:
                score += 20  # Pontuação mais alta para vitória
            elif result == -1:
                score -= 10  # Penalização para derrota
            elif result == 0:
                score += 5   # Pequena recompensa para empates

        return score

    def play_game(self, network, mode):
        board = [0] * 9
        current_player = 1  # 1 para a rede neural, -1 para Minimax
        for _ in range(9):
            if current_player == 1:
                move = self.network_move(network, board)
                board[move] = 1
            else:
                move = self.minimax_trainer.choose_move(board, mode)
                board[move] = -1
            current_player *= -1

            if self.check_winner(board) == 1:
                return 1  # Vitória da rede neural
            elif self.check_winner(board) == -1:
                return -1  # Vitória do Minimax

        return 0  # Empate

    def network_move(self, network, board):
        board_input = np.array(board).reshape(-1)
        output = network.forward(board_input)
        possible_moves = [i for i, x in enumerate(board) if x == 0]
        return possible_moves[np.argmax([output[i] for i in possible_moves])]

    def check_winner(self, board):
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for combo in winning_combinations:
            line_sum = board[combo[0]] + board[combo[1]] + board[combo[2]]
            if line_sum == 3:
                return 1  # Rede neural venceu
            elif line_sum == -3:
                return -1  # Minimax venceu
        return 0  # Empate

    def crossover(self, parent1, parent2):
        child_weights_input_hidden = (parent1.weights_input_hidden + parent2.weights_input_hidden) / 2
        child_weights_hidden_output = (parent1.weights_hidden_output + parent2.weights_hidden_output) / 2
        child = NeuralNetwork(parent1.weights_input_hidden.shape[0], parent1.weights_input_hidden.shape[1], 1)
        child.set_weights(child_weights_input_hidden, child_weights_hidden_output)
        return child

    def mutate(self, network):
        mutation_rate = 0.1
        network.weights_input_hidden += mutation_rate * np.random.uniform(-1, 1, network.weights_input_hidden.shape)
        network.weights_hidden_output += mutation_rate * np.random.uniform(-1, 1, network.weights_hidden_output.shape)

# Funções de utilidade para o jogo da velha
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nTabuleiro:")
    for i in range(3):
        row = [symbols[board[i * 3 + j]] for j in range(3)]
        print(" | ".join(row))
        if i < 2:
            print("---------")

def check_winner(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for combo in winning_combinations:
        line_sum = board[combo[0]] + board[combo[1]] + board[combo[2]]
        if line_sum == 3:
            return 1
        elif line_sum == -3:
            return -1
    return 0

# Jogar contra o Minimax
def play_with_minimax(minimax_trainer, mode):
    board = [0] * 9
    player_turn = True
    while True:
        print_board(board)
        if player_turn:
            move = int(input("Escolha uma posição (1-9): ")) - 1
            if 0 <= move < 9 and board[move] == 0:
                board[move] = 1
                player_turn = False
            else:
                print("Posição inválida ou já ocupada. Tente outra.")
        else:
            move = minimax_trainer.choose_move(board, mode)
            board[move] = -1
            print(f"O Minimax escolheu a posição {move + 1}.")
            player_turn = True

        winner = check_winner(board)
        if winner == 1:
            print_board(board)
            print("Você ganhou!")
            break
        elif winner == -1:
            print_board(board)
            print("O Minimax ganhou!")
            break
        elif all(x != 0 for x in board):
            print_board(board)
            print("Empate!")
            break

# Jogar contra a rede neural treinada
def play_with_trained_network(network):
    board = [0] * 9
    player_turn = True
    while True:
        print_board(board)
        if player_turn:
            move = int(input("Escolha uma posição (1-9): ")) - 1
            if 0 <= move < 9 and board[move] == 0:
                board[move] = 1
                player_turn = False
            else:
                print("Posição inválida ou já ocupada. Tente outra.")
        else:
            move = network_move(network, board)
            board[move] = -1
            print(f"A rede neural escolheu a posição {move + 1}.")
            player_turn = True

        winner = check_winner(board)
        if winner == 1:
            print_board(board)
            print("Você ganhou!")
            break
        elif winner == -1:
            print_board(board)
            print("A rede neural ganhou!")
            break
        elif all(x != 0 for x in board):
            print_board(board)
            print("Empate!")
            break

def network_move(network, board):
    board_input = np.array(board).reshape(-1)
    output = network.forward(board_input)
    possible_moves = [i for i, x in enumerate(board) if x == 0]
    return possible_moves[np.argmax([output[i] for i in possible_moves])]

# Interface de Console
if __name__ == "__main__":
    print("Bem-vindo ao Jogo da Velha com IA!")
    minimax_trainer = MinimaxTrainer()
    population_size = 10
    input_size = 9
    hidden_size = 9
    output_size = 9

    # Menu de opções
    trained_network = None
    while True:
        print("\nEscolha uma opção:")
        print("1. Jogar contra o Minimax")
        print("2. Treinar a rede neural jogando contra o Minimax")
        print("3. Jogar contra a rede neural treinada")
        print("4. Sair")

        choice = input("Sua escolha: ").strip()
        if choice == '1':
            difficulty = input("Escolha a dificuldade (easy, medium, hard): ").strip().lower()
            play_with_minimax(minimax_trainer, difficulty)
        elif choice == '2':
            generations = int(input("Número de gerações para o treinamento: "))
            mode = input("Dificuldade para o treinamento (easy, medium, hard): ").strip().lower()
            genetic_algorithm = GeneticAlgorithm(population_size, input_size, hidden_size, output_size, minimax_trainer)
            trained_network = genetic_algorithm.evolve_population(generations, mode)
        elif choice == '3':
            if trained_network:
                play_with_trained_network(trained_network)
            else:
                print("A rede neural ainda não foi treinada! Selecione a opção 2 primeiro.")
        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")
