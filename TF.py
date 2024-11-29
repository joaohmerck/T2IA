import numpy as np
import random

# Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initialize_weights()

    def initialize_weights(self):
        self.input_hidden_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.hidden_output_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.hidden_bias = np.random.uniform(-1, 1, self.hidden_size)
        self.output_bias = np.random.uniform(-1, 1, self.output_size)

    def forward(self, x):
        # Normaliza a entrada: 1 para jogador, -1 para oponente, 0 para vazio
        x = np.array(x, dtype=float)
        
        # Processa a camada oculta
        hidden_layer_input = np.dot(x, self.input_hidden_weights) + self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        
        # Processa a camada de saída
        output_layer_input = np.dot(hidden_layer_output, self.hidden_output_weights) + self.output_bias
        output = self.sigmoid(output_layer_input)
        
        return output

    @staticmethod
    def sigmoid(x):
        # Limita x para evitar overflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))


# Algoritmo Minimax com Poda Alfa-Beta
class MinimaxTrainer:
    def __init__(self):
        self.memo = {}

    def minimax_move(self, board, is_maximizing, alpha=float('-inf'), beta=float('inf')):
        board_tuple = tuple(board)
        if board_tuple in self.memo:
            return self.memo[board_tuple]

        winner = check_winner(board)
        if winner != 0:
            return 10 if winner == -1 else -10
        if all(x != 0 for x in board):
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1
                    score = self.minimax_move(board, False, alpha, beta)
                    board[i] = 0
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
            self.memo[board_tuple] = best_score
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1
                    score = self.minimax_move(board, True, alpha, beta)
                    board[i] = 0
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
            self.memo[board_tuple] = best_score
            return best_score

    def choose_move(self, board, mode):
        if mode == 'easy':
            # 25% de chance de usar o Minimax
            if random.random() < 0.25:
                return self.best_move(board)
            else:
                return random.choice([i for i, x in enumerate(board) if x == 0])
        elif mode == 'medium':
            # 50% de chance de usar o Minimax
            if random.random() < 0.5:
                return self.best_move(board)
            else:
                return random.choice([i for i, x in enumerate(board) if x == 0])
        elif mode == 'hard':
            # Sempre usa o Minimax
            return self.best_move(board)


    def best_move(self, board):
        best_score = float('-inf')
        best_move = None
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                score = self.minimax_move(board, False)
                board[i] = 0
                if score > best_score:
                    best_score = score
                    best_move = i
        return best_move


# Algoritmo Genético
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
                parent1 = self.tournament_selection(sorted_population, scores)
                parent2 = self.tournament_selection(sorted_population, scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            self.population = next_generation
            print(f"Melhor aptidão da geração {generation + 1}: {max(scores)}")
        return sorted_population[0]

    def tournament_selection(self, population, scores, k=3):
        indices = random.sample(range(len(population)), k)
        candidates = [population[i] for i in indices]
        return max(candidates, key=lambda ind: scores[self.population.index(ind)])

    def fitness(self, network, mode):
        score = 0
        num_games = 100
        for _ in range(num_games):
            result = self.play_game(network, mode)
            if result == 1:  # vitória
                score += 20
            elif result == 0:  # empate
                score += 10  # Aumentado de 5 para 10 para valorizar mais empates
            else:  # derrota
                score -= 30  # Aumentado de -10 para -30 para penalizar mais as derrotas
        return 5*(score / num_games)  # normaliza o score pelo número de jogos

    def play_game(self, network, mode):
        board = [0] * 9
        current_player = 1
        for _ in range(9):
            if current_player == 1:
                move = self.network_move(network, board)
                board[move] = 1
            else:
                move = self.minimax_trainer.choose_move(board, mode)
                board[move] = -1
            current_player *= -1

            winner = check_winner(board)
            if winner == 1:
                return 1
            elif winner == -1:
                return -1
        return 0

    def network_move(self, network, board):
        board_input = np.array(board)
        output = network.forward(board_input)
        possible_moves = [i for i, x in enumerate(board) if x == 0]
        
        if not possible_moves:
            return None
        
        # Pega apenas os scores dos movimentos possíveis
        move_scores = np.array([output[i] for i in possible_moves])
        
        # Aplica softmax nos scores válidos para melhor distribuição
        exp_scores = np.exp(move_scores)
        move_probabilities = exp_scores / np.sum(exp_scores)
        
        # Reduz a exploração aleatória para focar mais no aprendizado
        if np.random.random() < 0.05:  
            return random.choice(possible_moves)
            
        # Escolhe sempre o melhor movimento durante o treinamento
        return possible_moves[np.argmax(move_probabilities)]

    def crossover(self, parent1, parent2):
        weight_mix = random.random()
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        
        # Mistura dos pesos
        child.input_hidden_weights = weight_mix * parent1.input_hidden_weights + (1 - weight_mix) * parent2.input_hidden_weights
        child.hidden_output_weights = weight_mix * parent1.hidden_output_weights + (1 - weight_mix) * parent2.hidden_output_weights
        
        # Mistura dos biases
        child.hidden_bias = weight_mix * parent1.hidden_bias + (1 - weight_mix) * parent2.hidden_bias
        child.output_bias = weight_mix * parent1.output_bias + (1 - weight_mix) * parent2.output_bias
        
        return child

    def mutate(self, network):
        mutation_rate = 0.05  # Probabilidade de mutação
        if random.random() < mutation_rate:
            # Aplica mutação diretamente com valores da distribuição normal padrão
            network.input_hidden_weights += np.random.normal(0, 1, network.input_hidden_weights.shape)
            network.hidden_output_weights += np.random.normal(0, 1, network.hidden_output_weights.shape)
            network.hidden_bias += np.random.normal(0, 1, network.hidden_bias.shape)
            network.output_bias += np.random.normal(0, 1, network.output_bias.shape)



# Funções de utilidade
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


def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nTabuleiro:")
    for i in range(3):
        row = [symbols[board[i * 3 + j]] for j in range(3)]
        print(" | ".join(row))
        if i < 2:
            print("---------")


def network_move(network, board):
    """
    Função para calcular o próximo movimento da rede neural treinada.
    """
    board_input = np.array(board)
    output = network.forward(board_input)
    possible_moves = [i for i, x in enumerate(board) if x == 0]
    move_scores = [output[i] for i in possible_moves]
    return possible_moves[np.argmax(move_scores)]


def play_with_trained_network(network):
    """
    Função para jogar contra a rede neural treinada.
    """
    print("Jogando contra a rede neural treinada:")
    board = [0] * 9
    current_player = 1  # Jogador humano começa
    while True:
        print_board(board)
        if current_player == 1:
            move = int(input("Escolha uma posição (1-9): ")) - 1
            if 0 <= move < 9 and board[move] == 0:
                board[move] = 1
                current_player = -1
            else:
                print("Posição inválida. Tente novamente.")
        else:
            move = network_move(network, board)
            board[move] = -1
            print(f"A rede neural escolheu a posição {move + 1}.")
            current_player = 1

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


if __name__ == "__main__":
    print("Bem-vindo ao Jogo da Velha com IA!")
    minimax_trainer = MinimaxTrainer()
    population_size = 500
    input_size = 9
    hidden_size = 9
    output_size = 9

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
