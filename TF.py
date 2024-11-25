import numpy as np
from minimax import Minimax
from algoritmo_genetico import AlgoritmoGenetico
from utils.confere_vencedor import confere_vencedor

# Funções de utilidade para o jogo da velha
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nTabuleiro:")
    for i in range(3):
        row = [symbols[board[i * 3 + j]] for j in range(3)]
        print(" | ".join(row))
        if i < 2:
            print("---------")

# Jogar contra o Minimax
def jogar_com_minimax(minimax, dificuldade):
    board = [0] * 9
    turno_jogador = True
    while True:
        print_board(board)
        if turno_jogador:
            jogada = int(input("Escolha uma posição (1-9): ")) - 1
            if 0 <= jogada < 9 and board[jogada] == 0:
                board[jogada] = 1
                turno_jogador = False
            else:
                print("Posição inválida ou já ocupada. Tente outra.")
        else:
            jogada = minimax.escolhe_jogada(board, dificuldade)
            board[jogada] = -1
            print(f"O Minimax escolheu a posição {jogada + 1}.")
            turno_jogador = True

        winner = confere_vencedor(board)
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
def play_with_trained_rede(rede):
    board = [0] * 9
    turno_jogador = True
    while True:
        print_board(board)
        if turno_jogador:
            jogada = int(input("Escolha uma posição (1-9): ")) - 1
            if 0 <= jogada < 9 and board[jogada] == 0:
                board[jogada] = 1
                turno_jogador = False
            else:
                print("Posição inválida ou já ocupada. Tente outra.")
        else:
            jogada = rede_jogada(rede, board)
            board[jogada] = -1
            print(f"A rede neural escolheu a posição {jogada + 1}.")
            turno_jogador = True

        winner = confere_vencedor(board)
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

def rede_jogada(rede, board):
    board_input = np.array(board).reshape(-1)
    output = rede.forward(board_input)
    possible_jogadas = [i for i, x in enumerate(board) if x == 0]
    return possible_jogadas[np.argmax([output[i] for i in possible_jogadas])]

# Interface de Console
if __name__ == "__main__":
    print("Bem-vindo ao Jogo da Velha com IA!")
    minimax = Minimax()
    populacao_tamanho = 10
    entrada = 9
    oculta = 9
    saida = 9

    # Menu de opções
    trained_rede = None
    while True:
        print("\nEscolha uma opção:")
        print("1. Jogar contra o Minimax")
        print("2. Treinar a rede neural jogando contra o Minimax")
        print("3. Jogar contra a rede neural treinada")
        print("4. Sair")

        choice = input("Sua escolha: ").strip()

        if choice == '1':
            dificuldade = input("Escolha a dificuldade (easy, medium, hard): ").strip().lower()
            jogar_com_minimax(minimax, dificuldade)

        elif choice == '2':
            generations = int(input("Número de gerações para o treinamento: "))
            dificuldade = input("Dificuldade para o treinamento (easy, medium, hard): ").strip().lower()
            algoritmo_genetico = AlgoritmoGenetico(populacao_tamanho, entrada, oculta, saida, minimax)
            trained_rede = algoritmo_genetico.evolve_population(generations, dificuldade)

        elif choice == '3':
            if trained_rede:
                play_with_trained_rede(trained_rede)
            else:
                print("A rede neural ainda não foi treinada! Selecione a opção 2 primeiro.")

        elif choice == '4':
            print("Saindo...")
            break
        
        else:
            print("Opção inválida. Tente novamente.")
