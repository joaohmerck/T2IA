import random
from utils.confere_vencedor import confere_vencedor

class Minimax:
    def __init__(self):
        # Dicionário para armazenar estados de tabuleiro já avaliados
        self.memo = {}

    def minimax_joga(self, board, is_maximizing):
        # Converter o tabuleiro em uma tupla para usá-lo como chave no dicionário de memoização
        board_tuple = tuple(board)
        
        # Se o estado já foi avaliado antes, retornar o resultado armazenado
        if board_tuple in self.memo:
            return self.memo[board_tuple]

        # Verificar o vencedor
        vencedor = confere_vencedor(board)
        if vencedor != 0:  # Se há um vencedor
            return 10 if vencedor == -1 else -10  # Minimax ganha = 10, perde = -10
        if all(x != 0 for x in board):  # Empate
            return 0

        melhor_score = float('-inf') if is_maximizing else float('inf')
        melhor_jogada = None

        for i in range(9):
            if board[i] == 0:
                # Fazer a jogada temporária
                board[i] = -1 if is_maximizing else 1
                score = self.minimax_joga(board, not is_maximizing)  # Recursão
                board[i] = 0  # Desfazer a jogada

                # Atualizar a melhor pontuação
                if is_maximizing:
                    if score > melhor_score:
                        melhor_score = score
                        melhor_jogada = i
                else:
                    if score < melhor_score:
                        melhor_score = score
                        melhor_jogada = i

        # Armazenar o melhor resultado para o estado atual no dicionário de memoização
        self.memo[board_tuple] = melhor_jogada if melhor_jogada is not None else melhor_score
        return melhor_jogada if melhor_jogada is not None else melhor_score

    def escolhe_jogada(self, board, dificuldade):
        if dificuldade == 'easy':
            use_minimax = random.random() < 0.25
        elif dificuldade == 'medium':
            use_minimax = random.random() < 0.50
        elif dificuldade == 'hard':
            use_minimax = True
        else:
            raise ValueError("Modo desconhecido: Use 'easy', 'medium' ou 'hard'")

        if use_minimax:
            return self.minimax_joga(board, True)
        else:
            return random.choice([i for i, x in enumerate(board) if x == 0])