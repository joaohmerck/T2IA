def confere_vencedor(board):
    combinacoes_vitoria = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for combo in combinacoes_vitoria:
        line_sum = board[combo[0]] + board[combo[1]] + board[combo[2]]
        if line_sum == 3:
            return 1  # Rede neural venceu
        elif line_sum == -3:
            return -1  # Minimax venceu
    return 0  # Empate