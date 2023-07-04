def aishell_id_to_piece(tokens,token2char):
    str = ""
    for id in tokens:
        str += token2char[id]

    return str