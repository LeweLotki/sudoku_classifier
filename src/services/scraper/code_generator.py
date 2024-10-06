import string

def alphanumeric_code_generator(start_code: str = '000000'):
    characters = string.digits + string.ascii_uppercase

    if len(start_code) != 6 or any(c not in characters for c in start_code):
        raise ValueError("The start code must be a 6-character alphanumeric string containing only 0-9 and A-Z.")

    current_code = list(start_code)

    while True:
        yield ''.join(current_code)

        for i in range(5, -1, -1):
            if current_code[i] == 'Z': 
                current_code[i] = '0'
            else:
                current_code[i] = characters[characters.index(current_code[i]) + 1]
                break

