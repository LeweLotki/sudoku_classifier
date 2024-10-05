import string

def alphanumeric_code_generator():
    characters = string.digits + string.ascii_uppercase

    current_code = ['0'] * 6

    while True:
        yield ''.join(current_code)

        for i in range(5, -1, -1):
            if current_code[i] == 'Z': 
                current_code[i] = '0'
            else:
                current_code[i] = characters[characters.index(current_code[i]) + 1]
                break

