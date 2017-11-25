def generate_errors(words, error_type, dictionary, fixed_size=-1):
    dataset = []

    if (error_type == 0):
        """ No Error"""
        for word in words:
            if fixed_size == -1 or fixed_size == len(word):
                dataset.append((word, word, error_type))


    if (error_type == 1):
        """ Skip letter error """
        for word in words:
            errors = []
            # n total errors
            for k in range(len(word)):
                e = ""
                for i in range(len(word)):
                    if i != k:
                        e += str(word[i])
                if (e not in dictionary) and (e not in errors):
                    if fixed_size == -1 or fixed_size == len(e):
                        errors.append(e)

            for error in errors:
                dataset.append((word, error, error_type, ))

    elif (error_type == 2):
        """ Swap letter error """
        for word in words:
            errors = []
            for k in range(len(word)-1):
                e = ""
                for i in range(len(word)):
                    if i == k:
                        e += word[k+1]
                    elif i == k+1:
                        e += word[k]
                    else:
                        e += word[i]
                if (e not in dictionary) and (e not in errors):
                    if fixed_size == -1 or fixed_size == len(e):
                        errors.append(e)

            for error in errors:
                dataset.append((word, error, error_type))

    elif (error_type == 3):
        """ Double letters error """
        for word in words:
            errors = []
            for k in range(len(word)):
                e = ""
                for i in range(len(word)):
                    if i == k:
                        e += word[i]
                    e += word[i]

                if (e not in dictionary) and (e not in errors):
                    if fixed_size == -1 or fixed_size == len(e):
                        errors.append(e)

            for error in errors:
                dataset.append((word, error, error_type))

    return dataset


def read_data(filename):
    # Read the input file

    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        records = f.read().splitlines()

    return  records, dict((el,0) for el in records)

def save_data(dataset, filename):
    file = open(filename, 'w')

    for i in dataset:
        file.write(i[0] + '\t')
        file.write(i[1] + '\t')
        file.write(str(i[2]) + '\n')


def main():
    error_type = 0
    file_name = "errors" + str(error_type)

    words, dictionary = read_data("english_dictionary.txt")

    dataset = generate_errors(words, error_type, dictionary)

    save_data(dataset, file_name)


if __name__ == '__main__':
    main()