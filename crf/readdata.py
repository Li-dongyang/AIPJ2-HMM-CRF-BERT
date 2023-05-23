

def read_data(file_path):
    data = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                tokens = line.split(' ')
                word = tokens[0]
                label = tokens[1]
                sentence.append((word, label))
            else:
                if sentence:
                    data.append(sentence)
                    sentence = []
    return data

if __name__ == '__main__':
    language = "Chinese"
    train_file = f'..\\data\\{language}\\train22000.txt' if language == "English" else f'..\\data\\{language}\\train1150.txt' 
    val_file = f'..\\data\\{language}\\validation22000.txt' if language == "English" else f'..\\data\\{language}\\validation1150.txt'
    
    train_data = read_data(train_file)
    val_data = read_data(val_file)

    # print(train_data)
    print(val_data)
