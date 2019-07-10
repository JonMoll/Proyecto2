import re # Reemplazar simbolos con una expresion regular
import json

def DeleteSymbols(line):
    symbols = "[',.!?()$-_/]"
    new_line = re.sub(symbols, '', line)

    return new_line

def DeleteTokens(line, token_begin, token_end):
    new_line = line.replace(token_begin, '')
    new_line = new_line.replace(token_end, '')

    return new_line

def GetSentence(line):
    sentence = DeleteTokens(line, '<text>', '</text>')
    sentence = sentence.replace('\n', '') # Eliminar saltos de linea
    sentence = sentence.lstrip() # Eliminar espacion en blanco del principio
    sentence = sentence.rstrip() # Eliminar espacion en blanco del final
    sentence = sentence.lower() # Minusculas
    sentence = DeleteSymbols(sentence)

    return sentence

def GetCategoryPolarity(line):
    words = line.split()
    category = words[1]
    polarity = words[2]

    # Eliminando letras sobrantes
    category = category[10:len(category)-1]
    polarity = polarity[10:len(polarity)-3]

    category = category.replace('/', ' ')

    if polarity not in ['positive', 'negative', 'neutral']:
        polarity = 'neutral'

    return category, polarity

# ======================================================================

def main():
    path_dataset_train = './datasets/restaurants_train.xml'
    path_dataset_trial = './datasets/restaurants_trial.xml'
    paths = [path_dataset_train, path_dataset_trial]

    for path in paths:
        json_data = {}
        json_data['opinions'] = []

        with open(path, 'r') as xml_file:
            lines = xml_file.readlines()

            sentence = ''
            aspects = []

            for line in lines:
                if '<text>' in line:
                    # Escribir en diccionario del archivo json
                    if sentence != '' and len(aspects) != 0:
                        json_data['opinions'].append({'sentence': sentence,
                                                    'aspects': aspects})

                        sentence = ''
                        aspects = []

                    sentence = GetSentence(line)
                
                elif '<aspectCategory' in line:
                    category, polarity = GetCategoryPolarity(line)
                    aspects.append({'category': category,
                                    'polarity': polarity})
            
            # Escribir un nuevo archivo json
            with open(path[:len(path)-4] + '.json', 'w') as json_file:
                json.dump(json_data, json_file)
                json_file.close()
            
            xml_file.close()

# ======================================================================

if __name__ == '__main__':
    main()
