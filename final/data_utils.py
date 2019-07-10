import json
import torch

class EmbeddingGlove:
    def __init__(self, path):
        self.path = path
        self.vocab_size = 0
        self.embedding_dim = 0
        # Importante, el vector de ceros no esta en dictionary, solo en matrix
        self.dictionary = {}
        self.matrix = []

        self.OpenFile()
    
    def OpenFile(self):
        with open(self.path, 'r', encoding='utf8') as embedding_file:
            lines = embedding_file.readlines()
            
            for line in lines:
                words = line.split()
                
                word = words[0]
                vector = words[1:]
                self.embedding_dim = len(vector)
                self.AddWordVector(word, vector)
        
        # El ultimo es un vector de ceros para rellenar vectores
        vector_zeros = [0.0 for i in range(self.embedding_dim)]
        self.matrix.append(vector_zeros)

    def AddWordVector(self, word, vector):
        float_vector = [float(number) for number in vector]

        self.matrix.append(float_vector)
        self.dictionary[word] = {'position': self.vocab_size,
                                 'vector': float_vector}
        
        self.vocab_size += 1
    
    def GetPositionVector(self, word):
        if word in self.dictionary:
            position = self.dictionary[word]['position']
            vector = self.dictionary[word]['vector']
            
            return position, vector
        
        else:
            # Si no se ha encontrado la palabra, se retorna el vector de ceros
            position = len(self.matrix)-1
            vector = self.matrix[len(self.matrix)-1]

            return position, vector
    
    def WordsToIds(self, sentence, length):
        words = sentence.split()
        ids = []

        for word in words:
            id, _ = self.GetPositionVector(word)
            ids.append(id)
        
        # Se rellenar el vector de posciones a al tamaño estandar (length)
        if len(ids) < length:
            pos_vector_zeros = len(self.matrix)-1
            difference = length - len(ids)

            for i in range(difference):
                ids.append(pos_vector_zeros)
        
        # Si el tamaño de la oracion es mayor que el tamaño estandar (length), se recorta
        elif len(ids) > length:
            ids = ids[:length]

        return ids
    
    def PolarityToInt(self, polarity):
        if polarity == 'negative':
            return 0
        elif polarity == 'neutral':
            return 1
        elif polarity == 'positive':
            return 2
        else:
            return 1

# ======================================================================

class Metadata:
    def __init__(self):
        self.num_sentences = 0
        self.max_sentence_len = 0
        
        self.num_aspects = 0
        self.max_aspect_len = 0
        
        self.aspects = [{'category': 'food', 'positive': 0, 'negative': 0, 'neutral': 0}, 
                        {'category': 'price', 'positive': 0, 'negative': 0, 'neutral': 0},
                        {'category': 'service', 'positive': 0, 'negative': 0, 'neutral': 0},
                        {'category': 'ambience', 'positive': 0, 'negative': 0, 'neutral': 0},
                        {'category': 'anecdotes miscellaneous', 'positive': 0, 'negative': 0, 'neutral': 0}]
    
    # Contar el numero de oraciones y detectar el tamaño de la oracion mas larga
    def RegisterSentence(self, sentence):
        self.num_sentences += 1
        words = sentence.split()

        if len(words) > self.max_sentence_len:
            self.max_sentence_len = len(words)

    # Contar el numero de aspectos y detectar el tamaño del aspecto mas largo
    # Contar cuantos aspectos poseen las polaridades: positivo, negativo y neutral
    def RegisterCategoryPolarity(self, category, polarity):
        self.num_aspects += 1
        words = category.split()

        if len(words) > self.max_aspect_len:
            self.max_aspect_len = len(words)

        for aspect in self.aspects:
            if aspect['category'] == category:
                if polarity == 'positive':
                    aspect['positive'] += 1
                elif polarity == 'negative':
                    aspect['negative'] += 1
                elif polarity == 'neutral':
                    aspect['neutral'] += 1
    
    def PrintMetadata(self):
        print('num sentences: ' + str(self.num_sentences))
        print('max sentence len: ' + str(self.max_sentence_len))
        print('category: food')
        print('    positive: ' + str(self.aspects[0]['positive']))
        print('    negative: ' + str(self.aspects[0]['negative']))
        print('    neutral : ' + str(self.aspects[0]['neutral']))
        print('category: price')
        print('    positive: ' + str(self.aspects[1]['positive']))
        print('    negative: ' + str(self.aspects[1]['negative']))
        print('    neutral : ' + str(self.aspects[1]['neutral']))
        print('category: service')
        print('    positive: ' + str(self.aspects[2]['positive']))
        print('    negative: ' + str(self.aspects[2]['negative']))
        print('    neutral : ' + str(self.aspects[2]['neutral']))
        print('category: ambience')
        print('    positive: ' + str(self.aspects[3]['positive']))
        print('    negative: ' + str(self.aspects[3]['negative']))
        print('    neutral : ' + str(self.aspects[3]['neutral']))
        print('category: anecdotes miscellaneous')
        print('    positive: ' + str(self.aspects[4]['positive']))
        print('    negative: ' + str(self.aspects[4]['negative']))
        print('    neutral : ' + str(self.aspects[4]['neutral']) + '\n')

# ======================================================================

class Dataset:
    def __init__(self, path, embedding):
        self.path = path
        self.metadata = Metadata()
        self.embedding = embedding
        self.opinions = []
        
        self.OpenFile()
    
    def OpenFile(self):
        print("open: " + self.path)

        with open(self.path, 'r') as dataset_file:
            data = json.load(dataset_file)

            for opinion in data['opinions']:
                sentence = opinion['sentence']
                
                self.metadata.RegisterSentence(sentence)

                for aspect in opinion['aspects']:
                    category = aspect['category']
                    polarity = aspect['polarity']

                    self.AddOpinion(sentence, category, polarity)
                    self.metadata.RegisterCategoryPolarity(category, polarity)
            
            dataset_file.close()
        
        self.metadata.PrintMetadata()
    
    def AddOpinion(self, sentence, category, polarity):
        self.opinions.append({'sentence': sentence,
                              'category': category,
                              'polarity': polarity})

    def GenerateBatches(self, batch_size, max_sentence_len, max_aspect_len):
        # Vector con todos los batches
        batches_sentences = []
        batches_aspects = []
        batches_polarities = []

        # Vector de un batch
        batch_sentences = []
        batch_aspects = []
        batch_polarities = []

        save_batch = batch_size

        for i in range(len(self.opinions)):
            sentence = self.opinions[i]['sentence']
            category = self.opinions[i]['category']
            polarity = self.opinions[i]['polarity']

            # Transformando en vectores los elementos de la opinion
            vector_sentence = self.embedding.WordsToIds(sentence, max_sentence_len)
            vector_category = self.embedding.WordsToIds(category, max_aspect_len)
            int_polarity = self.embedding.PolarityToInt(polarity)

            if i == save_batch:
                # Guardando el batch en la lista de batches
                batches_sentences.append(batch_sentences)
                batches_aspects.append(batch_aspects)
                batches_polarities.append(batch_polarities)

                # Limpiando el vector de batch para guardar otro batch
                batch_sentences = []
                batch_aspects = []
                batch_polarities = []

                # Colocando los elementos de la iteracion actual en el nuevo batch
                batch_sentences.append(vector_sentence)
                batch_aspects.append(vector_category)
                batch_polarities.append(int_polarity)

                # Cuando guardar el siguiente batch
                save_batch += batch_size
            
            else:
                # Colocando los elementos de la iteracion actual en el batch actual
                batch_sentences.append(vector_sentence)
                batch_aspects.append(vector_category)
                batch_polarities.append(int_polarity)
            
            # Verificando si hay un batch que falta guardar
            if i == len(self.opinions)-1 and len(batch_sentences) != 0:
                batches_sentences.append(batch_sentences)
                batches_aspects.append(batch_aspects)
                batches_polarities.append(batch_polarities)
        
        return batches_sentences, batches_aspects, batches_polarities