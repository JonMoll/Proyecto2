import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
from data_utils import EmbeddingGlove, Dataset
from models import AOA

def ProbabilitiesToInt(vector_probabilities):
    max_probability = vector_probabilities[0]
    max_index = 0

    for i in range(len(vector_probabilities)):
        if vector_probabilities[i] > max_probability:
            max_probability = vector_probabilities[i]
            max_index = i
    
    return max_index

# ======================================================================

def CalculateAcuracyF1(model, device, batches_sentences, batches_aspects, batches_polarities):
    indices_failures = []

    y_pred = []
    y_true = []

    with torch.no_grad(): # En la prueba el modelo no debe aprender
        for i in range(len(batches_sentences)):
            batch_sentences = batches_sentences[i]
            batch_sentences = torch.tensor(batch_sentences, dtype=torch.long).to(device)

            batch_aspects = batches_aspects[i]
            batch_aspects = torch.tensor(batch_aspects, dtype=torch.long).to(device)

            batch_polarities = batches_polarities[i]

            prediction, _ = model(batch_sentences, batch_aspects)

            for j in range(len(prediction)):
                index_prediction = ProbabilitiesToInt(prediction[j])
                y_pred.append(index_prediction)
                y_true.append(batch_polarities[j])
    
    hits = 0
    failures = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            hits += 1
        else:
            failures += 1
            indices_failures.append(i)
    
    total = hits + failures
    acuracy = hits / total

    f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')

    return acuracy, f1, indices_failures

# ======================================================================

def main():
    path_embedding_glove = './glove.6B.100d.txt'
    path_dataset_train = './datasets/restaurants_train.json'
    path_dataset_trial = './datasets/restaurants_trial.json'
    path_study_cases = './datasets/study_cases.json'
    path_saved = './saved_aoa/'
    path_log = './log_aoa.txt'

    embedding = EmbeddingGlove(path_embedding_glove)
    dataset_train = Dataset(path_dataset_train, embedding)
    dataset_trial = Dataset(path_dataset_trial, embedding)
    study_cases = Dataset(path_study_cases, embedding)

    max_sentence_len_train = dataset_train.metadata.max_sentence_len
    max_sentence_len_trial = dataset_trial.metadata.max_sentence_len

    max_aspect_len_train = dataset_train.metadata.max_aspect_len
    max_aspect_len_trial = dataset_trial.metadata.max_aspect_len

# ======================================================================

    embedding_matrix = torch.tensor(embedding.matrix, dtype=torch.float)
    embedding_dim = embedding.embedding_dim
    hidden_dim = 150
    polarity_dim = 3
    batch_size = 40
    max_sentence_len = max(max_sentence_len_train, 
                           max_sentence_len_trial)
    max_aspect_len = max(max_aspect_len_train,
                         max_aspect_len_trial)
    epochs = 40
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('embedding_dim: ' + str(embedding_dim))
    print('hidden_dim: ' + str(hidden_dim))
    print('polarity_dim: ' + str(polarity_dim))
    print('batch_size: ' + str(batch_size))
    print('max_sentence_len: ' + str(max_sentence_len))
    print('max_aspect_len: ' + str(max_aspect_len))
    print('epochs: ' + str(epochs))
    print('device: ' + str(device))

# ======================================================================

    batches_train_sentences, batches_train_aspects, batches_train_polarities = dataset_train.GenerateBatches(batch_size, 
                                                                                                             max_sentence_len, 
                                                                                                             max_aspect_len)
    
    batches_trial_sentences, batches_trial_aspects, batches_trial_polarities = dataset_trial.GenerateBatches(batch_size, 
                                                                                                             max_sentence_len, 
                                                                                                             max_aspect_len)

    study_cases_sentences, study_cases_aspects, study_cases_polarities = study_cases.GenerateBatches(batch_size, 
                                                                                                     max_sentence_len, 
                                                                                                     max_aspect_len)

    num_batches = len(batches_train_sentences)

# ======================================================================

    model = AOA(embedding_matrix, embedding_dim, hidden_dim, polarity_dim)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======================================================================
    
    train = False
    file_name_saved = 'aoa_epoch39_acuracy1.0'

    if train:
        file_log = open(path_log, 'w')
        max_acuracy = 0.0

        for epoch in range(epochs):
            print('\n========== Epoch ' + str(epoch) + ' ==========')

            model.train()

            for i in range(num_batches):
                optimizer.zero_grad()

                batch_sentences = batches_train_sentences[i]
                batch_sentences = torch.tensor(batch_sentences, dtype=torch.long).to(device)

                batch_aspects = batches_train_aspects[i]
                batch_aspects = torch.tensor(batch_aspects, dtype=torch.long).to(device)

                batch_polarities = batches_train_polarities[i]
                batch_polarities = torch.tensor(batch_polarities, dtype=torch.long).to(device)

                prediction, _ = model(batch_sentences, batch_aspects)

                loss = loss_function(prediction, batch_polarities)
                loss.backward()
                optimizer.step()
            
            acuracy, f1, _ = CalculateAcuracyF1(model, device, batches_train_sentences, batches_train_aspects, batches_train_polarities)
            
            print('acuracy train: ' + str(acuracy))
            print('f1 train: ' + str(f1))
            
            file_log.write('epoch: ' + str(epoch) + '\n')
            file_log.write('acuracy_train: ' + str(acuracy) + ' f1_train: ' + str(f1) + '\n')

            if acuracy >= max_acuracy:
                max_acuracy = acuracy
                file_name_saved = 'aoa_epoch' + str(epoch) + '_acuracy' + str(acuracy)
                torch.save(model.state_dict(), path_saved + file_name_saved)
                print('saved: ' + path_saved + file_name_saved)

        file_log.close()
    
    else:
        print('\n========== Load saved ==========')

        model.load_state_dict(torch.load(path_saved + file_name_saved))
        print('load: ' + path_saved + file_name_saved)

        acuracy, f1, _ = CalculateAcuracyF1(model, device, batches_train_sentences, batches_train_aspects, batches_train_polarities)
        print('acuracy train: ' + str(acuracy))
        print('f1 train: ' + str(f1))

# ======================================================================

    print('\n********** Trial dataset **********')
    
    acuracy, f1, indices_failures = CalculateAcuracyF1(model, device, batches_trial_sentences, batches_trial_aspects, batches_trial_polarities)
    print('acuracy trial: ' + str(acuracy))
    print('f1 trial: ' + str(f1))
    print('indices failures:')
    print(indices_failures)

    for index in indices_failures:
        print(dataset_trial.opinions[index])

# ======================================================================

    print('\n********** Study cases **********')
    
    with torch.no_grad():
        for i in range(len(study_cases_sentences)):
            batch_sentences = study_cases_sentences[i]
            batch_sentences = torch.tensor(batch_sentences, dtype=torch.long).to(device)

            batch_aspects = study_cases_aspects[i]
            batch_aspects = torch.tensor(batch_aspects, dtype=torch.long).to(device)

            batch_polarities = study_cases_polarities[i]

            prediction, attention = model(batch_sentences, batch_aspects)
            
            print('Sentences: ')
            print(batch_sentences)
            print('Aspects: ')
            print(batch_aspects)
            print('Polarities: ')
            print(batch_polarities)
            print('Prediction: ')
            print(prediction)
            print('Attention: ')
            print(attention.squeeze(-1))

# ======================================================================

if __name__ == '__main__':
    main()
