from sklearn import metrics
from sklearn_crfsuite import CRF
from itertools import chain
from joblib import dump, load
from readdata import read_data

language = "Chinese"
sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]

sorted_labels_chn = [
    'O',
    'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
    , 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
    , 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
    , 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
    , 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
    , 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
    , 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
    , 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
    ]
train_file = f'./data/{language}/train.txt' 
val_file = f'./data/{language}/validation.txt' 
test_file = f'./data/{language}/test.txt' 

def word2features_eng(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'U00:[-2,0]': sent[i-2][0] if i > 1 else 'BOS',
        'U01:[-1,0]': sent[i-1][0] if i > 0 else 'BOS',
        'U02:[0,0]': word,
        'U03:[1,0]': sent[i+1][0] if i < len(sent)-1 else 'EOS',
        'U04:[2,0]': sent[i+2][0] if i < len(sent)-2 else 'EOS',
        'U05:[-2,0]/[-1,0]': f"{sent[i-2][0]}/{sent[i-1][0]}" if i > 1 else 'BOS',
        'U06:[-1,0]/[0,0]': f"{sent[i-1][0]}/{word}" if i > 0 else 'BOS',
        'U07:[-1,0]/[1,0]': f"{sent[i-1][0]}/{sent[i+1][0]}" if i < len(sent)-1 and i > 0 else 'BOS',
        'U08:[0,0]/[1,0]': f"{word}/{sent[i+1][0]}" if i < len(sent)-1 else 'EOS',
        'U09:[1,0]/[2,0]': f"{sent[i+1][0]}/{sent[i+2][0]}" if i < len(sent)-2 else 'EOS'
    }

    return features

def word2features_chn(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.isdigit()': word.isdigit(),
        'U00:%x[-2,0]': sent[i-2][0] if i > 1 else 'BOS',
        'U01:%x[-1,0]': sent[i-1][0] if i > 0 else 'BOS',
        'U02:%x[0,0]': word,
        'U03:%x[1,0]': sent[i+1][0] if i < len(sent)-1 else 'EOS',
        'U04:%x[2,0]': sent[i+2][0] if i < len(sent)-2 else 'EOS',
        'U05:%x[-2,0]/%x[-1,0]': f"{sent[i-2][0]}/{sent[i-1][0]}" if i > 1 else 'BOS',
        'U06:%x[-1,0]/%x[0,0]': f"{sent[i-1][0]}/{word}" if i > 0 else 'BOS',
        'U07:%x[-1,0]/%x[1,0]': f"{sent[i-1][0]}/{sent[i+1][0]}" if i < len(sent)-1 and i > 0 else 'BOS',
        'U08:%x[0,0]/%x[1,0]': f"{word}/{sent[i+1][0]}" if i < len(sent)-1 else 'EOS',
        'U09:%x[1,0]/%x[2,0]': f"{sent[i+1][0]}/{sent[i+2][0]}" if i < len(sent)-2 else 'EOS'
    }

    return features


def extract_features(sent):
    if language == "English":
        return [word2features_eng(sent, i) for i in range(len(sent))]
    else:
        return [word2features_chn(sent, i) for i in range(len(sent))]

def get_labels(sent):
    return [label for token, label in sent]

def train(): 
    train_data = read_data(train_file)
    val_data = read_data(val_file)
    sort_labels = sorted_labels_eng if language == "English" else sorted_labels_chn

    X_train = [extract_features(sent) for sent in train_data]
    y_train = [get_labels(sent) for sent in train_data]

    X_test = [extract_features(sent) for sent in val_data]
    y_test = [get_labels(sent) for sent in val_data]

    # Create and train the CRF model
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1)
    crf.fit(X_train, y_train, X_test, y_test)

    # Make predictions on the test data
    y_pred = crf.predict(X_test)
    y_pred = list(chain(*y_pred))
    y_test = list(chain(*y_test))
    # Evaluate the model's performance
    report = metrics.classification_report(
        y_true = y_test, y_pred=y_pred, labels=sort_labels[1:], digits=4
    )
    print(report)

    # Save the trained CRF model
    dump(crf, f'./ckpt/crf_model-{language}-v2.joblib')

def test():
    crf = load(f'./ckpt/crf_model-{language}-v2.joblib')

    test_data = read_data(test_file)

    X_test = [extract_features(sent) for sent in test_data]
    y_test = [get_labels(sent) for sent in test_data]
    y_pred = crf.predict(X_test)
    y_pred = list(chain(*y_pred))
    y_test = list(chain(*y_test))
    # Evaluate the model's performance
    sort_labels = sorted_labels_eng if language == "English" else sorted_labels_chn
    report = metrics.classification_report(
        y_true = y_test, y_pred=y_pred, labels=sort_labels[1:], digits=4
    )
    print(report)

if __name__ == "__main__":
    train()
    test()