import numpy as np
from sklearn.metrics import classification_report
from readdata import read_data

class HMM:
    def __init__(self, unique_states):
        self.states = unique_states
        self.N = len(unique_states) # Number of states
        self.A = np.full((self.N, self.N), 1e-8) # Transition Probabilities
        self.B = {} # Emission Probabilities
        self.Pi = np.full(self.N, 1e-8) # Initial Probabilities

    def train(self, train_data):
        # Counting occurrences
        for sentence in train_data:
            for i in range(len(sentence)):
                word, state = sentence[i]
                state_index = self.states.index(state)

                self.B[word] = self.B.get(word, np.zeros(self.N))
                self.B[word][state_index] += 1

                if i == 0: # First word in sentence
                    self.Pi[state_index] += 1
                else: # Following words
                    prev_state = sentence[i-1][1]
                    prev_state_index = self.states.index(prev_state)
                    self.A[prev_state_index][state_index] += 1

        # Normalizing counts to probabilities
        self.A /= self.A.sum(axis=1, keepdims=True)
        for word in self.B:
            self.B[word] /= self.B[word].sum()
        self.Pi /= self.Pi.sum()

    def predict(self, test_data):
        y_true, y_pred = [], []

        for sentence in test_data:
            obs = [word for word, _ in sentence]
            states = [state for _, state in sentence]
            y_true.extend(states)

            V = np.zeros((self.N, len(obs))) 
            ptr = np.zeros((self.N, len(obs)), dtype=int)

            # Initialization
            V[:, 0] = self.Pi * self.B.get(obs[0], np.zeros(self.N))

            # Recursion
            for t in range(1, len(obs)):
                for s in range(self.N):
                    trans_p = V[:, t-1] * self.A[:, s]
                    ptr[s, t] = np.argmax(trans_p)
                    V[s, t] = np.max(trans_p) * self.B.get(obs[t], np.zeros(self.N))[s]

            # Backtracking
            pred_states = [np.argmax(V[:, -1])]
            for t in range(len(obs)-2, -1, -1):
                pred_states.insert(0, ptr[pred_states[0], t+1])

            y_pred.extend([self.states[i] for i in pred_states])

        return y_true, y_pred

    
if __name__ == '__main__':
    language = "English"
    sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
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
    test_file = f'./data/{language}/validation.txt'
    sorted_labels = sorted_labels_eng if language == "English" else sorted_labels_chn
    train_data = read_data(train_file)
    test_data = read_data(test_file)
    model = HMM(sorted_labels)
    model.train(train_data)
    y_true, y_pred = model.predict(test_data)

    print(classification_report(y_true, y_pred, labels=sorted_labels[1:], zero_division=0, digits=4))
