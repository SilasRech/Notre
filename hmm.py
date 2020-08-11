import numpy as np
import recognizer.tools as tools

# default HMM
WORDS = {
    'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'size': [1, 9, 15, 12, 6, 9, 9, 9, 12, 15, 9, 9],
    'gram': [100, 40, 100, 100, 100, 100, 100, 100, 100, 100, 80, 100],
}


class HMM:  

    words = {}

    def __init__(self, words=WORDS):
        """
        Constructor of HMM class. Inits with provided structure words
        :param input: word of the defined HMM.
        """
        self.words = words
        
        # number of all states
        num_states = sum(words['size'])
        
        # initialize pi vector without glue states
        pi = np.zeros(num_states)
        
        for index in range(len(words['name'])):
            pi[sum(words['size'][0:index+1])-words['size'][index]] = words['gram'][index]/sum(words['gram'])
        
        self.logPi = np.log(np.maximum(pi, 1e-100))
        
        # initialize transition matrix A
        # get states for every name
        states = {}
        state = 0
        for name in range(len(words['name'])):
            states[words['name'][name]] = np.arange(state,state+words['size'][name])
            state = state+words['size'][name]
            
        self.states = states
            
        # set elements on principle diagonal set 1
        ones = np.ones(num_states)
        # create matrix
        A = np.diag(ones)
        
        # set elements on secondary diagonal to 1
        for state in range(num_states-1):
            A[state,state+1] = 1
            
        # set transitions between words to 1
        # get start and end points of state vectors
        starts = []
        ends = []
        for name in states.keys():
            starts.append(states[name][0])
            ends.append(states[name][-1])
        
        
        for end in ends:
            for start in starts:
                A[end, start] = 1
                
        # normalize likelihoods
        for row in range(num_states):
            A[row,:] = A[row,:]/sum(A[row,:]) 
            
        self.logA = np.log(np.maximum(A, 1e-100))
        

    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.
        :return: number of states.
        """
        return sum(self.words['size'])

    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.
        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state) ]
      
      
    def getTranscription(self,stateSequence):
        """
        Returns the word sequence for a given state sequence
        :param stateSequence: state sequence
        :return: transcription
        """
          
        words = []
        
        # get states from initialization
        states_init = self.states
        
        # iterate over all states in stateSequence
        for state in range(len(stateSequence)):
            # find state in initialization states
            for state_init in states_init:
                if stateSequence[state] in states_init[state_init]:
                    # get associated word
                    word = state_init
                    break            
            # start with first word
            if words == []:
                words.append(word)
            # check if still in current word or if there is a new word
            elif words[-1] != word or stateSequence[state] < stateSequence[state-1]:
                words.append(word)
                
        # delete silence
        while 'sil' in words:
            words.remove('sil')
            
        # change to upper case
        for index in range(len(words)):
            words[index] = words[index].upper()
        
        return words
      
      
    def posteriors_to_transcription(self, posteriors):
        """
        Returns a transcription  for a given posteriors matrix
        :param posteriors: non-log posteriors matrix
        :return: transcription
        """ 
        
        logPosteriors = tools.limLog(posteriors)
        
        stateSequence, pStar = tools.viterbi(logPosteriors, self.logPi, self.logA)
        words = self.getTranscription(stateSequence)
        
        return words
