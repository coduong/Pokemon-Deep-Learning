import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

class ActionClassifier(nn.Module) :
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0) :
        super(ActionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths) :
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to the output of the LSTM
        output = self.fc(lstm_out)  # Pass through the fully connected layer
        return output    
    
class MoveClassifier(nn.Module) :
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0) :
        super(MoveClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim) # You added an fc2

    def forward(self, x, lengths) :
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        currStep = self.dropout(self.fc1(lstm_out)) 
        currStep = torch.relu(currStep) # This step is new
        output = self.fc2(currStep) # This step is new
        return output

class Pokemon_AI :
    def __init__(self) :
        self.ActionClassifier = torch.load("C:/Users/coliv/summerProjects/Summer-Repository/actionClassifier.pt")
        self.MoveClassifier = torch.load("C:/Users/coliv/summerProjects/Summer-Repository/moveClassifier.pt")
        self.battle_length = 0
        self.memory = np.full((99, 216), -1)
        self.pokemonMap = pd.read_csv("C:/Users/coliv/summerProjects/Summer-Repository/monDict.csv")
        self.moveMap = pd.read_csv("C:/Users/coliv/summerProjects/Summer-Repository/index_to_move.csv")
        self.pokemonEmbeddingMap = pd.read_csv("C:/Users/coliv/summerProjects/Summer-Repository/CollinPokemonEmbeddings.csv")
        self.moveEmbeddingMap = pd.read_csv("C:/Users/coliv/summerProjects/Summer-Repository/moveEmbeddings.csv")
        self.possibilities = pd.read_csv("C:/Users/coliv/summerProjects/Summer-Repository/moveset_dictionary.csv").drop("Unnamed: 0", axis = 1).set_index("name").T
        self.statusMap = {"burn" : 0, "freeze" : 1, "paralysis" : 2, "poison" : 3, "toxic" : 4, "sleep" : 5}

    def update_memory(self, vector) :
        self.memory[self.battle_length] = vector
        self.battle_length += 1
        return
    
    def reset_memory(self) :
        self.battle_length = 0
        self.memory = np.full((99, 216), -1)
        return
    
    def predict(self, pokemon) :
        with torch.no_grad() :
            input_tensor = torch.FloatTensor(self.memory).unsqueeze(0)
            length_tensor = torch.IntTensor(np.array([self.battle_length]))
            output = self.ActionClassifier(input_tensor, length_tensor)
            output = output[0]
            output = output[-1]
            print(output)
            output = torch.softmax(output, 0)
            action = np.argmax(output.numpy())
            if action == 1 :
                return "I think your opponent might switch their Pokemon."
            else :
                with torch.no_grad() :
                    pokemon_possibilities = torch.FloatTensor(self.possibilities[pokemon].to_numpy())
                    output = self.MoveClassifier(input_tensor, length_tensor)
                    output = output[0]
                    output = output[-1]
                    #don't forget your filter. 
                    output = torch.mul(output, pokemon_possibilities)
                    output = torch.softmax(output, 0)
                    move = np.argsort(output.numpy(), axis = 0)[:3]

                    guess1 = self.moveMap[str(move[0])][0]
                    guess2 = self.moveMap[str(move[1])][0]
                    guess3 = self.moveMap[str(move[2])][0]
                    return f"I think your opponent might use {guess1}, {guess2}, or {guess3}."
    
    
    def query(self) :
        print("What Pokemon are you currently using?")
        your_mon = input()
        your_mon = your_mon.lower()
        if your_mon == "farfetchd" or your_mon == "farfetch'd" :
            your_mon = "farfetch’d"
        assert your_mon in self.pokemonMap.columns, "You must input a valid Pokemon."
        your_mon = self.pokemonEmbeddingMap[your_mon]
        print("Did you use a move last turn?")
        moveOrNot = input()
        assert moveOrNot in set(["yes", "Yes", "no", "No"]), "must be a yes or no answer."
        if moveOrNot == "yes" or moveOrNot == "Yes" :
            print("What was your last move?")
            your_move = input()
            your_move = your_move.lower().replace(" ", "-")
            your_move = self.moveEmbeddingMap[your_move]
        else:
            your_move = np.zeros(32)
        print("Does your Pokemon have a status condition?")
        status_or_not = input()
        assert status_or_not in set(["yes", "Yes", "no", "No"]), "must be a yes or no answer."
        if status_or_not == "yes" or status_or_not == "Yes" :
            print("What status?")
            your_status = input()
            your_status = your_status.lower()
            status_vector = np.zeros(6)
            status_vector[self.statusMap.get(your_status)] = 1
            your_status = status_vector
        else:
            your_status = np.zeros(6)
        print("How much health does your Pokemon have as a proportion of its total health? Please enter a decimal.")
        your_hp = input()
        your_hp = np.array([float(your_hp)])

        print("Who is your opponent?")
        opponent = input()
        opponent = opponent.lower()
        if opponent == "farfetchd" or opponent == "farfetch'd" :
            opponent = "farfetch’d"
        pokemon_helper = opponent
        opponent = self.pokemonEmbeddingMap[opponent]
        print("Did your opponent make a move last turn?")
        moveOrNot = input()
        assert moveOrNot in set(["yes", "Yes", "no", "No"]), "must be a yes or no answer."
        if moveOrNot == "yes" or moveOrNot == "Yes" :
            print("What was their last move?")
            their_move = input()
            their_move = their_move.lower().replace(" ", "-")
            their_move = self.moveEmbeddingMap[their_move]
        else :
            their_move = np.zeros(32)
        print("Does your opponent have a status condition?")
        status_or_not = input()
        if status_or_not == "yes" or status_or_not == "Yes" :
            print("Which status?")
            their_status = input()
            their_status = their_status.lower()
            status_vector = np.zeros(6)
            status_vector[self.statusMap.get(their_status)] = 1
            their_status = status_vector
        else :
            their_status = np.zeros(6)
        print("How much health does your opponent have as a proportion of their total health? Please enter a decimal.")
        their_hp = input()
        their_hp = np.array([float(their_hp)])

        print("Are there any stat modifiers active on the field?")
        stats_or_not = input()
        assert stats_or_not in set(["yes", "Yes", "no", "No"]), "must be a yes or no answer."
        if stats_or_not == "yes" or stats_or_not == "Yes" :
            print("Please enter a list formatted like the following: your_attack, opponent's_attack, your_defense, opponent's_defense, your_spatk, opponent's_spatk, your_spdef, opponent's_spdef, your_spd, opponent's_spd")
            stats = input()
            stats = np.array(stats.split(", "))
            stats.as_type(float)
        else :
            stats = np.zeros(10)
        
        thisTurnInput = np.concatenate([your_mon, opponent, your_status, their_status, your_move, their_move, your_hp, their_hp, stats])

        self.update_memory(thisTurnInput)

        return pokemon_helper

def main() :
    pokemonAI = Pokemon_AI()
    alive = True
    while alive == True :
        print("Choose your following action: [predict, shutoff]")
        status = input()
        assert status in set(["shutoff", "predict"])
        if status == "shutoff" :
            pokemonAI.reset_memory()
            alive = False

        if status == "predict" :
            query = pokemonAI.query()
            prediction = pokemonAI.predict(query)
            print(prediction)
    return
        

main()