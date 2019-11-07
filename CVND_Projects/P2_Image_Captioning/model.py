import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #super().__init__()

        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #Initialize the layers of this model
        #super(DecoderRNN, self).__init__()
        super().__init__()
        
        # define the properties
        self.hidden_size = hidden_size # Keep track of hidden_size for initialization of hidden state
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        #self.dropout = nn.Dropout(drop_prob)

         # Embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states of size hidden_size.
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # The linear layer that maps the hidden state output dimension to the number of words we want as output(vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size) 
        
        # Initialize the hidden state
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
           
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """        
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]
        
        # Create embedded word vectors for each word in the caption
        embeddings = self.embed(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)

         #Concatenating features and captions together. Features are always the first vectors after concatenation.
         #So we can always feed features to LSTM layer first then follow by word after word in each sequence of captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings) 
        #lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs   
    
    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for i in range(max_len):
            # the state of the lstm is changing so keep track
            outputs, hidden = self.lstm(inputs, hidden)   # lstm_out shape : (1, 1, hidden_size)
#             print('lstm output shape ', outputs.shape)
#             print('lstm output.squeeze(1) shape ', outputs.squeeze(1).shape)
            # convert LSTM output to word predictions
            outputs = self.linear(outputs.squeeze(1)) # outputs shape : (1, 1, vocab_size), after squeezing, outputs shape : (1, vocab_size)
#             print('linear output shape ', outputs.shape)
            target_index = outputs.max(1)[1]   # predict the most likely next word
#             print('target_index shape ', target_index.shape)
            sentence.append(target_index.item())  # storing the word predicted  
            # We predicted the <end> word, so there is no further prediction to do
            if (target_index == 1):
                break
                
            ## Prepare to embed the last predicted word to be the new input of the lstm     
            inputs = self.embed(target_index) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
#             print('new inputs shape ', inputs.shape, '\n')
        return sentence
    
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
#         return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
#                 weight.new(self.num_layers, batch_size, self.hidden_size).zero_()) 

# #     def sample(self, inputs, states=None, max_len=20):
# #         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
# #         predicted_caption_idx = []
        
# #         #loop over max_len
# #         for i in range(max_len):
            
# #             #inputs had been embedded
# #             out, states = self.lstm(inputs, states)
# #             fc_out = self.linear(out)
# #             _, pred_idx = torch.max(fc_out, dim=2) #shape of pred_idx:[1,1]
            
# #             predicted_caption_idx.append(torch.squeeze(pred_idx).item())
            
# #             #re-calculating inputs as input for next LSTM layer
# #             inputs = self.embed(pred_idx)#.unsqueeze(1)
        
# #         return predicted_caption_idx
    
    
#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         output = []
#         batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
#         hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
    
#         while True:
#             #print("ENTERS")
#             lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
#             outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
#             outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
#             _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
                       
#             output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
#             if (max_indice == 1):
#                 # We predicted the <end> word, so there is no further prediction to do
#                 break
            
#             ## Prepare to embed the last predicted word to be the new input of the lstm
#             inputs = self.embed(max_indice) # inputs shape : (1, embed_size)
#             inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
#         return output
    
# # class DecoderRNN(nn.Module):
# #     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
# #         super(DecoderRNN, self).__init__()
        
# #         self.hidden_size = hidden_size
        
# #         self.word_embeddings = nn.Embedding(vocab_size, embed_size)
# #         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
# #         self.fc = nn.Linear(hidden_size, vocab_size)
    
# #     def forward(self, features, captions):
        
# #         #remove ending token (batch_size, caption_length - 1)
# #         captions = captions[:,:-1]
        
# #         #embed captions
# #         caption_embeds = self.word_embeddings(captions)
        
# #         '''
# #         Concatenating features and captions together. Features are always the first vectors after concatenation.
# #         So we can always feed features to LSTM layer first then follow by word after word in each sequence of captions
# #         '''
# #         inputs = torch.cat([features.unsqueeze_(1), caption_embeds], dim=1)
        
# #         #inputs.size()[1]： feature vector + caption length
# #         hidden = (torch.randn(1,inputs.size()[1],self.hidden_size).cuda(), torch.randn(1,inputs.size()[1],self.hidden_size).cuda())
        
# #         lstm_out, hidden = self.lstm(inputs, hidden)
        
# #         scores = self.fc(lstm_out)
        
# #         return scores

#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
#         predicted_caption_idx = []
        
#         #loop over max_len
#         for i in range(max_len):
            
#             #inputs had been embedded
#             out, states = self.lstm(inputs, states)
#             fc_out = self.linear(out)
#             _, pred_idx = torch.max(fc_out, dim=2) #shape of pred_idx:[1,1]
            
#             predicted_caption_idx.append(torch.squeeze(pred_idx).item())
            
#             #re-calculating inputs as input for next LSTM layer
#             inputs = self.embed(pred_idx)#.unsqueeze(1)
        
#         return predicted_caption_idx